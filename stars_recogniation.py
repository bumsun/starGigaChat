

from common_points import getCommonPoints
import star_names

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import figure
from skyfield.api import Star, load
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.data import hipparcos, mpc, stellarium
from skyfield.projections import build_stereographic_projection
from skyfield.toposlib import Topos
from skyfield.api import N, W, wgs84
from numpy import random
from scipy.spatial import distance

# from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import warnings
import torch
from torch import nn
import cv2
import cv2 as cv
import requests
from PIL import Image
from io import BytesIO
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors.nearest_centroid import NearestCentroid
import torchvision.models as models
import math
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
np.set_printoptions(suppress=True)

from scipy import signal
from scipy import misc



# import common_points as icp

input_size = 256
output_size = 129
thresholdNearestStars = 40

df_names = star_names.get_star_names()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
eph = load('de421.bsp')
with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

url = ('https://raw.githubusercontent.com/Stellarium/stellarium/master'
       '/skycultures/western_SnT/constellationship.fab')

with load.open(url) as f:
    constellations = stellarium.parse_constellations(f)

def normalizeImage(im):
    newImage = im.copy()
    newImageBinar = im.copy()
    print("im shape:",im.shape)
    tileRow = math.trunc(im.shape[0]/20)
    tileCol = math.trunc(im.shape[1]/20)
    y=0
    while y < im.shape[0]-1:
        x = 0
        while x < im.shape[1]-1:
            height = tileRow
            width = tileCol
            if y + tileRow > im.shape[0]:
                height = im.shape[0] - y - 1
            if x + tileCol > im.shape[1]:
                width = im.shape[1] - x - 1
            # print("x:",x," y:",y)
            # print("width:",width," height:",height)
            tileImage = im[y:y+height,x:x+width]
            counts, bins = np.histogram(tileImage, range(257))
            # print("counts:",counts)
            treshIndex = np.argmax(counts)
            # print("treshIndex:",treshIndex)

            M = np.full(tileImage.shape, treshIndex, dtype="uint8")
            # M255 = np.full(tileImage.shape, 255, dtype="uint8")
            tileImage = cv2.subtract(tileImage, M)
            tileImage = tileImage.clip(min=0)
            # print("M255/np.max(tileImage):",M255/np.max(tileImage))
            # tileImage = tileImage*(M255/np.max(tileImage))
            # counts, bins = np.histogram(tileImage, range(257))
            newImage[y:y+height,x:x+width] = tileImage
            newImageBinar
            x += tileCol
            
        y += tileRow
    return newImage

def image_to_binary(im, threshold_grad):
    scharr = np.array([[1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1],
                      [1,1,1,1,1]]) # Gx + j*Gy                 
    grad = signal.convolve2d(im, scharr, boundary='symm', mode='same')
    normal_grad = grad/np.amax(grad)
    bool_grad = normal_grad > threshold_grad #0.03
    bool_grad = bool_grad.astype(int)
    white_grad = bool_grad*255
    white_grad = white_grad.astype(int)
    # imgplot = plt.imshow(white_grad, cmap="gray")
    # plt.show()
    return white_grad

def reduce_tail(l, sizes, index):
    elm = l[index]
    mask = np.linalg.norm(elm-l, axis=1) > thresholdNearestStars
    ind_max_size = np.argmax(sizes[~mask])    
    sizes[index] = sizes[~mask][ind_max_size]
    l[index] = l[~mask][ind_max_size]
    mask[:index+1] = True  #ensure to return the head of the array unchanged
    
    return l[mask], sizes[mask]

def my_reduce(z,sizes):
    z = np.array(z)
    s = np.array(sizes)
    index = 0
    while True:
        z,s = reduce_tail(z,s, index)
        index += 1
        if index == z.shape[0]:
            break
    return z, s

def image_find_blobs(im, thresholdNearestStarsLocal = None):
    # im = normalizeImage(im)
    
    count_points = 0
    threshold_grad = 0.10
    attemp_count = 0
    while count_points < 40 or count_points > 110:
            attemp_count = attemp_count + 1
            im_bin = image_to_binary(im,threshold_grad)
            im_bin = cv2.convertScaleAbs(im_bin)
            params = cv2.SimpleBlobDetector_Params()



            params.blobColor = 255
            params.filterByColor = True
            params.filterByArea = True
            params.minDistBetweenBlobs = 1
            params.minArea = 0
            params.maxArea = 3000
            # enabling these can cause us to miss points
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(im_bin)

            count_points = len(keypoints)

            if count_points < 40:
                threshold_grad -= 0.01
            else:
                threshold_grad += 0.01
            print("threshold_grad:",threshold_grad)
            print("count_points:",count_points)
            if threshold_grad < 0.01:
                return None, None
            if attemp_count > 10:
                return None, None

    if count_points > 300:
        return None, None

    print("count_points:",count_points)
    
    pts = cv2.KeyPoint_convert(keypoints)
    sizes = np.array([p.size for p in keypoints])
    
    pts, sizes = my_reduce(pts,sizes)
    pointsPhoto = np.column_stack( (pts,sizes) )
    print("pts:",pts.shape)
    print("sizes:",sizes)

    pts_temp = pts.astype(int)
    sizes_temp = sizes.astype(int)
    for i in range(len(pts)):
        im_with_keypoints = cv2.circle(im, tuple(pts_temp[i]), radius=sizes_temp[i], color=(0, 255, 0), thickness=1)

    # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.rcParams['figure.dpi'] = 600
    imgplot = plt.imshow(im_with_keypoints)

    return pointsPhoto, im_with_keypoints

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
def getSpace(lat,lon,t,limiting_magnitude = 5,limiting_magnitude_big = 100, center_star=None):
    earth = eph['earth']
    space_position = (Topos(str(lat)+' N', str(lat)+' W', elevation_m=625 * 100000)).at(t)
 
    boston = earth + wgs84.latlon(lat * N, lon * W)
    projection = build_stereographic_projection(space_position)
    if center_star is not None:
        megrecObserve = boston.at(t).observe(center_star)
        projection = build_stereographic_projection(megrecObserve)
    
    field_of_view_degrees = 45.0
    star_positions = earth.at(t).observe(Star.from_dataframe(stars))
    stars['x'], stars['y'] = projection(star_positions)
    
    wrap_rate = 0.465
    
    stars['r'],stars['phi'] = cart2pol(stars['x'],stars['y'])
    # print("stars['phi'] min shape:",np.min(stars['phi']))
    # print("stars['phi'] max shape:",np.max(stars['phi']))
    stars['phi'] = stars['phi'] + random.uniform(-np.pi,np.pi)
    stars['r'] = (stars['r']) + wrap_rate * np.power(stars['r'], 2)
    stars['x'], stars['y'] = pol2cart(stars['r'],stars['phi'])
    
    # stars['x'] = np.sign(stars['x'])*(np.abs(stars['x']) + wrap_rate * np.power(stars['hypotenuse'], 2))
    # stars['y'] = np.sign(stars['y'])*(np.abs(stars['y']) + wrap_rate * np.power(stars['hypotenuse'], 2))
 
    pointsStars_df = stars
    pointsStars_df = pointsStars_df.loc[pointsStars_df['magnitude'] <= limiting_magnitude]
    pointsStars_df = pointsStars_df.join(df_names)
 
    pointsStars_df_big = stars
    pointsStars_df_big = pointsStars_df_big.loc[pointsStars_df_big['magnitude'] <= limiting_magnitude_big]
    pointsStars_df_big = pointsStars_df_big.join(df_names)
    return pointsStars_df, pointsStars_df_big

def init_nn():
    print("torch.cuda.is_available():",torch.cuda.is_available())

    skyfield_nn = models.vgg11(pretrained=True)
    for name, param in skyfield_nn.named_parameters():
        if("bn" not in name):
            param.requires_grad = True

    skyfield_nn.classifier = nn.Sequential(nn.Linear(skyfield_nn.classifier[0].in_features,1024),
        nn.ReLU(),                                 
         nn.Linear(1024,output_size),nn.Softmax())
    # Загружать веса с диска?
    is_load_weights = True

    if is_load_weights:
        current_path = os.path.abspath(os.getcwd())
        if "root" in current_path:
            skyfield_dict = torch.load(current_path+"/sky_field_nets/find_center_star_net2",map_location=device)
        else:
            skyfield_dict = torch.load("/content/gdrive/MyDrive/sky_field_nets/find_center_star_net2",map_location=device)
        skyfield_nn.load_state_dict(skyfield_dict,strict=True)
    return skyfield_nn


# Загрузка фотографии
def load_image(url):
    res = requests.get(url)
    if 'jpeg' in res.headers['content-type'] or 'png' in res.headers['content-type']:
        img_arr = np.array(Image.open(BytesIO(res.content)))
        return img_arr
    else:
        return None

def image_to_features(im):
    # im = cv2.imread("/content/IMG_20210913_233340.jpg", cv2.IMREAD_GRAYSCALE) # IMG_0077.jpg
    pointsPhoto, im_with_keypoints = image_find_blobs(im)
    if pointsPhoto is None:
        return None, None, None
    inputNP,reference_points,points_photo_origin = getDatasetFromDF(pointsPhoto, im.shape)

    count_stars = 43
    pointsPhoto = pointsPhoto[pointsPhoto[:, 2].argsort()[::-1]] # pointsPhoto[pointsPhoto[:, 2].argsort()] 
    pointsPhoto = pointsPhoto[:count_stars]

    im_copy = im.copy()
    for item in pointsPhoto.astype(int):
        im_copy = cv2.circle(im_copy,(item[0], item[1]), item[2], 255, thickness=4)
    print("im_copy")
    cv2.imwrite('/root/images/temp_key_points.jpg', im_copy)
    imgplot = plt.imshow(im_copy)
    plt.show()

    print("inputNP:",inputNP.shape)
    imgplot = plt.imshow(np.moveaxis(inputNP, 0, 2))
    plt.show()

    reference_points = (reference_points-np.min(reference_points))/(np.max(reference_points)-np.min(reference_points))
    print("points_photo_origin:",points_photo_origin)
    inputs = torch.from_numpy(inputNP).float().to(device).unsqueeze(0)

    return inputs,reference_points,points_photo_origin


skyfield_nn = init_nn()

def find_center_star(inputs):
    lat = 40
    lon = 40
    limiting_magnitude = 2.78
    limiting_big_magnitude = 3.9
    ts = load.timescale()
    t = ts.utc(2021, 4, 16)
    alioth_star = Star.from_dataframe(stars.loc[62956])
    vega_star = Star.from_dataframe(stars.loc[91262])

    pointsStars_df, pointsStars_df_big = getSpace(lat,lon,t,limiting_magnitude = limiting_magnitude,limiting_magnitude_big = 5,center_star=alioth_star)

    outputs = skyfield_nn(inputs)
    outputs = outputs[0].cpu().detach().numpy().reshape(-1)
    print("outputs:", outputs)
    outputIndex = np.argmax(outputs)
    print("outputIndex:", outputIndex)
    
    df_final = pointsStars_df.iloc[outputIndex]
    return df_final
def getDatasetFromDF(pointsPhoto, im_shape):
    limit_image = im_shape[0]/2
    if im_shape[1] > im_shape[0]:
        limit_image = im_shape[1]/2
    count_stars = 50
    pointsPhoto = pointsPhoto[pointsPhoto[:, 2].argsort()[::-1]] # pointsPhoto[pointsPhoto[:, 2].argsort()] 
    pointsPhoto = pointsPhoto[:count_stars]
    
    print("pointsPhoto test:",pointsPhoto)
    # pointsPhotoTemp = np.copy(pointsPhoto)
    # pointsPhotoTemp[:,1] = pointsPhoto[:,0]
    # pointsPhotoTemp[:,0] = pointsPhoto[:,1]
    pointsPhoto = np.copy(pointsPhoto)
    points_photo_origin = np.copy(pointsPhoto[:,:2])
    print("min pointsPhoto:",np.min(pointsPhoto[:,1]))

    pointsPhoto[:,0] = pointsPhoto[:,0]-round(np.max(pointsPhoto[:,0])/2)
    pointsPhoto[:,1] = pointsPhoto[:,1]-round(np.max(pointsPhoto[:,1])/2)

    #limit_image
    # pointsPhoto[:,:2] = pointsPhoto[:,:2]/limit_image
    pointsPhoto[:,:2] = (pointsPhoto[:,:2]+limit_image)/(limit_image+limit_image)
    print("pointsPhoto 1:",pointsPhoto)
    # pointsPhoto[:,:2] = (pointsPhoto[:,:2]-np.min(pointsPhoto[:,:2]))/(np.max(pointsPhoto[:,:2])-np.min(pointsPhoto[:,:2])) +0.00000001
    
    counter = np.array(range(pointsPhoto.shape[0]))
    magnitude = pointsPhoto.shape[0] - counter
    pointsPhoto[:,2] = magnitude
    pointsPhoto[:,2] = (pointsPhoto[:,2]-np.min(pointsPhoto[:,2]))/(np.max(pointsPhoto[:,2])-np.min(pointsPhoto[:,2])) #+0.000000001
    pointsPhoto[:,2] = np.round(pointsPhoto[:,2]*255)
    pointsPhoto[:,2] = pointsPhoto[:,2].astype(int)

    # pointsPhoto[:,2] = 255
    
    print("midle pointsPhoto:",pointsPhoto)
    
    # Загрузка валидацияонных данных из гипаркоса, закоменитть во время нормального теста
    # skyfield_dataset = SkyFieldDataset()
    # skyfield_dataset.updateState(is_validate=True)
    # print(skyfield_dataset[0][0].shape)
    # pointsPhoto = skyfield_dataset[0][0].reshape((32, 3)) 
    # print("end pointsPhoto:",pointsPhoto)
    # Загрузка валидацияонных данных из гипаркоса, закоменитть во время нормального теста

    inputImage = np.zeros([input_size,input_size])
    points = pointsPhoto[:,:2]*(input_size-1)
    points = points.astype(int)

    inputImage[points[:,1],points[:,0]] = pointsPhoto[:,2]
    inputImage = np.stack((inputImage,)*3, axis=0)

    print("max inputImage :",np.max(inputImage))
    imgplot = plt.imshow(np.moveaxis(inputImage, 0, 2))
    plt.show()
    return inputImage, points[:,:2],points_photo_origin
def find_stars_info(url = "https://psv4.userapi.com/c537232/u23203712/docs/d40/6a40beb6bf3b/IMG_20210913_233340.jpg?extra=lBoIjd4yWwWK0qlzMKMFSU4Xe2eaWcHfE-JoHQsittadAwisURSJoBXh5IUZZfcX1lnd-p4GMGmbuJs-W4vEwVwnwvADy0aBZ8kKNuHQc5cIularIzy9ySxS-1v_lFkqZpuJFuxIN4zmcDlnMUnfF4s", thresholdNearestStarsLocal = 40, tresholdDegree = 7, decimals = 3):
    global thresholdNearestStars
    thresholdNearestStars = thresholdNearestStarsLocal

    lat = 40
    lon = 40
    limiting_magnitude = 2.78
    limiting_big_magnitude = 3.9
    ts = load.timescale()
    t = ts.utc(2021, 4, 16)

    if url.startswith("http"):
        im = load_image(url)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    else:
        im = cv2.imread(url)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # imgplot = plt.imshow(im)
        # plt.show()
    im = normalizeImage(im)
    print("im shape:",im.shape)
    print("im:",im)
    
    inputs,reference_points,points_photo_origin = image_to_features(im)
    if inputs is None:
        return None
    print("inputs:",inputs.shape)
    # temp_img = np.squeeze(inputs)
    # temp_img = np.moveaxis(temp_img,0,2)
    # imgplot = plt.imshow(temp_img)
    # plt.show()
    reference_points = points_photo_origin
    df_final = find_center_star(inputs)
    print("df_final: ",df_final)

    vega_star = Star.from_dataframe(df_final)
    points_to_be_aligned_df, pointsStars_df_big = getSpace(lat,lon,t,limiting_magnitude = 5,limiting_magnitude_big = limiting_big_magnitude,center_star=vega_star)
    points_to_be_aligned_df = points_to_be_aligned_df.sort_values('magnitude')

    count_key_stars = 25
    field_of_view_degrees = 45.0
    angle = np.pi - field_of_view_degrees / 360.0 * np.pi
    limit = np.sin(angle) / (1.0 - np.cos(angle))
    limit = limit * 1.5
    points_to_be_aligned_df = points_to_be_aligned_df[(points_to_be_aligned_df.x > -limit) & (points_to_be_aligned_df.x < limit) & (points_to_be_aligned_df.y > -limit) & (points_to_be_aligned_df.y < limit)]
    points_to_be_aligned_df = points_to_be_aligned_df[:count_key_stars]
    pointsX = points_to_be_aligned_df['x'].to_numpy()
    pointsY = points_to_be_aligned_df['y'].to_numpy()

    points_to_be_aligned = np.vstack((pointsX, pointsY)).T

    reference_points = reference_points[:count_key_stars]
    
    common_points1, common_points2, mutated_reference_df, structure_list1,structure_list2 = getCommonPoints(reference_points,points_to_be_aligned,points_to_be_aligned_df,tresholdDegree, decimals)

    common_points1 = common_points1.reshape(-1,1,2)
    common_points2 = common_points2.reshape(-1,1,2)
    M, mask = cv.findHomography(common_points2, common_points1, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    pts = np.array(points_to_be_aligned.reshape(-1,1,2))
    try:
        dst = cv.perspectiveTransform(pts,M)
        dst = dst.reshape(-1,2) # позиции, некоторые звезды могут выходить за пределы экрана
        return dst, points_to_be_aligned_df['name'], df_final['name']
    except Exception as e:
        print("find_stars_info error: ",e)
        print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        return None
    

# magnitudeMap: 52
# dst, names, name_center_star = find_stars_info("/root/IMG_0077_2.jpg") # для тестирования при апдейте сервака раскоментить потом
# dst, names = find_stars_info("/content/IMG_20210913_233340.jpg")
# print("dst:",dst)
# print("names:",names)

# im = cv2.imread("/content/IMG_20210913_233340.jpg")











