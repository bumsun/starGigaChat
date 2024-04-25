#@title
import numpy as np
from collections import Counter
from math import sqrt

tresholdDegree = 7
decimals = 3
connection_distance = 0.9

def trunc(values, decimals=0):
    return np.trunc(values*10**decimals)/(10**decimals)
def calculateAngle(trios):    
    a = trios[:,2]
    b = trios[:,1]
    c = trios[:,0]
    ba = a - b
    bc = c - b
    # print("norm: ",(np.linalg.norm(ba,axis=1)))
    # print("norm 2: ",np.linalg.norm(bc,axis=1))
    # print("einsum: ",np.einsum("ij,ij->i", ba, bc))
    # np.einsum("ij,ij->i", ba, bc) разобраться умножением этих разных векторов. Чтобы перемножение четко работало
    cosine_angle = np.einsum("ij,ij->i", ba, bc) / (np.linalg.norm(ba,axis=1) * np.linalg.norm(bc,axis=1))
    angle = np.arccos(cosine_angle)
    degrees = np.degrees(angle)
    degrees = trunc(degrees/tresholdDegree, decimals=decimals)*tresholdDegree
    return degrees
def calculateAngleNew(points):
    pointsManyC = points
    pointsManyB = getMany(pointsManyC)
    pointsManyA = getMany(pointsManyB)
    # changeInX = pointsOfPoints[:,:,:,0] - points[:,:,0]
    # changeInY = pointsOfPoints[:,:,:,1] - points[:,:,1]
    # return np.rad2deg(np.arctan2(changeInY,changeInX) % (2 * np.pi))

    ba = pointsManyA - pointsManyB
    bc = pointsManyC - pointsManyB
    
    # cosine_angle = np.einsum("ij,ij->i", ba, bc) / (np.linalg.norm(ba,axis=3) * np.linalg.norm(ba,axis=3))
    cosine_angle = np.sum((ba*bc), axis=3) / (np.linalg.norm(ba,axis=3) * np.linalg.norm(bc,axis=2))
    angle = np.arccos(cosine_angle)
    degrees = np.degrees(angle)
    degrees = trunc(degrees/tresholdDegree, decimals=decimals)*tresholdDegree
    return degrees, pointsManyA, getManyWithoutRoll(pointsManyB), getManyWithoutRoll(getManyWithoutRoll(pointsManyC))
def getMany(source):
    many = []
    for i in range(source.shape[0]-1):
        x = np.roll(source, i+1, axis=0)
        many.append(x)
    many = np.array(many)
    return many

def getManyWithoutRoll(source):
    many = []
    for i in range(source.shape[0]-1):
        many.append(source)
    many = np.array(many)
    
    return many

def calculateOneAngle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    degrees = np.degrees(angle)
    degrees = trunc(degrees/tresholdDegree, decimals=decimals)*tresholdDegree
    return degrees
def calculateDistanceRatio(trios):
    dist1, dist2 = calculateDistance(trios)
    return trunc(dist1/dist2, decimals=2)

def calculateOneDistanceRatio(point1,point2,point3):
    dist1 = sqrt( (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 )
    dist2 = sqrt( (point3[0] - point2[0])**2 + (point3[1] - point2[1])**2 )
    return trunc(dist1/dist2, decimals=1), trunc(dist2/dist1, decimals=1)


def calculateDistance(trios):
    pointsManyC = trios[:,2]
    pointsManyB = trios[:,1]
    pointsManyA = trios[:,0]
    # changeInX = pointsOfPoints[:,:,:,0] - points[:,:,0]
    # changeInY = pointsOfPoints[:,:,:,1] - points[:,:,1]
    # return np.rad2deg(np.arctan2(changeInY,changeInX) % (2 * np.pi))

    dist1 = np.sqrt( (pointsManyB[:,0] - pointsManyA[:,0])**2 + (pointsManyB[:,1] - pointsManyA[:,1])**2 )
    dist2 = np.sqrt( (pointsManyC[:,0] - pointsManyB[:,0])**2 + (pointsManyC[:,1] - pointsManyB[:,1])**2 )

    return dist1,dist2

def findIntersectInTrios(trios,trios2):
    roundValue = 5
    zipTrios = triosZipBySum(trios)
    zipTrios2 = triosZipBySum(trios2)
    zipTrios = trunc(zipTrios, decimals=roundValue)
    zipTrios2 = trunc(zipTrios2, decimals=roundValue)
    intersectTrios = np.intersect1d(zipTrios,zipTrios2, return_indices=True)[0]
    indTrios = np.isin(zipTrios,intersectTrios)
    trios = trios[indTrios]
    return trios

def getCommonPoints(pointsPhoto,pointsHipparcos,points_to_be_aligned_df, tresholdDegreeLocal = 7, decimalsLocal = 3):
    global tresholdDegree,decimals
    tresholdDegree = tresholdDegreeLocal
    decimals = decimalsLocal

    points_to_be_aligned_df['x_y'] = np.char.add(points_to_be_aligned_df['x'].to_numpy().astype(str), points_to_be_aligned_df['y'].to_numpy().astype(str))
    points_to_be_aligned_df = points_to_be_aligned_df.set_index('x_y')
    
    angles1, pointsPhotoManyA, pointsPhotoManyB, pointsPhotoManyC = calculateAngleNew(pointsPhoto)
    angles2, pointsHipparcosManyA, pointsHipparcosManyB, pointsHipparcosManyC = calculateAngleNew(pointsHipparcos)


    # angles2 = calculateAngle(pointsHipparcosMany)
    # print("angles1:",angles1)
    commonValue = np.intersect1d(angles2,angles1, return_indices=True)[0]
    
    commonMap1 = np.isin(angles1,commonValue)
    commonMap2 = np.isin(angles2,commonValue)

    ratios1,angleTrios1,trioUnique1 = getSimilarRatioAndAngles(pointsPhotoManyA,pointsPhotoManyB,pointsPhotoManyC,commonMap1)
    ratios2,angleTrios2,trioUnique2 = getSimilarRatioAndAngles(pointsHipparcosManyA,pointsHipparcosManyB,pointsHipparcosManyC,commonMap2)

    ratiosAndAngles1 = angleTrios1*100+ratios1
    ratiosAndAngles2 = angleTrios2*100+ratios2

    commonValueRatiosAndAngles, ids1, ids2 = np.intersect1d(ratiosAndAngles1,ratiosAndAngles2, return_indices=True)

    new_founded_angles = trioUnique1[ids1]
    new_founded_angles2 = trioUnique2[ids2]
    print("new_founded_angles:",new_founded_angles)
    print("new_founded_angles2:",new_founded_angles2)
    ### Ищем стартовое trio точек для поиска общей структуры точек
    best_start_trio,best_start_trio2,best_start_squeeze,best_start_squeeze2,new_founded_angles_squeeze,new_founded_angles2_squeeze,score_trios = find_best_start_trio(new_founded_angles,new_founded_angles2)
    

    print("score_trios:",score_trios)
    print("best_start_trio:",best_start_trio)
    print("best_start_trio2:",best_start_trio2)


    ### Ищем общую структуру в точках, когда эти точки начинают формировать общие фиугры, 
    structure_list1,structure_list2 = find_sctructure_list(best_start_trio,best_start_trio2,best_start_squeeze,best_start_squeeze2,new_founded_angles,new_founded_angles2,new_founded_angles_squeeze,new_founded_angles2_squeeze)
    new_founded_angles = structure_list1
    new_founded_angles2 = structure_list2
    print("structure_list1 len:",len(structure_list1))
    print("structure_list2 len:",len(structure_list2))

    print("structure_list1:",structure_list1)
    print("structure_list2:",structure_list2)

    new_founded_angles = new_founded_angles[:,1,:]
    new_founded_angles2 = new_founded_angles2[:,1,:]

    x_y = np.char.add(new_founded_angles2[:,0].astype(str),new_founded_angles2[:,1].astype(str))
    mutated_reference_df = points_to_be_aligned_df.loc[x_y]


    return new_founded_angles, new_founded_angles2, mutated_reference_df,structure_list1,structure_list2
def find_best_start_trio(new_founded_angles,new_founded_angles2):
    new_founded_angles_squeeze = np.char.add(new_founded_angles[:,:,0].astype(str),new_founded_angles[:,:,1].astype(str))
    new_founded_angles2_squeeze = np.char.add(new_founded_angles2[:,:,0].astype(str),new_founded_angles2[:,:,1].astype(str))
    most_freq1 = dict(Counter(new_founded_angles_squeeze.flat).most_common())
    most_freq2 = dict(Counter(new_founded_angles2_squeeze.flat).most_common())

    score_trios = {}
    for num in range(len(new_founded_angles_squeeze)):
        row = new_founded_angles_squeeze[num]
        row2 = new_founded_angles2_squeeze[num]
        p1,p2,p3 = str(row[0]),str(row[1]),str(row[2])
        p1_2,p2_2, p3_2 = str(row2[0]),str(row2[1]),str(row2[2])

        score_trios[num] = most_freq1[p1]+most_freq1[p2]+most_freq1[p3] + most_freq2[p1_2]+most_freq2[p2_2]+most_freq2[p3_2]
        # score_trios2[num] = most_freq2[p1_2]+most_freq2[p2_2]+most_freq2[p3_2]

    score_trios = {k: v for k, v in sorted(score_trios.items(), key=lambda item: item[1], reverse=True)}
    num_best_trio = list(score_trios.keys())[0]

    best_start_squeeze = new_founded_angles_squeeze[num_best_trio]
    best_start_squeeze2 = new_founded_angles2_squeeze[num_best_trio]

    best_start_trio = new_founded_angles[num_best_trio]
    best_start_trio2 = new_founded_angles2[num_best_trio]

    return best_start_trio,best_start_trio2,best_start_squeeze,best_start_squeeze2,new_founded_angles_squeeze,new_founded_angles2_squeeze,score_trios

def find_sctructure_list(best_start_trio,best_start_trio2,best_start_squeeze,best_start_squeeze2,new_founded_angles,new_founded_angles2,new_founded_angles_squeeze,new_founded_angles2_squeeze):
    structure_list1 = []
    structure_list2 = []
    structure_list1.append(best_start_trio)
    structure_list2.append(best_start_trio2)

    structure_list_squeeze1 = []
    structure_list_squeeze2 = []
    structure_list_squeeze1.append(best_start_squeeze)
    structure_list_squeeze2.append(best_start_squeeze2)

    structure_list_squeeze_center_index = []
    structure_list_squeeze_center_index.append(best_start_squeeze[1])

    ### Ищем общую структуру в точках
    need_new_search = True
    while need_new_search:
        print("need_new_search")
        need_new_search = False
        for num in range(len(new_founded_angles_squeeze)):
            row_full = new_founded_angles[num]
            row_full2 = new_founded_angles2[num]

            row = new_founded_angles_squeeze[num]
            row2 = new_founded_angles2_squeeze[num]
            for num2 in range(len(structure_list_squeeze1)):
                structure_squeeze = structure_list_squeeze1[num2]
                structure_squeeze2 = structure_list_squeeze2[num2]
                if (structure_squeeze[0] == row[0] and structure_squeeze2[0] == row2[0]) or (structure_squeeze[1] == row[1] and structure_squeeze2[1] == row2[1]) or (structure_squeeze[2] == row[2] and structure_squeeze2[2] == row2[2]) or (structure_squeeze[2] == row[2] and structure_squeeze2[0] == row2[0]):
                    if (row[1] not in structure_list_squeeze_center_index):
                        structure_list_squeeze_center_index.append(row[1])
                        structure_list_squeeze1.append(row)
                        structure_list_squeeze2.append(row2)
                        structure_list1.append(row_full)
                        structure_list2.append(row_full2)
                        need_new_search = True
        if len(structure_list1) > 4:
            need_new_search = False
    structure_list1 = np.array(structure_list1)
    structure_list2 = np.array(structure_list2)
    return structure_list1,structure_list2

def getSimilarRatioAndAngles(pointsManyA,pointsManyB,pointsManyC,commonMap):
    filteredPointsA = pointsManyA[commonMap]
    filteredPointsB = pointsManyB[commonMap]
    filteredPointsC = pointsManyC[commonMap]
    # Выше значения отобраны

    trio = np.concatenate((filteredPointsA, filteredPointsB,filteredPointsC), axis=1)
    trio = trio.reshape(-1, 3, 2)

    trioSum = np.add.reduce(trio,axis=1)
    trioSum = np.add.reduce(trioSum,axis=1)

    trioInd = np.unique(trioSum, axis=0, return_index=True)[1]
    # trioUnique = trio[trioInd]
    trioUnique = trio
    dist1, dist2 = calculateDistance(trioUnique)

    # indexDistance = np.argwhere((dist1 < connection_distance) & (dist2 < connection_distance)).flatten() 
    # trioUnique = trioUnique[indexDistance]

    ratio = calculateDistanceRatio(trioUnique)
    angleTrios = calculateAngle(trioUnique)

    return ratio,angleTrios,trioUnique