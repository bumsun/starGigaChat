from typing import Optional,List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
app = FastAPI()
import stars_recogniation as sr
from starlette.responses import StreamingResponse
import cv2
import io
import pandas as pd
import requests

import telebot
bot = telebot.TeleBot('5146697109:AAEWjDKPGKczuIVbdU4rYsB13F0As1YpfnM')
chat_id = "32951559"


# @app.get("/")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

@app.post("/star_recognize/")
async def star_recognize(thresholdNearestStars: str = Form(...), tresholdDegree: str = Form(...),decimals: str = Form(...),image: UploadFile = File(...)):
    print(image.file)
    # print('../'+os.path.isdir(os.getcwd()+"images"),"*************")
    try:
        os.mkdir("images")
        print(os.getcwd())
    except Exception as e:
        print(e) 
    file_name = os.getcwd()+"/images/"+image.filename.replace(" ", "-")
    with open(file_name,'wb+') as f:
        f.write(image.file.read())
        f.close()
    # file = jsonable_encoder({"imagePath":file_name})
    print("textColumnNames:",str(int(tresholdDegree)))
    print("decimals:",str(int(decimals)))
    response = sr.find_stars_info(file_name, int(thresholdNearestStars), int(tresholdDegree), int(decimals))
    if response is None:
        return {"error": "stars not found"}
    # response = sr.find_stars_info("/root/IMG_0077_2.jpg", int(thresholdNearestStars), int(tresholdDegree), int(decimals))
    print("response:",response)

    positions_x = response[0][:,0].astype(int).tolist()
    positions_y =response[0][:,1].astype(int).tolist()
    print("positions_x:",positions_x)
    names = response[1].fillna('').values.tolist()
    indexes = response[1].index.astype(int).tolist()
    print("names:",names)
    print("indexes:",indexes)
    return {"indexes": indexes,"names": names,"positions_x": positions_x,"positions_y": positions_y}

@app.post("/get_key_points/")
async def get_key_points(tresholdContrastStars: str = Form(...), thresholdNearestStars: str = Form(...), image: UploadFile = File(...)):
    print(image.file)
    # print('../'+os.path.isdir(os.getcwd()+"images"),"*************")
    try:
        os.mkdir("images")
        print(os.getcwd())
    except Exception as e:
        print(e) 
    file_name = os.getcwd()+"/images/"+image.filename.replace(" ", "-")
    with open(file_name,'wb+') as f:
        f.write(image.file.read())
        f.close()
    # file = jsonable_encoder({"imagePath":file_name})


    cv_img = sr.get_key_points(file_name, float(tresholdContrastStars), int(thresholdNearestStars))

    res, im_png = cv2.imencode(".png", cv_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.post("/send_log/")
async def send_log(message: str = Form(...), url: str = Form(...)):
    bot.send_message(chat_id, message + " " + url)
    return {"state": "ok", "response": "1"}


@bot.message_handler(func=lambda m: True)
def handle_message_text(message):
    if(message.chat.id in users_active):
        print("message: "+str(message))
        # bot.reply_to(message, message.text)

        res = requests.post("http://77.244.221.121:8080/api/getAnswerWithContext", json={
                "system_message": "ты научный ассистент",
                "human_message": message.text,
                "user_id": str(message.chat.id)
            })
        print("res: "+ res.text)
        res = res.json()
            
        bot.send_message(message.chat.id, res['text']['content'])
    else:
        bot.send_message(message.chat.id, "Загрузите фотографию со звездами документом. Фотография должна быть четкой и не иметь посторонних предметов.")


from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

from telebot import types

users_active = set()
lists = {}

@bot.message_handler(content_types=['document','photo'])
def handle_message_document(message):
    print("message: "+str(message))
    if(message.content_type == "photo"):
        bot.send_message(message.chat.id, "Загрузка должна идти как документ, а не как фотография. Чтобы качество снимка не терялось. Фотография должна быть четкой и не иметь посторонних предметов.")
        return
    file_name = message.document.file_name
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.send_message(message.chat.id, "Подождите примерно 15 сек...")
    response = sr.find_stars_info(file_name, 40, 7, 3)
    if response is None:
        bot.reply_to(message, "stars not found")
        return
    # response = sr.find_stars_info("/root/IMG_0077_2.jpg", int(thresholdNearestStars), int(tresholdDegree), int(decimals))
    print("response:",response)

    positions_x = response[0][:,0].astype(int).tolist()
    positions_y =response[0][:,1].astype(int).tolist()
    print("positions_x:",positions_x)
    print("positions_y:",positions_y)
    names = response[1].fillna('').values.tolist()
    indexes = response[1].index.astype(int).tolist()
    print("names:",names)
    print("indexes:",indexes)
    
    img = Image.open(file_name)
    for i in range(len(positions_x)):
        x = positions_x[i]
        y = positions_y[i]
        name = names[i]
        
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype("sans_serif.ttf", 40)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((x+10, y+10),name,(30,100,240),font=font)
    img.save(file_name)
    bot.send_photo(message.chat.id, photo=open(file_name, 'rb'), caption="В центре вашего снимка звезда: " + response[2])


    
    res = requests.post("http://77.244.221.121:8080/api/getAnswerWithContext", json={
            "system_message": "ты научный ассистент",
            "human_message": "Расскажи подробно про звезду " + response[2],
            "user_id": str(message.chat.id)
        })
    print("res: "+ res.text)
    res = res.json()
    


    
    users_active.add(message.chat.id)
    bot.send_message(message.chat.id, res['text']['content'])

    res = requests.get("https://us-west1-semanticxplorer.cloudfunctions.net/semantic-xplorer-db?query="+response[2]+"+star")
    print("res: "+ res.text)
    res = res.json()

    for item in res:
        item['year'] = item['metadata']['year']
        item['metadata']['title'] = item['metadata']['title'] + " " + str(item['year'])

    newlist = sorted(res, key=lambda d: d['year'])
    newlist = newlist[::-1][:5] # топ 5 последних статей
    print(newlist)
    keyboard = types.InlineKeyboardMarkup()   
    for item in newlist:
        # item['metadata']['abstract']
        
        lists[item['id']] = item
        new_title = translateToRus(item['metadata']['title'])
        item['metadata']['title'] = new_title
        keyboard.add(types.InlineKeyboardButton(new_title, callback_data=item['id']))
    bot.send_message(message.chat.id, text='Почитайте последние статьи, связанные со звездой '+response[2]+':', reply_markup=keyboard)
    bot.send_message(message.chat.id, "Или задайте вопрос ниже, который касается звезды: "+response[2])

@bot.callback_query_handler(func=lambda call: True) #вешаем обработчик событий на нажатие всех inline-кнопок
def callback_inline(call):
    if call.data in lists:
        bot.send_message(call.message.chat.id, lists[call.data]['metadata']['title'] + "\n\n" + translateToRus(lists[call.data]['metadata']['abstract']) + "\n\nЧитать статью в оригинале: https://arxiv.org/abs/"+call.data)

from threading import Thread
from time import sleep

def translateToRus(text):
    res = requests.post("http://77.244.221.121:8080/api/getAnswer", json={
        "system_message": "ты переводчик",
        "human_message": "Переведи на русский язык: " + text
    })
    print("res: "+ res.text)
    res = res.json()

    
    return res['text']['content']

def threaded_function():
    print("running")
    bot.infinity_polling()

thread = Thread(target = threaded_function)
thread.start()
uvicorn.run(app, port=80, host='0.0.0.0')



