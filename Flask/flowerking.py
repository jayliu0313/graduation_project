# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:19:20 2020

@author: qqwdf
"""
# from keras.preprocessing.image import img_to_array
import numpy as np
import json
import cv2
import os
import pandas as pd
import random
from datetime import datetime

from flask import *
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
######################
from build_model import model
######################

db = SQLAlchemy()
# initialize our Flask application and the Keras model
app = Flask(__name__)
#Configure db
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:cvmle814@localhost:3306/flowerlive"
app.config['JSON_AS_ASCII'] = False   #chinese ascii decode and encode
db.init_app(app)
multichoice_index = 0
# 模型( model )定義
class flower_species(db.Model):
    __tablename__ = 'flower_species'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(45), unique=True)
    nickname = db.Column(db.String(200), nullable=False)
    scientific_name = db.Column(db.String(200), nullable=False)
    traits = db.Column(db.String(200), nullable=False)
    use = db.Column(db.String(200), nullable=False)
    link = db.Column(db.String(500))

    def __init__(self, id, name, nickname, scientific_name, traits, use, link):
        self.id = id
        self.name = name
        self.nickname = nickname
        self.scientific_name = scientific_name
        self.traits = traits
        self.use = use
        self.link = link


# 指定要查詢的路徑
directory = 'static/images/flowerImg/'
allFilePath = ['0.png'] * 101

#猜猜看
def get_choice():
  sql = "SELECT * FROM flowerlive.flower_species  where `id` > 0 order by rand() limit 4"
  temp = db.engine.execute(sql)
  result = [row for row in temp]
  index =  random.randint(0, 3) #0-3取一個數當作index，use random function
  return index, result

def get_filepath(allFileList):
  for file in allFileList:
    if os.path.isfile(directory+file):
      allFilePath[int(file[:file.index(".")])] = file
      print(file[:file.index(".")])

def preprocess_image(image, target):
    if(image.shape[2] != 3):  ### ch只有一
      image = np.tile(image, (1,1,3))
    image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    return image


@app.route('/')
@app.route('/home')
def index():
  now = datetime.now()
  now_str = "'" + str(now.month) + "月" + str(now.day) + "日'"
  sql = "SELECT * FROM flowerlive.flower_meaning where `date` = " + now_str
  conn = db.engine.connect().connection
  dataframe = pd.read_sql(sql, conn)
  mean_data = {}
  print(dataframe)
  filepath = "images/flower_meaning/" + str(dataframe['id'][0]) + ".jpg"
  # 將預測結果整理後回傳 json 檔案（分類和可能機率）
  # for (_, label, prob) in results[0]:
  mean_data['name'] = dataframe['name'][0]
  mean_data['date'] = dataframe['date'][0]
  mean_data['sentence'] = dataframe['sentence'][0]
  mean_data['filepath'] = filepath  

  multichoice_index, multichoice = get_choice()

  page = request.args.get('page', 1, type=int)
  posts = flower_species.query.filter(flower_species.id>0).paginate(page = page, per_page=6)
  return render_template('index.html', mean_data = mean_data, posts=posts, multichoice = multichoice, multichoice_index=multichoice_index)

@app.route('/up_photo', methods = ['GET', 'POST'])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {}
    print('request')
    # ensure an image was properly uploaded to our endpoint
    if request.method == 'POST':
        if request.files.get('photo'):
            # 從 flask request 中讀取圖片（byte str）
            image = request.files['photo'].read()
            # 將圖片轉成 PIL 可以使用的格式
            
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            # image = Image.open(io.BytesIO(image))
            
            # 進行圖片前處理方便預測模型使用
            image = preprocess_image(image, target=(224, 224))
            t=image[:,:].copy()
            image=image[:,:,::-1]#test
            # image[:,:,0]=t[:,:,2]
            # image[:,:,2]=t[:,:,0]
            # 原本初始化的 tensorflow graph 搭配 sesstion context，預測結果
            temp=0
            image_1 =[]
            image_1.append(image)
            image_1=np.array(image_1)
            image_1=image_1/255
            preds = model.predict(image_1)
            top_k=3#除了最大的以外，後3個相似的 
            top_k_idx=preds[0].argsort()[::-1][0:top_k]#用np.argsort 然後將其倒過來 0至top_k-1
            print(top_k_idx)
            # print(preds)
            for x in preds:
              for i in range(101):
                if x[i]>0.8:
                  x[i]=1
                  temp=i
                  print(i)
                else:
                  x[i]=0
            
            sql = "SELECT * FROM flowerlive.flower_species WHERE id = " + str(temp)
            filepath = "static/images/flowerImg/" + str(temp) + ".jpg"
            conn = db.engine.connect().connection
            dataframe = pd.read_sql(sql, conn)
            #print(filepath)
            #print(dataframe)
            # 將預測結果整理後回傳 json 檔案（分類和可能機率）
            # for (_, label, prob) in results[0]:
            data['name'] = dataframe['name'][0]
            data['nickname'] = dataframe['nickname'][0]
            data['sci_name'] = dataframe['scientific_name'][0]
            data['traits'] = dataframe['traits'][0]
            data['use'] = dataframe['use'][0]
            data['link'] = dataframe['link'][0]
            data['filepath'] = filepath
            #data['name'] = str(query_data.fetchone()[1])
            return jsonify(data)
    return 'OK'

if __name__ == '__main__':
    app.debug = True
    #allFileList = os.listdir(directory)
    #get_filepath(allFileList)
    app.run(port = 3000)

