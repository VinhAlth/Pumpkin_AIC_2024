import flask
from flask import request
import os 
import numpy as np
import urllib.parse
import json
import sys
import ujson
import bisect
import pillow_avif
import requests
import subprocess


os.environ['HF_HOME'] = '/workspace/competitions/AIC_2024/SIU_Pumpkin/cache/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

from libs.vector_database.QDRANT.qdrant import QDRANT

# Đường dẫn lưu file feature của clip model
FEATURES_PATH= ['/dataset/AIC2024/pumkin_dataset/0/eva/', '/dataset/AIC2024/pumkin_dataset/1/eva/']
KEYFRAME_FOLDER_PATH = "/dataset/AIC2024/pumkin_dataset/"
SPLIT_NAME = "autoshot"
DATASET_INDEX = "/dataset/AIC2024/pumkin_dataset/utils/index_autoshot"

from models.Eva import EVA
model = EVA()

def count_other_processes_using_current_file():
    current_file = os.path.abspath(sys.argv[0])
    output = subprocess.run(['ps', 'aux'], capture_output=True, text=True).stdout
    return output.count(current_file) - 1

# if count_other_processes_using_current_file()==1:
#     qdrant = QDRANT("LAIONB88K", 1024, FILTER_FILE, PARALLEL=False)
#     qdrant.addDatabase(FEATURES_PATH, SPLIT_NAME)
# else:
#     qdrant = QDRANT("LAIONB88K", 1024, FILTER_FILE, PARALLEL=True)

qdrant = QDRANT("EVA")
qdrant.addDatabase("EVA", 1280, FEATURES_PATH, SPLIT_NAME)

dummy_query = np.load("/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/vector_database/QDRANT/cat_eva.npy").reshape(1,-1).astype('float32')[0]
qdrant.search(dummy_query, 3, "", "", "")
print("Dummy Query Finished")

def preprocessing_text(text):
    global model
    text_feat_arr = model.get_text_features(text) 
    text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32') #=> float32
    return text_feat_arr[0]

def preprocessing_image(image):
    global model
    image_feat_arr = model.get_image_features(image) 
    image_feat_arr = image_feat_arr.reshape(1,-1).astype('float32') #=> float32
    return image_feat_arr[0]


# Gọi lên chạy 
app = flask.Flask("API Text Search")
app.config["DEBUG"] = False

from flask_compress import Compress
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_HOST'] = 'localhost'
app.config['CACHE_REDIS_PORT'] = 6379
from flask_compress import Compress
app.config['COMPRESS_MIMETYPES'] = ['text/html', 'text/css', 'application/json']
app.config['COMPRESS_LEVEL'] = 6  # Default compression level
app.config['COMPRESS_MIN_SIZE'] = 500  # Minimum response size (in bytes) to trigger compression
Compress(app)

@app.route('/preprocess')
def preprocess():
    text = ""
    k = ""
    
    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']
    else:
        text = request.args.get('text')
        k = request.args.get('k')
    
    if text[-1] == '.': 
        text = text[:-1]
    
    text_feat_arr = preprocessing_text(text)
    print(f"text: {text}")
    
    response = flask.jsonify(text_feat_arr.tolist())
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response

@app.route('/scroll')
def scroll():
    video_filter = ""
    time_in = ""
    time_out = ""
    
    if request.method == "POST":
        k = request.json['k']
        video_filter = request.json['video_filter']
        time_in = request.json['time_in']
        time_out = request.json['time_out']
    else:
        k = request.args.get('k')
        video_filter = request.args.get('video_filter')
        time_in = request.args.get('time_in')
        time_out = request.args.get('time_out')
    
    scroll_result = qdrant.scroll_video(k, video_filter, time_in, time_out)
    
    response = flask.jsonify(scroll_result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

@app.route('/text_search')
def text_search():
    global db, faiss_db
    
    text = ""
    k = ""
    video_filter = ""
    time_in = ""
    time_out = ""
    
    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']
        video_filter = request.json['video_filter']
        time_in = request.json['time_in']
        time_out = request.json['time_out']
    else:
        text = request.args.get('text')
        k = request.args.get('k')
        video_filter = request.args.get('video_filter')
        time_in = request.args.get('time_in')
        time_out = request.args.get('time_out')       
    
    #MODEL A
    text_feat_arr_A = preprocessing_text(text)
    search_results_A = qdrant.search(text_feat_arr_A, int(k), video_filter, time_in, time_out)
    
    response = flask.jsonify(search_results_A)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

@app.route('/image_search')
def image_search():      
    global db, faiss_db
    
    image = ""
    k = ""
    video_filter = ""
    time_in = ""
    time_out = ""
    
    if request.method == "POST":
        image = request.json['image_url']
        k = request.json['k']
        video_filter = request.json['video_filter']
        time_in = request.json['time_in']
        time_out = request.json['time_out']
    else:
        image = request.args.get('image_url')
        k = request.args.get('k')
        video_filter = request.args.get('video_filter')
        time_in = request.args.get('time_in')
        time_out = request.args.get('time_out')   
        
    #MODEL A
    img_feat_arr_A = preprocessing_image(image)
    search_results_A = qdrant.search(img_feat_arr_A, int(k), video_filter, time_in, time_out)
    
    response = flask.jsonify(search_results_A)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

@app.route('/temporal_search')
def temporal_search():
    global db, faiss_db
    
    text = ""
    k = ""
    video_filter = ""
    time_in = ""
    time_out = ""
    
    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']
        video_filter = request.json['video_filter']
        time_in = request.json['time_in']
        time_out = request.json['time_out']
    else:
        text = request.args.get('text')
        k = request.args.get('k')
        video_filter = request.args.get('video_filter')
        time_in = request.args.get('time_in')
        time_out = request.args.get('time_out')   
    
    if text[-1] == '.': 
        text = text[:-1]
    print(f"text: {text}")
    
    text_list = text.split('.',100)
    queryList = []
    re_sort = None
    for idx, item in enumerate(text_list):
        if str(item).startswith('*'):
            print("re_sort base on scene ", idx+1)
            re_sort = idx
        queryList.append(preprocessing_text(item.rstrip('.').replace("*","")))
    
    search_results = qdrant.search_temporal(queryList, int(k), video_filter, time_in, time_out, re_sort)
    
    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8502, debug=False)