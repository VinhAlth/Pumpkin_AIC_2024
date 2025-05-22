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
import ast


os.environ['HF_HOME'] = '/workspace/competitions/AIC_2024/SIU_Pumpkin/cache/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

from libs.vector_database.QDRANT.qdrant import QDRANT

# Đường dẫn lưu file feature của clip model
FEATURES_PATH= ['/dataset/AIC2024/pumkin_dataset/0/dfn5b/', '/dataset/AIC2024/pumkin_dataset/1/dfn5b/', '/dataset/AIC2024/pumkin_dataset/2/dfn5b/']
KEYFRAME_FOLDER_PATH = "/dataset/AIC2024/pumkin_dataset/"
SPLIT_NAME = "autoshot"
DATASET_INDEX = "/dataset/AIC2024/pumkin_dataset/utils/index_autoshot"
RERANK_SIZE = 1024

from models.DFN5B import DFN5B
model = DFN5B()

def count_other_processes_using_current_file():
    current_file = os.path.abspath(sys.argv[0])
    output = subprocess.run(['ps', 'aux'], capture_output=True, text=True).stdout
    return output.count(current_file) - 1

# if count_other_processes_using_current_file()==1:
#     qdrant = QDRANT("LAIONB88K", 1024, FILTER_FILE, PARALLEL=False)
#     qdrant.addDatabase(FEATURES_PATH, SPLIT_NAME)
# else:
#     qdrant = QDRANT("LAIONB88K", 1024, FILTER_FILE, PARALLEL=True)

qdrant = QDRANT("DFN5B")
#qdrant.addDatabase("DFN5B", 1024, FEATURES_PATH, SPLIT_NAME)

dummy_query = np.load("/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/vector_database/QDRANT/cat_dfn5b.npy").reshape(1,-1).astype('float32')[0]
qdrant.search(dummy_query, 3, "", "", "",[0])
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

list_shot_dict = {}

def getShot(videoName, frameName):
    if videoName not in list_shot_dict.keys():
        json_path = os.path.join("/workspace/competitions/AIC_2024/SIU_Pumpkin/dataset/video_name", f"{videoName}.json")
        with open(json_path, 'r') as file:
            data = ujson.load(file)
        list_shot_dict[videoName]=data
    else:
        data = list_shot_dict[videoName]
    for i, list_frame in enumerate(data):
        #print(list_frame)
        if int(list_frame[0][0][:-4])>int(frameName):
            return data[i-1]
    return data[len(data)-1]

def sort_by_shot(results):
    shot_groups = {}
    image_index = {}

    # Create a mapping of each frame path to its original index
    for index, result in enumerate(results):
        frame_path = result['video_name']  # Use video_name as the frame identifier
        image_index[frame_path] = index  # Store the original index

    # Group frames by shots using getShot
    for result in results:
        video_name = result['video_name'].replace('.mp4','')
        keyframe_id = result['keyframe_id']  # Use keyframe_id to identify frames
        keyframe_number = int(keyframe_id)  # Convert keyframe_id to an integer

        # Get the shot information using the video name and keyframe number
        shot = getShot(video_name, keyframe_number)

        # Convert the shot list to a tuple to make it hashable
        shot_key = tuple(shot[0])  # Assuming first element is the list of frames

        # Create a group for each shot if not already present
        if shot_key not in shot_groups:
            shot_groups[shot_key] = []
        shot_groups[shot_key].append(result)  # Store the entire result

    # Sort the groups by the minimum index of the frames within each group
    sorted_groups = sorted(
        shot_groups.values(),
        key=lambda group: min(image_index[result['video_name']] for result in group)
    )

    # Flatten the sorted groups back into a single list of results
    sorted_results = [result for group in sorted_groups for result in group]

    return sorted_results



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
    s2t_filter = ""
    
    if request.method == "POST":
        k = request.json['k']
        video_filter = request.json['video_filter']
        time_in = request.json['time_in']
        time_out = request.json['time_out']
        s2t_filter = request.json['s2t_filter']
    else:
        k = request.args.get('k')
        video_filter = request.args.get('video_filter')
        time_in = request.args.get('time_in')
        time_out = request.args.get('time_out')
        s2t_filter = request.args.get('s2t_filter')
    
    if s2t_filter == "":
        s2t_filter = None
    
    scroll_result = qdrant.scroll_video(k, video_filter, time_in, time_out, s2t_filter)
    
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
    batchs = ""
    s2t_filter = ""
    
    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']
        video_filter = request.json['video_filter']
        time_in = request.json['time_in']
        time_out = request.json['time_out']
        batchs = request.json['batchs']
        s2t_filter = request.json['s2t_filter']
    else:
        text = request.args.get('text')
        k = request.args.get('k')
        video_filter = request.args.get('video_filter')
        time_in = request.args.get('time_in')
        time_out = request.args.get('time_out')
        batchs = request.args.get('batchs')   
        s2t_filter = request.args.get('s2t_filter')  
    
    batchs = ast.literal_eval(batchs)
    
    if s2t_filter == "":
        s2t_filter = None
    
    #MODEL A
    text_feat_arr_A = preprocessing_text(text)
    search_results_A = qdrant.search(text_feat_arr_A, int(k), video_filter, time_in, time_out, batchs, s2t_filter)
    
    search_results_A = sort_by_shot(search_results_A)
    
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
    s2t_filter = ""
    
    if request.method == "POST":
        image = request.json['image_url']
        k = request.json['k']
        video_filter = request.json['video_filter']
        time_in = request.json['time_in']
        time_out = request.json['time_out']
        batchs = request.json['batchs']
        s2t_filter = request.json['s2t_filter']
    else:
        image = request.args.get('image_url')
        k = request.args.get('k')
        video_filter = request.args.get('video_filter')
        time_in = request.args.get('time_in')
        time_out = request.args.get('time_out')  
        batchs = request.args.get('batchs')  
        s2t_filter = request.args.get('s2t_filter')
    
    batchs = ast.literal_eval(batchs)
    
    if s2t_filter == "":
        s2t_filter = None
      
    #MODEL A
    img_feat_arr_A = preprocessing_image(image)
    search_results_A = qdrant.search(img_feat_arr_A, int(k), video_filter, time_in, time_out, batchs, s2t_filter)
    search_results_A = sort_by_shot(search_results_A)
    response = flask.jsonify(search_results_A)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

@app.route('/temporal_search')
def temporal_search():
    
    text = ""
    k = ""
    video_filter = ""
    time_in = ""
    time_out = ""
    s2t_filter = ""
    
    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']
        video_filter = request.json['video_filter']
        time_in = request.json['time_in']
        time_out = request.json['time_out']
        batchs = request.json['batchs']
        s2t_filter = request.json['s2t_filter']
    else:
        text = request.args.get('text')
        k = request.args.get('k')
        video_filter = request.args.get('video_filter')
        time_in = request.args.get('time_in')
        time_out = request.args.get('time_out')
        batchs = request.args.get('batchs')   
        s2t_filter = request.args.get('s2t_filter')
    
    batchs = ast.literal_eval(batchs)
    
    if s2t_filter == "":
        s2t_filter = None
    
    
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
    
    search_results = qdrant.search_temporal(queryList, int(k), video_filter, time_in, time_out, re_sort, batchs, s2t_filter)
    search_results = sort_by_shot(search_results)
    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8502, debug=False)