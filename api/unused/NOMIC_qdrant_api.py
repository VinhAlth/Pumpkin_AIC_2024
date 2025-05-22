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
from tqdm import tqdm
import ujson
import subprocess
import asyncio

sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

from libs.vector_database.QDRANT.qdrant import QDRANT

# Đường dẫn lưu file feature của clip model
FEATURES_PATH= ['/dataset/AIC2024/pumkin_dataset/0/nomicai/']
KEYFRAME_FOLDER_PATH = "/dataset/AIC2024/pumkin_dataset/"
SPLIT_NAME = "pyscenedetect_t_5"
DATASET_INDEX = "/dataset/AIC2024/pumkin_dataset/0/utils/index"
FILTER_FILE = "/dataset/AIC2024/pumkin_dataset/0/utils/valid_frame_h12_0.json"
RERANK_MODEL_URL = "https://api.siu.edu.vn/aic/2/preprocess?text={}&k={}"
RERANK_FEATURES_PATH = ['/dataset/AIC2024/pumkin_dataset/0/mobileclip/']
RERANK_SIZE = 512

import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/nhihtc/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152


from models.Nomicai import NOMICAI
model = NOMICAI()

def count_other_processes_using_current_file():
    current_file = os.path.abspath(sys.argv[0])
    output = subprocess.run(['ps', 'aux'], capture_output=True, text=True).stdout
    return output.count(current_file) - 1

qdrant = QDRANT()
qdrant.addDatabase("NOMIC", 768, FEATURES_PATH, SPLIT_NAME, FILTER_FILE)

# if count_other_processes_using_current_file()==0:
#     qdrant = QDRANT("BLIP2Coco", 256, FILTER_FILE)
#     qdrant.addDatabase(FEATURES_PATH, SPLIT_NAME)
# else:
#     qdrant = QDRANT("BLIP2Coco", 256, FILTER_FILE, PARALLEL=True)


dummy_query = np.load("/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/vector_database/QDRANT/cat_nomic.npy").reshape(1,-1).astype('float32')[0]
qdrant.search(dummy_query, 3)
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

# from utils.temporal.sceneA_result_process import preprocess
def get_nearest_index(index_by_video_path, video, frame):
    #open video index file
    with open(index_by_video_path + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    #reformat
    index_dict = {int(k):v for k,v in index_dict.items()}
    
    #get the index right after the lower bound (logN)
    ind = bisect.bisect_left(list(index_dict.keys()), frame)
    
    #ind-1 = lower_bound
    return ind-1


def preprocessing_temporal(index_by_video_path, sceneA_result):
    #store min frame by video
    min_frame = {}
    #store max frame by video
    max_frame = {}
    #return result: {video:(start_index, end_index)}
    result = {}
    #get min and max frame per video in scene A
    #max frame = real max frame + 1000 (~40s video time)
    for video, frame in sceneA_result:
        if video in min_frame:
            min_frame[video] = min(min_frame[video], int(frame))
            max_frame[video] = max(max_frame[video], int(frame) + 1000)
        else:
            min_frame[video] = int(frame)
            max_frame[video] = int(frame) + 1000

    #only store 1 record per video
    added = {}
    for video, frame in sceneA_result:
        if video not in added:
            #get_nearest_index return the index of lower bound of frame in video
            result[video] = (get_nearest_index(index_by_video_path, video, min_frame[video]), get_nearest_index(index_by_video_path, video, max_frame[video]))
            added[video] = True
    print(result)
    return result

def link_scene(current_scene,next_scene):
    result = []
    for current_item in current_scene: 
        for next_item in next_scene: 
            if current_item['video_name'] == next_item['video_name']: 
                if int(current_item['keyframe_id']) < int(next_item['keyframe_id']): 
                    if current_item not in result: 
                        result.append(current_item)
    return result


# def transform_result_temporal(I, D, db):
#     search_results = []
#     search_temporal = []
#     # search_temporal = {}

#     for instance in zip(I[0],D[0]):
#         ins_id, distance = instance
#         video_name, idx, idx_folder = db[ins_id]
        
#         frames_folder = KEYFRAME_FOLDER_PATH + str(idx_folder) + "/frames/" + SPLIT_NAME + "/Keyframes_" + str(video_name.split('_')[0]) +'/keyframes/'+ video_name
        
#         keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
        
#         video_name = video_name + '.mp4'
        
#         result = {"idx_folder": str(idx_folder),"video_name":str(video_name),
#                                 "keyframe_id": str(keyframe_id),
#                                 "score": str(distance)}
#         temporal = (str(video_name).split('.')[0],str(keyframe_id))
#         # search_temporal[str(video_name).split('.')[0]] = (0,99999)

        
#         search_results.append(result)
#         search_temporal.append(temporal)
#     return [search_results, search_temporal]

# def transform_result(I, D, db):
#     search_results = []
#     for instance in zip(I[0],D[0]):
#         ins_id, distance = instance
#         video_name, idx, idx_folder = db[ins_id]
        
#         print(video_name, idx, idx_folder)
#         frames_folder = KEYFRAME_FOLDER_PATH + str(idx_folder) + "/frames/" + SPLIT_NAME + "/Keyframes_" + str(video_name.split('_')[0]) +'/keyframes/'+ video_name
        
#         keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
        
#         video_name = video_name + '.mp4'
        
#         result = {"key":str(ins_id),"idx_folder": str(idx_folder),"video_name":str(video_name),
#                                 "keyframe_id": str(keyframe_id),
#                                 "score": str(distance)}
#         print("result: ", result)
#         search_results.append(result)
#     return search_results

# Gọi lên chạy 
app = flask.Flask("API Text Search")
app.config["DEBUG"] = False

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

@app.route('/text_search')
def text_search():
    global db, faiss_db
    
    text = ""
    k = ""
    rerank = ""
    
    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']
        rerank = request.json['rerank']
    else:
        text = request.args.get('text')
        k = request.args.get('k')
        rerank = request.args.get('rerank')
    
    #MODEL A
    text_feat_arr_A = preprocessing_text(text)
    search_results_A = qdrant.search(text_feat_arr_A, int(k))
    
    if rerank=='MODEL':
        #MODEL B
        text_feat_arr_B = np.array(requests.get(RERANK_MODEL_URL.format(text,k)).json()).astype('float32')
        search_results_B = qdrant.search_rerank(RERANK_FEATURES_PATH, search_results_A, text_feat_arr_B, RERANK_SIZE, int(k))
        response = flask.jsonify(search_results_B)
    else:
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
    
    if request.method == "POST":
        image = request.json['image_url']
        k = request.json['k']
        rerank = request.json['rerank']
    else:
        image = request.args.get('image_url')
        k = request.args.get('k')
        rerank = request.args.get('rerank')
        
    #MODEL A
    img_feat_arr_A = preprocessing_image(image)
    search_results_A = qdrant.search(img_feat_arr_A, int(k))
    
    if rerank=='model':
        #MODEL B
        img_feat_arr_B = np.array(requests.get(RERANK_MODEL_URL.format(image,k)).json()).astype('float32')
        search_results_B = qdrant.search(img_feat_arr_B, int(k))
        response = flask.jsonify(search_results_B)
    else:
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
    
    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']
    else:
        text = request.args.get('text')
        k = request.args.get('k')
    
    if text[-1] == '.': 
        text = text[:-1]
    print(f"text: {text}")
    
    text_list = text.split('.',100)
    queryList = []
    for item in text_list:
        queryList.append(preprocessing_text(item.rstrip('.')))
    
    search_results = qdrant.search_temporal(FEATURES_PATH, queryList, int(k))
    
    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8502, debug=False)