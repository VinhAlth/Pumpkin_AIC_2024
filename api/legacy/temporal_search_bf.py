import flask
from flask import jsonify, request
import os 
import requests
from tqdm import tqdm
import json 
import ujson
import bisect
import urllib.parse

import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

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


def preprocess(index_by_video_path, sceneA_result):
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




from libs.models.legacy.BLIP2CocoLavisFeaturesExtractor import BLIP2Coco
from libs.vector_database.FAISS.faiss import indexing_methods_faiss, indexing_methods_faiss_temp

# Đường dẫn lưu file feature của clip model
# FEATURES_PATH= ['/dataset/AIC2023/pumkin_dataset/0/bc_t5/', '/dataset/AIC2023/pumkin_dataset/1/bc_t5/', '/dataset/AIC2023/pumkin_dataset/2/bc_t5/']
FEATURES_PATH= ['/dataset/AIC2024/pumkin_dataset/1/bc/']
KEYFRAME_FOLDER_PATH = "/dataset/AIC2024/pumkin_dataset/"
SPLIT_NAME = "pyscenedetect"

dataset_index = "/workspace/competitions/AIC_2024/SIU_Pumpkin/base_2023/SIU_Pumpkin/utils/temporal/index_test"


# MODEL
model = BLIP2Coco()


def preprocessing_text(text):
    global model
    text_feat_arr = model.get_text_features(text) 
    text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32') #=> float32
    return text_feat_arr

def transform_result(I, D, db):
    search_results = []
    search_temporal = []
    # search_temporal = {}

    for instance in zip(I[0],D[0]):
        ins_id, distance = instance
        video_name, idx, idx_folder = db[ins_id]
        
        frames_folder = KEYFRAME_FOLDER_PATH + str(idx_folder) + "/frames/" + SPLIT_NAME + "/Keyframes_" + str(video_name.split('_')[0]) +'/keyframes/'+ video_name
        
        keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
        
        video_name = video_name + '.mp4'
        
        result = {"idx_folder": str(idx_folder),"video_name":str(video_name),
                                "keyframe_id": str(keyframe_id),
                                "score": str(distance)}
        temporal = (str(video_name).split('.')[0],str(keyframe_id))
        # search_temporal[str(video_name).split('.')[0]] = (0,99999)

        
        search_results.append(result)
        search_temporal.append(temporal)
    return search_results, search_temporal

def link_scene(current_scene,next_scene):
    result = []
    for current_item in current_scene: 
        for next_item in next_scene: 
            if current_item['video_name'] == next_item['video_name']: 
                if int(current_item['keyframe_id']) < int(next_item['keyframe_id']): 
                    if current_item not in result: 
                        result.append(current_item)
    return result
# Gọi lên chạy 
app = flask.Flask("API Text Search")
app.config["DEBUG"] = True

@app.route('/predict', methods=['POST', 'GET'])
def updateCurrentCode():
    global KEYFRAME_FOLDER_PATH
    text ="" #câu truy vấn => duong dan anh

    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']

    else:
        text = request.args.get('text')
        k = request.args.get('k')

    # preprocessing text 
    
    # text_en = TextTranslate(text)
    # text = text_en
    if text[-1] == '.': 
        text = text[:-1]
    print(f"text: {text}")
    
    text_list = text.split('.',100)

    print(f"text: {text_list[0]}")
    text_feat_arr = preprocessing_text(text_list[0])
    # Convert the feature array to a string and encode it
    text_feat_arr_str = urllib.parse.quote(json.dumps(text_feat_arr.tolist()))
    faiss_url = f'http://0.0.0.0:8502/faiss_search?text_feat_arr={text_feat_arr_str}&k={k}&temp=1&dataset=1'
    faiss_response = requests.get(faiss_url)
    
    search_results_A, search_temporal = faiss_response.json()

    search_temporal = preprocess(dataset_index, search_temporal)
    for text in text_list[1:]: 
        print(f"text: {text}")
        db_t, faiss_db_t = indexing_methods_faiss_temp(FEATURES_PATH, 256, search_temporal)
        text_feat_arr = preprocessing_text(text)
        D, I = faiss_db_t.search(text_feat_arr, k=int(k))
        search_results, search_temporal = transform_result(I, D,db_t) 
        search_results_A = link_scene(search_results_A,search_results)
        search_temporal = preprocess(dataset_index, search_temporal)


    response = flask.jsonify(search_results_A)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8506, debug=False)