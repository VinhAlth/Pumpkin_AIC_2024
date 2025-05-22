import os
import json
import math
import requests
import flask
from flask import Flask, request
import re
import ujson

DATASET_INDEX = "/dataset/AIC2024/pumkin_dataset/utils/index_autoshot"

def getShot(videoName, frameName):
    json_path = os.path.join("/workspace/competitions/AIC_2024/SIU_Pumpkin/dataset/video_name", f"{videoName}.json")

    with open(json_path, 'r') as file:
        data = json.load(file)
    for i, list_frame in enumerate(data):
        if int(list_frame[0][0][:-4])>int(frameName):
            return data[i-1]
    return data[len(data)-1]

def frame_to_index(index_by_video_path, video, frame):
    #open video index file
    with open(index_by_video_path + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    return index_dict[frame]

def listFrameRetrieval(frame_path):
    idx_folder = frame_path.split('/')[frame_path.split('/').index('pumkin_dataset') + 1]
    frame_path = frame_path.split('keyframes/')[-1]
    video_name, frame_name = frame_path.split('/')
    frame_number = int(frame_name.split('.')[0])  # Convert frame name to an integer
    # Get the shot information using the video name and frame number
    shot_infos = getShot(video_name, frame_number)
    results = []
    for idx, frames in enumerate(shot_infos[0]):
        keyframe_id = str(int(frames.split('.')[0])).zfill(5)
        frames_info = {"key":str(2703+idx),"idx_folder": str(idx_folder),"video_name":str(video_name) + '.mp4',
                        "keyframe_id": keyframe_id,
                        "score": str(0.2703-idx)}
        results.append(frames_info)
    #{'idx_folder': '0', 'key': '176545', 'keyframe_id': '21300', 'score': '0.5527009', 'video_name': 'L10_V013.mp4'}
    return results

# Gọi lên chạy 
app = flask.Flask("API Rerank")
app.config["DEBUG"] = True

@app.route('/get_shot')  # Specify POST method here
def get_shot():
    if request.method == "POST":
        data = request.get_json()  # Get JSON data from request body
        frame_path = data.get('frame_path', [])
    elif request.method == "GET":
        frame_path = request.args.get('frame_path')
    results = listFrameRetrieval(frame_path)
    
    
    response = flask.jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response

# if __name__ == '__main__':
#     method_object = CNN()
    #app.run(host="0.0.0.0", port= 8508, debug=False, threaded=True)