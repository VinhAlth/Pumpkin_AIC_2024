import math
import requests
import flask
from flask import Flask, request
import os
from collections import OrderedDict
import pillow_avif
import sys
from imagededup.methods import CNN
import json
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")
from utils.sort_by_color import get_dominant_color
import colorsys
import ast
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import unicodedata
import ujson
import torch

DATASET_PATH_ORIGIN = 'dataset/AIC2023/original_dataset/'
DATASET_PATH_TEAM = '/dataset/AIC2024/pumkin_dataset/'
# SPLIT_NAME = 'pyscenedetect_t_5'
# SPLIT_NAME_LOW_RES = 'low_res_t_5'
# IMG_FORMAT = '.jpg'
# LOWRES_FORMAT = '.avif'
DATASET_INDEX = "/dataset/AIC2024/pumkin_dataset/utils/index_autoshot"

SPLIT_NAME = 'autoshot'
SPLIT_NAME_LOW_RES = 'low_res_autoshot'
IMG_FORMAT = '.jpg'
LOWRES_FORMAT = '.avif'

s2t_0 = '/dataset/AIC2024/pumkin_dataset/0/speech_to_text/transcript_all_autoshot_segmented.json'
s2t_1 = '/dataset/AIC2024/pumkin_dataset/1/speech_to_text/transcript_all_autoshot_segmented.json'
s2t_2 = '/dataset/AIC2024/pumkin_dataset/2/speech_to_text/transcript_all_autoshot_segmented.json'

with open(s2t_0, encoding='utf-8-sig') as json_file:
    dict_s2t_0 = ujson.load(json_file)
with open(s2t_1, encoding='utf-8-sig') as json_file:
    dict_s2t_1 = ujson.load(json_file)
with open(s2t_2, encoding='utf-8-sig') as json_file:
    dict_s2t_2 = ujson.load(json_file)
    
dict_s2t = dict_s2t_0 | dict_s2t_1 | dict_s2t_2
print("S2T Dict loaded")

duplicate_0 = '/dataset/AIC2024/pumkin_dataset/0/utils/duplicate_0_all.json'
duplicate_1 = '/dataset/AIC2024/pumkin_dataset/1/utils/duplicate_1_all.json'
duplicate_2 = '/dataset/AIC2024/pumkin_dataset/2/utils/duplicate_2_all.json'

with open(duplicate_0, encoding='utf-8-sig') as json_file:
    dict_duplicate_0 = ujson.load(json_file)
with open(duplicate_1, encoding='utf-8-sig') as json_file:
    dict_duplicate_1 = ujson.load(json_file)
with open(duplicate_2, encoding='utf-8-sig') as json_file:
    dict_duplicate_2 = ujson.load(json_file)
dict_duplicate = dict_duplicate_0 | dict_duplicate_1 | dict_duplicate_2
print("Duplicate Dict loaded")

dict_fps_0 = '/dataset/AIC2024/pumkin_dataset/0/utils/video_fps_0.json'
dict_fps_1 = '/dataset/AIC2024/pumkin_dataset/1/utils/video_fps_1.json'
dict_fps_2 = '/dataset/AIC2024/pumkin_dataset/2/utils/video_fps_2.json'

with open(dict_fps_0, encoding='utf-8-sig') as json_file:
    dict_fps_0 = ujson.load(json_file)
with open(dict_fps_1, encoding='utf-8-sig') as json_file:
    dict_fps_1 = ujson.load(json_file)
with open(dict_fps_2, encoding='utf-8-sig') as json_file:
    dict_fps_2 = ujson.load(json_file)    
dict_fps = dict_fps_0 | dict_fps_1 | dict_fps_2
print("FPS Dict loaded")

# {
#     "L13_V001.mp4": {
#         "00000.jpg": [


def merge_and_remove_accents(word_list):
    # Merge the array of words into a sentence
    sentence = ' '.join(word_list)
    # Normalize the text to decompose accented characters
    nfkd_form = unicodedata.normalize('NFKD', sentence)
    # Filter out combining characters (marks) and return the plain text
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)]).replace('đ', 'd')



method_object = CNN()

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")
device_vi2en = torch.device("cuda")
model_vi2en.to(device_vi2en)


list_shot_dict = {}
list_colorsys = {}
list_lum = {}

def translate_vi2en(vi_texts: str) -> str:
    input_ids = tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(device_vi2en)
    output_ids = model_vi2en.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    en_texts = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    return en_texts





def get_split(video_name):
    #print(video_name)
    if int(video_name[1:3])<=12:
        return "0"
    elif int(video_name[1:3])<=24:
        return "1"
    return "2"

def get_true_frame_path(frame_path):
    return DATASET_PATH_TEAM + get_split(frame_path[0:8]) + "/frames/" + SPLIT_NAME + "/Keyframes_" + frame_path[0:3] + "/keyframes/" + frame_path.replace(LOWRES_FORMAT, IMG_FORMAT)

def get_lum(r, g, b):
    return math.sqrt(0.241 * r + 0.691 * g + 0.068 * b)

def step(r, g, b, repetitions=1):
    lum = get_lum(r, g, b)

    if (r,g,b) not in list_lum.keys():
        list_lum[(r,g,b)] = colorsys.rgb_to_hsv(r, g, b)
    h, s, v = list_lum[(r,g,b)]
    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)

    if h2 % 2 == 1:
        v2 = repetitions - v2
        lum = repetitions - lum

    return (h2, lum, v2)

def sort_by_color(files):
    files_sorted = []
    for index, list_frame in enumerate(files):
        frame_path = files[index][1]
        frame_path = get_true_frame_path(frame_path)
        if frame_path not in list_colorsys.keys():
            json_path = frame_path.replace(SPLIT_NAME, "colorsys_"+SPLIT_NAME+'/'+SPLIT_NAME).replace(IMG_FORMAT, ".json").replace("/frames/","/")
            with open(json_path,'r') as infile:
                data = ujson.load(infile)
            list_colorsys[frame_path]=data
        step_sort = list_colorsys[frame_path]
        frame_path.replace(SPLIT_NAME, SPLIT_NAME_LOW_RES).replace(IMG_FORMAT, LOWRES_FORMAT)
        files_sorted.append((files[index][0], frame_path, step_sort))
    files_sorted = sorted(files_sorted, key=lambda k: k[len(k)-1]['hls'])
    files_sorted = sorted(files_sorted, key=lambda k: get_lum(*k[len(k)-1]['rgb']))
    files_sorted = sorted(files_sorted, key=lambda k: step(k[len(k)-1]['rgb'][0], k[len(k)-1]['rgb'][1], k[len(k)-1]['rgb'][2], 8))
    files_sorted = [[item[0],item[1]] for item in files_sorted]
    return files_sorted

def group_images(duplicates):
    visited = set()
    groups = []

    def dfs(image, group):
        if image not in visited:
            visited.add(image)
            group.add(image)
            for dup in duplicates.get(image, []):
                if dup not in visited:
                    dfs(dup, group)

    for image in duplicates:
        if image not in visited:
            group = set()
            dfs(image, group)
            # Append the group, ensuring all images including those with no duplicates are included
            groups.append(sorted(group))

    return groups

def sort_by_duplicate(files):
    files_sorted = []
    encoding_maps = {}
    image_to_group = {}
    image_index = {}
    
    files = [[item[0], get_true_frame_path(item[1])] for item in files]
    
    for index, list_frame in enumerate(files):
        frame_path = files[index][1].replace(SPLIT_NAME_LOW_RES, SPLIT_NAME).replace(LOWRES_FORMAT, IMG_FORMAT)
        image_index[frame_path] = index
        #encoding_maps[frame_path] = method_object.encode_image(image_file=frame_path)[0]
        encoding_maps[frame_path] = np.load(frame_path.replace('/frames/', '/imagededup_autoshot/').replace('.jpg', '.npy'))[0]
    
    duplicates = method_object.find_duplicates(encoding_map=encoding_maps, min_similarity_threshold=0.75)
    groups = group_images(duplicates)
    
    # Determine the order of groups based on the index of the first image in each group
    group_order = []
    for group in groups:
        # Find the index of the first image in the group
        min_index = min(image_index.get(image, float('inf')) for image in group)
        group_order.append((min_index, group))
    
    # Sort groups based on the minimum index of the images
    group_order.sort(key=lambda x: x[0])

    # Rebuild the groups list in the new order
    sorted_groups = [group for _, group in group_order]

    # Sort images within each group by their order in `files`
    for i in range(len(sorted_groups)):
        sorted_groups[i].sort(key=lambda image: image_index.get(image, float('inf')))

    # Map each image to its group index
    for group_index, group in enumerate(sorted_groups):
        for image in group:
            image_to_group[image] = group_index
    
    # Sort files based on the group index of the image paths
    files_sorted = sorted(files, key=lambda file: image_to_group.get(file[1].replace(SPLIT_NAME_LOW_RES, SPLIT_NAME).replace(LOWRES_FORMAT, IMG_FORMAT), float('inf')))
    return files_sorted

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

def sort_by_shot(files):
    shot_groups = {}
    image_index = {}

    # Create a mapping of each file path to its original index
    for index, (file_index, frame_path) in enumerate(files):
        image_index[frame_path] = file_index

    # Group frames by shots using getShot
    for file_index, frame_path in files:
        # Extract video name and frame number from the frame path
        video_name, frame_name = frame_path.split('/')
        frame_number = int(frame_name.split('.')[0])  # Convert frame name to an integer
        
        # Get the shot information using the video name and frame number
        shot = getShot(video_name, frame_number)

        # Convert the shot list to a tuple to make it hashable
        shot_key = tuple(shot[0])  # Assuming first element is the list of frames

        # Create a group for each shot if not already present
        if shot_key not in shot_groups:
            shot_groups[shot_key] = []
        shot_groups[shot_key].append((file_index, frame_path))

    # Sort the groups by the minimum index of the frames within each group
    sorted_groups = sorted(
        shot_groups.values(),
        key=lambda group: min(image_index[frame_path] for _, frame_path in group)
    )

    # Sort the frames within each group by their index
    for group in sorted_groups:
        group.sort(key=lambda item: image_index[item[1]])

    # Flatten the sorted groups back into a single list of files
    sorted_files = [item for group in sorted_groups for item in group]

    return sorted_files


def get_duplicate_frame_in_other(frame_path):
    #open duplicate file
    duplicate_dict = []
    
    with open(frame_path.replace("frames/"+SPLIT_NAME, "duplicate_json/").replace(".jpg",".json"), 'r') as duplicate_json:
        duplicate_dict = ujson.load(duplicate_json)
    
    return duplicate_dict
    
    



def frame_to_index(index_by_video_path, video, frame):
    #open video index file
    with open(index_by_video_path + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    return index_dict[frame]

def listFrameRetrieval(frame_path):
    try:
        idx_folder = frame_path.split('/')[frame_path.split('/').index('pumkin_dataset') + 1]
    except:
        idx_folder = frame_path.split('/')[frame_path.split('/').index('pumpkin_dataset') + 1]
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
                        "duplicate_status": [dict_duplicate[str(video_name)][str(keyframe_id)]],
                        "s2t": merge_and_remove_accents(dict_s2t[str(video_name) + ".mp4"][str(keyframe_id)+'.jpg']),
                        "fps": dict_fps[str(video_name)],
                        "score": str(0.2703-idx)}
        results.append(frames_info)
    #{'idx_folder': '0', 'key': '176545', 'keyframe_id': '21300', 'score': '0.5527009', 'video_name': 'L10_V013.mp4'}
    return results

# Gọi lên chạy 
app = flask.Flask("API Rerank")
app.config["DEBUG"] = True
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_HOST'] = 'localhost'
app.config['CACHE_REDIS_PORT'] = 6379

@app.route('/get_shot')  # Specify POST method here
def get_shot():
    if request.method == "POST":
        data = request.get_json()  # Get JSON data from request body
        frame_path = data.get('frame_path', [])
    elif request.method == "GET":
        frame_path = request.args.get('frame_path')
    
    #print(frame_path)    
    
    results = listFrameRetrieval(frame_path)
    
    
    response = flask.jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response


@app.route('/get_duplicate')  # Specify POST method here
def get_duplicate():
    if request.method == "POST":
        data = request.get_json()  # Get JSON data from request body
        frame_path = data.get('frame_path', [])
    elif request.method == "GET":
        frame_path = request.args.get('frame_path')
    
    #print(frame_path)
    
    frame_path = get_true_frame_path(frame_path.split('keyframes/')[-1])    
    duplicate_frames = get_duplicate_frame_in_other(frame_path)

    response = flask.jsonify(duplicate_frames)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response


@app.route('/preprocess')
def preprocess():
    text = ""
    
    if request.method == "POST":
        text = request.json['text']
    else:
        text = request.args.get('text')
    
    text.rstrip('.')
    
    parts = text.split('"')
    #print(parts)
    translated_text = ""
    for i, part in enumerate(parts):
        if len(part)==0:
            continue
        if i % 2 == 0:  # Even indices: unquoted parts (to be translated)
            translated_text += translate_vi2en(part)[0].rstrip('.') + " "
        else:  # Odd indices: quoted parts (to remain unchanged)
            translated_text += part + " "
    
    translated_text.rstrip(" ")
    #print(translated_text)
    response = flask.jsonify(translated_text)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response

@app.route('/rerank', methods=['POST'])  # Specify POST method here
def rerank():
    if request.method == "POST":
        data = request.get_json()  # Get JSON data from request body
        files = data.get('files', [])
        method = data.get('method', '')
    
    #files = ast.literal_eval(files)

    if method=="STEP":
        files = sort_by_color(files)
    elif method=="IMAGEDEDUP":
        files = sort_by_duplicate(files)
    elif method=="SHOT":
        files = sort_by_shot(files)
        #files = files
    else:
        files=files
    
    
    response = flask.jsonify(files)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response

# if __name__ == '__main__':
#     method_object = CNN()
    #app.run(host="0.0.0.0", port= 8506, debug=False, threaded=True)