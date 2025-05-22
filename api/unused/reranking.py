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

DATASET_PATH_ORIGIN = 'dataset/AIC2023/original_dataset/'
DATASET_PATH_TEAM = '/dataset/AIC2024/pumkin_dataset/'
# SPLIT_NAME = 'pyscenedetect_t_5'
# SPLIT_NAME_LOW_RES = 'low_res_t_5'
# IMG_FORMAT = '.jpg'
# LOWRES_FORMAT = '.avif'

SPLIT_NAME = 'autoshot'
SPLIT_NAME_LOW_RES = 'autoshot'
IMG_FORMAT = '.jpg'
LOWRES_FORMAT = '.jpg'

method_object = CNN()

def get_split(video_name):
    if int(video_name[1:2])<=12:
        return "0"
    return "X"

def get_true_frame_path(frame_path):
    return DATASET_PATH_TEAM + get_split(frame_path[0:8]) + "/frames/" + SPLIT_NAME + "/Keyframes_" + frame_path[0:3] + "/keyframes/" + frame_path.replace(LOWRES_FORMAT, IMG_FORMAT)

def get_lum(r, g, b):
    return math.sqrt(0.241 * r + 0.691 * g + 0.068 * b)

def step(r, g, b, repetitions=1):
    lum = get_lum(r, g, b)

    h, s, v = colorsys.rgb_to_hsv(r, g, b)
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
        dominant_color = get_dominant_color(frame_path)
        frame_path.replace(SPLIT_NAME, SPLIT_NAME_LOW_RES).replace(IMG_FORMAT, LOWRES_FORMAT)
        step_sort = {'rgb': dominant_color, 'hsv': colorsys.rgb_to_hsv(*dominant_color), 'hls': colorsys.rgb_to_hls(*dominant_color)}
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
    json_path = os.path.join("/workspace/competitions/AIC_2024/SIU_Pumpkin/dataset/video_name", f"{videoName}.json")

    with open(json_path, 'r') as file:
        data = json.load(file)
    for i, list_frame in enumerate(data):
        print(list_frame)
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

# Gọi lên chạy 
app = flask.Flask("API Rerank")
app.config["DEBUG"] = True

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