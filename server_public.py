import math
import requests
from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import ujson
from collections import OrderedDict
import pillow_avif
import sys
import json
import urllib.parse
import ast
import time

sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

app = Flask(__name__)

# app.debug = True
app.config["APPLICATION_ROOT"] = "/aic/1/"
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_HOST'] = 'localhost'
app.config['CACHE_REDIS_PORT'] = 6379

DATASET_PATH_ORIGIN = 'dataset/AIC2024/original_dataset/'
DATASET_PATH_TEAM = '/dataset/AIC2024/pumkin_dataset/'
#SPLIT_NAME = 'pyscenedetect'
#SPLIT_NAME = 'low_res_t_5'
SPLIT_NAME = 'low_res_autoshot'

IMG_FORMAT = '.avif'
#IMG_FORMAT = '.jpg'

#init
frame_dir_dict = {}
    

print("Server boosted!")


def convert_time_to_frame(video_name, input_time):
    parts = input_time.split(":")
    try:
        url_text = "https://phamgiakiet273.ngrok.app/scroll?k={}&video_filter={}&time_in={}&time_out={}".format(1,video_name,0,0)
        result = requests.get(url_text).json()
        fps = result[0]['fps']
    except:
        url_text = "https://api.siu.edu.vn/aic/3/scroll?k={}&video_filter={}&time_in={}&time_out={}".format(1,video_name,0,0)
        result = requests.get(url_text).json()
        fps = result[0]['fps']
    return int(float(fps)*(60*int(parts[0])+int(parts[1])))

@app.route('/img/<path:filename>')
def download_file(filename): 
    filename = filename.rstrip('/')
    directory = os.path.dirname(filename)
    video_name = os.path.basename(filename)
    return send_from_directory(directory="/" + directory, path=video_name)
    
@app.route('/video/<path:filename>/<path:keyframe>')
def video(filename, keyframe):
    filename = filename + '/' + keyframe
    filename = filename.split("/dataset/")[0]
    video_name = keyframe.split('/', keyframe.count("/"))[-2] #video in server
    frame_name = keyframe.split('/', keyframe.count("/"))[-1]

    true_id = int(frame_name.split('.')[0])
    try:
        url_text = "https://phamgiakiet273.ngrok.app/scroll?k={}&video_filter={}&time_in={}&time_out={}".format(1,video_name,0,0)
        result = requests.get(url_text).json()
        fps = result[0]['fps']
    except:
        url_text = "https://api.siu.edu.vn/aic/3/scroll?k={}&video_filter={}&time_in={}&time_out={}".format(1,video_name,0,0)
        result = requests.get(url_text).json()
        fps = result[0]['fps']
        
    true_id = int(true_id) / fps
    mi = str(int(true_id//60))
    if len(mi)==1: 
        mi="0"+mi
    se = str(int(true_id%60))
    if len(se)==1: 
        se="0"+se
    video_info = video_name + ", " + mi +":"+se + ', '+ str(int(fps))
    return render_template('video.html', source=filename, keyframe=true_id, id=video_info)

@app.route('/keyframes/<path:keyframe>')
def keyframes(keyframe):
    keyframes_path = os.path.dirname(keyframe)
    frame_name = keyframe.split('/', keyframe.count("/"))[-1]
    list_frame = []
    list_from_dir = sorted(os.listdir('/'+keyframes_path))
    id_frame_name = list_from_dir.index(frame_name)
    if id_frame_name < 20: 
        start_id = 0
    else: 
        start_id = id_frame_name - 50 
    
    if len(list_from_dir) - id_frame_name < 20: 
        end_id = len(list_from_dir)
    else: 
        end_id = id_frame_name + 50 
    for i in sorted(os.listdir('/'+keyframes_path))[start_id:end_id]: 
        list_frame.append([keyframes_path+'/'+i,i])
    return render_template('keyframes.html', files=list_frame, current_frame=frame_name)


@app.route('/get_rerank', methods=['GET', 'POST'])
def get_rerank():
    if request.method == 'POST':
        
        #GET VALUES
        data = request.get_json()

        # Get values from the JSON payload
        rerank =  urllib.parse.unquote(data.get('method'))
        shortened_files_str = urllib.parse.unquote(data.get('files'))
        shortened_files = ast.literal_eval(shortened_files_str)

        url_rerank = "https://api.siu.edu.vn/aic/4/rerank"

        # Create payload for POST request
        payload = {
            'files': shortened_files,
            'method': rerank
        }


        response = requests.post(url_rerank, json=payload)
        response.raise_for_status()
        shortened_files = response.json()

        return jsonify(shortened_files)

@app.route('/get_session_id', methods=['GET', 'POST'])
def get_session_id():
    username = 'team53'
    password = 'dFMgSERanV'
    url_login = "https://eventretrieval.one/api/v2/login"
    payload = {
        "username": username,
        "password": password
    }

    response = requests.post(url_login, json=payload)
    return_response = response.json()['sessionId']

    return jsonify(return_response)

@app.route('/get_eval_id', methods=['GET', 'POST'])
def get_eval_id():
    session_id = "XpofF0Coh9HF_bO-a03-Yja7gQMa5syD"
    url_eval = "https://eventretrieval.one/api/v2/client/evaluation/list?session={}".format(session_id)
    response = requests.get(url_eval)
    return_response = response.json()
    return jsonify(return_response)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    
    eval_id = "e3ed921d-af52-45b8-96fc-f86bcc7a39e4"
    session_id = "XpofF0Coh9HF_bO-a03-Yja7gQMa5syD"
    
    if request.method == 'POST':
        
        #GET VALUES
        data = request.get_json()
        # Get values from the JSON payload
        try:
            fps = float(urllib.parse.unquote(data.get('fps')))
        except:
            fps = float(data.get('fps'))
        video_name =  urllib.parse.unquote(data.get('video_name'))
        frame_name =  urllib.parse.unquote(data.get('frame_name'))
        qa = urllib.parse.unquote(data.get('qa'))
        url_submit = "https://eventretrieval.one/api/v2/submit/{}?session={}".format(eval_id, session_id)
        
        payload = {}
        
        if qa=="":
            # Create payload for POST request
            milisec = int(float(frame_name)/fps*1000)
            payload = {
                    "answerSets": [
                        {
                        "answers": [
                            {
                            "mediaItemName": video_name,
                            "start": milisec,
                            "end": milisec
                            }
                        ]
                        }
                    ]
                    }
        else:
            milisec = int(float(frame_name)/fps*1000)
            # Create payload for POST request
            payload = {
                "answerSets": [
                    {
                    "answers": [
                        {
                        "text": "{}-{}-{}".format(qa, video_name, str(milisec))
                        }
                    ]
                    }
                ]
                }

        print(payload)
        
        response = requests.post(url_submit, json=payload)
        return_response = response.json()
    
        
        return jsonify(return_response)


# @app.route('/register_last_frame', methods=['POST'])
# def register_last_frame():
#     global last_frame
#     if request.method == 'POST':
        
#         #GET VALUES
#         data = request.get_json()

#         # Get values from the JSON payload
#         video_name =  urllib.parse.unquote(data.get('video_name'))
#         frame_name =  urllib.parse.unquote(data.get('frame_name'))

#     url_register_last_frame = "https://api.siu.edu.vn/aic/1/register_last_frame"

#     # Create payload for POST request
#     payload = {
#         'video_name': video_name,
#         'frame_name': frame_name
#     }


#     response = requests.post(url_register_last_frame, json=payload)

#     last_frame = (video_name, frame_name)
    
#     response = jsonify({"video_name": video_name, "frame_name": frame_name})
    
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.success = True
    
#     return response

@app.route('/register_last_frame', methods=['POST'])
def register_last_frame():
    global last_frame
    if request.method == 'POST':
        
        #GET VALUES
        data = request.get_json()

        # Get values from the JSON payload
        video_name =  urllib.parse.unquote(data.get('video_name'))
        frame_name =  urllib.parse.unquote(data.get('frame_name'))

    with open("last_frame.json",'w') as read_file:
        ujson.dump({
            "video_name": video_name,
            "frame_name": frame_name
        }, read_file, indent=4)
    
    response = jsonify({"video_name": video_name, "frame_name": frame_name})
    
    
    return response

# @app.route('/get_last_frame', methods=['POST', 'GET'])
# def get_last_frame():
#     global last_frame
#     url_get_last_frame = "https://api.siu.edu.vn/aic/1/get_last_frame"
#     data = requests.post(url_get_last_frame)
    
#     data = data.json()

#     # Get values from the JSON payload
#     video_name =  urllib.parse.unquote(data.get('video_name'))
#     frame_name =  urllib.parse.unquote(data.get('frame_name'))

#     response = jsonify({"video_name": video_name, "frame_name": frame_name})
#     last_frame = (video_name, frame_name)
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.success = True
#     return response

@app.route('/get_last_frame', methods=['POST', 'GET'])
def get_last_frame():

    # Get values from the JSON payload
    video_name =  ""
    frame_name =  ""

    with open("last_frame.json",'r') as read_file:
        last_frame = ujson.load(read_file)
        video_name = last_frame["video_name"]
        frame_name = last_frame["frame_name"]

    response = jsonify({"video_name": video_name, "frame_name": frame_name})
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    st = time.time()
    if request.method == 'POST':
    
        #GET VALUES
        text = request.form['query']
        image = request.form['fname']
        translate = request.form['translate']
        model_port = request.form['model']
        k = request.form['k']
        video_filter =  request.form['video_filter']
        time_in =  request.form['time_in']
        time_out =  request.form['time_out']
        rerank = request.form['rerank']
        s2t_switch = request.form['s2t_switch']
        batchs = request.form.getlist('batch')
        s2t_filter = request.form['query_s2t']
        #-----RESULT 
        lst_video_name = []
        
        #print(s2t_filter)

        if ":" in time_in:
            time_in = convert_time_to_frame(video_filter, time_in)
        if ":" in time_out:
            time_out = convert_time_to_frame(video_filter, time_out)
        
        if "BACKUP" in model_port:
            if text != "":
                if "shot-" in text:
                    url_text = "https://phamgiakiet273.ngrok.app/get_shot?frame_path={}".format(text)
                    result = requests.get(url_text).json()
                    lst_video_name = result
                elif '/' in text:
                    if "http" not in text:
                        text = "E:/SIU_Pumpkin/dataset/pumpkin_dataset/" + text[text.find("frames")-2:]
                    #url_text = "https://api.siu.edu.vn/aic/3/image_search?image_url={}&k={}&video_filter={}&time_in={}&time_out={}".format("/dataset/AIC2024/pumkin_dataset/" + text[text.find("frames")-2:],k,video_filter,time_in,time_out)
                    url_text = "https://phamgiakiet273.ngrok.app/image_search?image_url={}&k={}&video_filter={}&time_in={}&time_out={}&batchs={}&s2t_filter={}".format(text,k,video_filter,time_in,time_out, str(batchs), s2t_filter)
                        
                    result = requests.get(url_text).json()
                    lst_video_name = result
                else:
                    if translate=="YES":
                        url_translation = "https://phamgiakiet273.ngrok.app/trans?text={}".format(text)
                        text = requests.get(url_translation).json()
                    if model_port=="BACKUP_SIGLIP":
                        url_text = "https://phamgiakiet273.ngrok.app/text_search?text={}&k={}&video_filter={}&time_in={}&time_out={}&batchs={}&s2t_filter={}".format(text,k,video_filter,time_in,time_out, str(batchs), s2t_filter)
                        result = requests.get(url_text).json() ######
                        lst_video_name = result
                    elif model_port=="BACKUP_TEMPORAL_SIGLIP":
                        url_text = "https://phamgiakiet273.ngrok.app/temporal_search?text={}&k={}&video_filter={}&time_in={}&time_out={}&batchs={}&s2t_filter={}".format(text,k,video_filter,time_in,time_out, str(batchs), s2t_filter)
                        result = requests.get(url_text).json() ######
                        lst_video_name = result
            elif video_filter!="":
                video_filter = str(video_filter).upper()
                url_text = "https://phamgiakiet273.ngrok.app/scroll?k={}&video_filter={}&time_in={}&time_out={}&s2t_filter={}".format(k,video_filter,time_in,time_out, s2t_filter)
                result = requests.get(url_text).json()
                lst_video_name = result
        else:    
            if text != "":
                if "duplicate-" in text:
                    url_text = "https://api.siu.edu.vn/aic/4/get_duplicate?frame_path={}".format(text)
                    result = requests.get(url_text).json()
                    lst_video_name = result                
                elif "shot-" in text:
                    url_text = "https://api.siu.edu.vn/aic/4/get_shot?frame_path={}".format(text)
                    result = requests.get(url_text).json()
                    lst_video_name = result
                elif '/' in text:
                    if model_port=="MOBILE":
                        #url_text = "https://api.siu.edu.vn/aic/2/image_search?image_url={}&k={}&video_filter={}&time_in={}&time_out={}".format("/dataset/AIC2024/pumkin_dataset/" + text[text.find("frames")-2:],k,video_filter,time_in,time_out)
                        if "http" not in text:
                            text = "/dataset/AIC2024/pumkin_dataset/" + text[text.find("frames")-2:]
                        url_text = "https://api.siu.edu.vn/aic/2/image_search?image_url={}&k={}&video_filter={}&time_in={}&time_out={}&batchs={}&s2t_filter={}".format(text,k,video_filter,time_in,time_out, str(batchs), s2t_filter)
                    else:
                        if "http" not in text:
                            text = "/dataset/AIC2024/pumkin_dataset/" + text[text.find("frames")-2:]
                        #url_text = "https://api.siu.edu.vn/aic/3/image_search?image_url={}&k={}&video_filter={}&time_in={}&time_out={}".format("/dataset/AIC2024/pumkin_dataset/" + text[text.find("frames")-2:],k,video_filter,time_in,time_out)
                        url_text = "https://api.siu.edu.vn/aic/3/image_search?image_url={}&k={}&video_filter={}&time_in={}&time_out={}&batchs={}&s2t_filter={}".format(text,k,video_filter,time_in,time_out, str(batchs), s2t_filter)
                        
                    result = requests.get(url_text).json()
                    lst_video_name = result
                else:
                    if translate=="YES":
                        url_translation = "https://api.siu.edu.vn/aic/4/preprocess?text={}".format(text)
                        text = requests.get(url_translation).json()
                    if model_port=="SIGLIP":
                        url_text = "https://api.siu.edu.vn/aic/3/text_search?text={}&k={}&video_filter={}&time_in={}&time_out={}&batchs={}&s2t_filter={}".format(text,k,video_filter,time_in,time_out, str(batchs), s2t_filter)
                        result = requests.get(url_text).json() ######
                        lst_video_name = result
                    elif model_port=="DFN5B":
                        url_text = "https://api.siu.edu.vn/aic/2/text_search?text={}&k={}&video_filter={}&time_in={}&time_out={}&batchs={}&s2t_filter={}".format(text,k,video_filter,time_in,time_out, str(batchs), s2t_filter)
                        result = requests.get(url_text).json() ######
                        lst_video_name = result
                    elif model_port=="TEMPORAL_SIGLIP":
                        url_text = "https://api.siu.edu.vn/aic/3/temporal_search?text={}&k={}&video_filter={}&time_in={}&time_out={}&batchs={}&s2t_filter={}".format(text,k,video_filter,time_in,time_out, str(batchs), s2t_filter)
                        result = requests.get(url_text).json() ######
                        lst_video_name = result
                    elif model_port=="TEMPORAL_DFN5B":
                        url_text = "https://api.siu.edu.vn/aic/2/temporal_search?text={}&k={}&video_filter={}&time_in={}&time_out={}&batchs={}&s2t_filter={}".format(text,k,video_filter,time_in,time_out, str(batchs), s2t_filter)
                        result = requests.get(url_text).json() ######
                        lst_video_name = result
            elif video_filter!="":
                video_filter = str(video_filter).upper()
                url_text = "https://api.siu.edu.vn/aic/3/scroll?k={}&video_filter={}&time_in={}&time_out={}&s2t_filter={}".format(k,video_filter,time_in,time_out, s2t_filter)
                result = requests.get(url_text).json()
                lst_video_name = result
        
        #print(time.time()-st)
        #RETURN VALUES
        files = []
        list_frames = []
        if int(k) > len(lst_video_name): 
            k = int(len(lst_video_name))

        for _, info in enumerate(lst_video_name):
            video_path = DATASET_PATH_ORIGIN + str(info['idx_folder']) + "/videos/Videos_" + str(info['video_name']).split("_")[0]  + "/video/" + str(info['video_name'])
            frame_path = DATASET_PATH_TEAM + str(info['idx_folder']) + "/frames/" + SPLIT_NAME + "/Keyframes_" + str(info['video_name']).split("_")[0]  + "/keyframes/"  + str(info['video_name'].split(".")[0]) + "/" + info['keyframe_id'] + IMG_FORMAT

            fps = float(info['fps']) 
            frame_name = info['keyframe_id'].replace("'","") + IMG_FORMAT

            frame_status = "GREEN"
            if int(info['duplicate_status'][0])==1:
                frame_status = 'RED'
            
            
            #get true time in video 
            true_id = int(info['keyframe_id'].replace("'",""))
            video_time = math.floor(true_id/fps)
            mi = str(video_time//60)
            if len(mi)==1: 
                mi="0"+mi
            se = str(int(video_time%60))
            if len(se)==1: 
                se="0"+se
            video_info  = info['video_name'] + ", " + mi +":"+se + ', ' + str(float(fps))

            
            #objects and s2t content 
            #print(info[1], info[2])
            if s2t_switch == "ON":
                s2t_content = info['s2t']
            else:
                s2t_content = ""
            
            #list next frame
            frame_dir = os.path.dirname(frame_path)
            if frame_dir not in frame_dir_dict.keys():
                frame_dir_dict[frame_dir] = sorted(os.listdir(frame_dir))
                
            list_frame_in_dir = frame_dir_dict[frame_dir]

            start_index = list_frame_in_dir.index(frame_name)
            end_index = len(list_frame_in_dir) - 1
            if end_index - start_index > 20: 
                end_index = start_index + 20 

            if start_index < 20: 
                start_index = 0 
            else: 
                start_index -= 20
            list_frame = [frame_dir + '/' + image for image in list_frame_in_dir[start_index:end_index:2]]
            
            #full list
            list_frames.append(list_frame)
            files.append((_, frame_path, frame_name, video_info, video_path, s2t_content, frame_status, fps))
        shortened_files = [[item[0], "/".join(item[1].split("/")[-2:])] for item in files]

        # #DEFAULT SHOT SORT
        # url_rerank = "https://api.siu.edu.vn/aic/4/rerank"
        # # Create payload for POST request
        # payload = {
        #     'files': shortened_files,
        #     'method': "SHOT"
        # }
        # response = requests.post(url_rerank, json=payload)
        # shortened_files = response.json()
    
        # index_to_position = {index: pos for pos, (index, _) in enumerate(shortened_files)}
        # files = sorted(files, key=lambda x: index_to_position.get(x[0], float('inf')))
        scroll = True

        return render_template('index_public.html', files=files, shortened_files=shortened_files, query=text, image=image, count=str(len(files)) +' files found.' , s2t={}, list_frames = json.dumps(list_frames), model=model_port, scroll=scroll, k = k)
    else:
        return render_template('index_public.html')


 