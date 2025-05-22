# import numpy as np
# from PIL import Image
# from feature_extractor import FeatureExtractor
import math
import requests
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory
import os
import json
import pickle
import csv
from collections import OrderedDict
import sys
# from googletrans import Translator
from itertools import chain
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/base_2023/SIU_Pumpkin/")

#from utils.metadata_filter.metadata_filter import program_filter, datetime_filter
from models.envit5_translation import Translator
# from models.YOLOE_object_search import ObjectSearch
# from models.ShotClusterization import ShotClustering

app = Flask(__name__)
# app.debug = True
translator = Translator()
# o_search = ObjectSearch()
# cluster = ShotClustering()
true_frame_path = "/dataset/AIC2022/Keyframe_P_JSON"
basepath = "/dataset/AIC2022/"

path_a = "0"
path_b = "KeyFramesC00_V00"
path_c = "C00_V0000"

path_a_lst = ["0", "1"]
path_b_lst = ["KeyFramesC00_V0", "KeyFramesC01_V0", "KeyFramesC02_V0"]
path_c_lst = [str(i).zfill(2) for i in range(0, 100)]  # 00 --> 99

DATASET_PATH_ORIGIN = 'dataset/AIC2024/original_dataset/'
DATASET_PATH_TEAM = '/dataset/AIC2024/pumkin_dataset/'

@app.route('/img/<path:filename>')
def download_file(filename):
    directory = "/".join(filename.split("/")[:-1])
    video_name = filename.split("/")[-1]
    print(directory)
    print(video_name)
    return send_from_directory(directory="/" + directory, path=video_name)

@app.route('/video/<path:filename>/<path:keyframe>')
def video(filename, keyframe):
    filename = filename + '/' + keyframe
    filename = filename.split("/dataset/")[0]
    video_name = keyframe.split('/', keyframe.count("/"))[-2] #video in server
    frame_name = keyframe.split('/', keyframe.count("/"))[-1]

    true_id = int(frame_name.split('.')[0])
    fps = int(dict_fps[video_name.replace('.mp4','')])
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


def get_result_from_api(lst_video_name, result):
    lst_video_name_api = [(res['idx_folder'], res['video_name'], res['keyframe_id']) for res in result]           
    if len(lst_video_name)==0: 
        lst_video_name = lst_video_name_api
    else: 
        lst_video_name = list(set(lst_video_name) & set(lst_video_name_api))
    return lst_video_name

def TextTranslate(src_text):
  global translator
  translation = translator(src_text)
  return translation

def list_2_dict(list_result): 
    dict_result = OrderedDict()
    for d in list_result: 
        if d["video_name"] not in dict_result: 
            value = []
            value.append(d["keyframe_id"])
            dict_result[d["video_name"]] = value
            # print(dict_result[d["video_name"]])

        else: 
            value = dict_result[d["video_name"]]
            value.append(d["keyframe_id"])
            dict_result[d["video_name"]] = value
            # print(dict_result[d["video_name"]])

    return dict_result

def get_index_folder(video_name): 
    index_folder = int(video_name.split('_')[0][1:]) 
    if 0 <= index_folder <= 10: 
        index_folder = 0 
    elif 11 <= index_folder <= 20:
        index_folder = 1
    else: 
        index_folder = 2
    return index_folder

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #GET VALUES
        text = request.form['query']
        image = request.form['fname']
        obj = request.form['object'] 
        # s2t = request.form['s2t'] 
        #-----
        # program_query = request.form['program']
        model_port = request.form['model']
        translate = request.form['translate']
        history_name = request.form['history_name']
        his_1 = request.form['history_1']
        his_2 = request.form['history_2']
        logic = request.form['logic']
        k = request.form['k']
        #-------
        video_filter =  request.form['video_filter']
        time_in =  request.form['time_in']
        time_out =  request.form['time_out']
        
        #-----RESULT 
        lst_video_name = []

        if translate == "true": 
            text = TextTranslate(text)

        if text != "": 
            if int(model_port) == 8520:
                scene_1 = text.split('.')[0] 
                url_text = "http://192.168.1.252:{}/predict?text={}&k={}".format(8503,scene_1,k)
                result = requests.get(url_text).json() ######
                list_video = list_2_dict(result)
                all_scene = text
                url_text_temp = "http://192.168.1.252:{}/predict?text={}&video={}&k={}".format(int(model_port),all_scene,list_video,k)
                result = requests.get(url_text_temp).json()
                lst_video_name = get_result_from_api(lst_video_name,result)
            else: 
                url_text = "http://192.168.1.252:{}/predict?text={}&k={}".format(int(model_port),text,k)
                result = requests.get(url_text).json() ######
                lst_video_name = get_result_from_api(lst_video_name,result)

        if image != "": 
            url_text = "http://192.168.1.252:{}/predict?image_url={}&k={}".format(int(model_port)+10,image,k)
            result = requests.get(url_text).json()
            lst_video_name = get_result_from_api(lst_video_name,result)


        # if obj != "": 
        #     obj = obj.split(',', 50)
        #     obj = [tuple(item.strip().split(' ', 50)) for item in obj]
        #     print(obj)
        #     obj_search_result = o_search.forward(obj, int(k))
        #     result = [{'idx_folder':get_index_folder(item[0][0]), 'video_name':item[0][0]+'.mp4', 'keyframe_id':item[0][1]} for item in obj_search_result]
        #     lst_video_name = get_result_from_api(lst_video_name,result)

        # if s2t != "": 
        #     url_text = "http://192.168.1.252:8508/predict?text={}&k={}".format(s2t,k)
        #     result = requests.get(url_text).json() ######
        #     lst_video_name = get_result_from_api(lst_video_name,result)

        # #FILTER BY DATE 
        # if date_query != "": 
        #     lst_video_name = datetime_filter(lst_video_name,date_query)
        
        #FILTER BY PROGRAM
        # if program_query != "all": 
        #     if program_query=="60s": 
        #         program_query = "60 Giây Official"
        #     elif program_query=="htv_gt":
        #         program_query = "HTV Giải Trí"
        #     elif program_query=="htv_sport":
        #         program_query = "HTV Sports"
        #     elif program_query=="btt":
        #         program_query = "Báo Tuổi Trẻ"
        #     else:
        #         program_query = program_query
        #     print(program_query)
        #     lst_video_name = program_filter(lst_video_name,program_query)

        #SAVING HISTORY
        now = datetime.now()
        if history_name == "": 
            current_time = now.strftime("%H:%M:%S")
        else: 
            current_time = history_name
        info_current = {'text':text, 'image':image, 'obj':obj, 's2t':s2t, 'model_port': model_port}
        history[current_time] = (info_current,lst_video_name)
        history_query.append(text+'\n')

        #RETURN RESULT FROM HISTORY
        if his_1 != "---select---" and his_2 != "---select---": 
            if logic == 'and': 
                lst_video_name = history[his_1][1]
                lst_video_name = list(set(lst_video_name) & set(history[his_2][1]))
            elif logic == 'or':
                lst_video_name = []
                for his_1_value, his_2_value in zip(history[his_1][1], history[his_2][1]): 
                    lst_video_name.append(his_1_value)
                    lst_video_name.append(his_2_value)
            elif logic == "ofc": 
                same_value = history[his_1][1]
                same_value = list(set(lst_video_name) & set(history[his_2][1]))
                lst_video_name  = history[his_1][1] + list(set(same_value) - set(history[his_1][1]))
            else: 
                lst_video_name = list(set(history[his_1][1]) - set(history[his_2][1]))

        #VIDEO FILTERING 
        if video_filter != "": 
            lst_video_name = []
            fps = int(dict_fps[video_filter])

            if time_in == "": 
                time_in = 0
            else:
                if ":" in time_in: 
                    time_in = int(time_in.split(":")[0])*60*fps + int(time_in.split(":")[1])*fps
                else:
                    time_in = int(time_in)

            if time_out == "": 
                time_out = 90000
            else: 
                if ":" in time_out:
                    time_out = int(time_out.split(":")[0])*60*fps + int(time_out.split(":")[1])*fps
                else: 
                    time_out = int(time_out)
            order = int(video_filter.split("_")[0][1:])
            if 0 <= order <= 10: 
                index_folder = 0
            elif 11 <= order <= 20:
                index_folder = 1
            else: 
                index_folder = 2
            frame_folder_path = DATASET_PATH_TEAM + str(index_folder) + "/frames/pyscenedetect/Keyframes_L" + str(order).zfill(2) + "/keyframes/" + video_filter + '/'       
            list_frame_in_folder = []
            for file in sorted(os.listdir(frame_folder_path)): 
                if 'jpg' in file: 
                    list_frame_in_folder.append(int(file.replace(".jpg","")))
            
            list_frame_in_folder = list(filter(lambda x: time_in<= x <= time_out, list_frame_in_folder))
            list_frame_in_folder = [str(num).zfill(5) for num in list_frame_in_folder]
            lst_video_name = [(index_folder, video_filter+'.mp4', frame) for frame in list_frame_in_folder]           
        
        # #SUBMISSION VIEW 
        # dir_submission = '/workspace/competitions/AIC_2023/SIU_Pumpkin/submission/'
        # if sub_name != "---select---": 
        #     lst_video_name = []
        #     sub_file = dir_submission + sub_name
        #     with open(sub_file, mode='r') as f:
        #         sub_content = csv.reader(f, delimiter=',')
        #         for r in sub_content: 
        #             if 0 <= int(r[0].split('_')[0][1:]) <= 10:
        #                 index_folder = 0
        #             elif 11 <= int(r[0].split('_')[0][1:]) <= 20:
        #                 index_folder = 1
        #             else: 
        #                 index_folder = 2
        #             lst_video_name.append((index_folder,r[0]+'.mp4',r[1], dict_obj_date[r[0]], dict_obj_program[r[0]]))

        # #SAVING SUBMISSION 
        # if submission != "": 
        #     name = "/workspace/competitions/AIC_2023/SIU_Pumpkin/submission/" + submission + ".csv"
        #     with open(name, 'w', encoding='utf-8') as f:
        #         writer = csv.writer(f, delimiter=",", skipinitialspace=True)
        #         for tup in lst_video_name[0:100]:
        #         # for tup in lst_video_name:
        #             writer.writerow((tup[1].replace(".mp4",""), tup[2]))
        # list_sub_name = sorted(list(os.listdir(dir_submission)))
            
        # #WATCHING VIDEO 
        # if link_video != '': 
        #     video_info_to_watch.append(("/" + link_video.split('//')[2], int(link_video[-9:].split('.')[0])/25))
        # if watch_video == 'true':
        #     lst_video_name = []
        #     ping_source = video_info_to_watch[-1][0]
        #     ping_keyframe = video_info_to_watch[-1][1]
        #     watch_video_name = ping_source[-12:]
        # else: 
        #     ping_source = ''
        #     ping_keyframe = '' 
        #     watch_video_name = ''

        #RETURN VALUES
        files = []
        list_frames = []
        if int(k) > len(lst_video_name): 
            k = int(len(lst_video_name))
        for _, info in enumerate(lst_video_name):
        # for _, info in enumerate(lst_video_name[-int(k):]):
        

            #path 
            video_path = DATASET_PATH_ORIGIN + str(info[0]) +"/videos/Videos_" + str(info[1]).split("_")[0]  + "/video/" + str(info[1])
            frame_path = DATASET_PATH_TEAM + str(info[0]) +"/frames/pyscenedetect/Keyframes_" + str(info[1]).split("_")[0]  + "/keyframes/"  + str(info[1].split(".")[0]) + "/" + info[2] + ".jpg"
            
            fps = dict_fps[str(info[1]).replace('.mp4','')] 
            frame_name = info[2].replace("'","") + '.jpg'

            #get true time in video 
            true_id = int(info[2].replace("'",""))
            time = math.floor(true_id/fps)
            mi = str(time//60)
            if len(mi)==1: 
                mi="0"+mi
            se = str(int(time%60))
            if len(se)==1: 
                se="0"+se
            video_info  = info[1] + ", " + mi +":"+se + ', ' + str(int(fps))

            
            #objects and s2t content 
            obj_content = '' # dict_obj_text[info[1]][info[2] +'.jpg']
            print(info[1], info[2])
            s2t_content = dict_s2t[info[1]][info[2] +'.jpg']
            
            #list next frame
            frame_dir = os.path.dirname(frame_path)
            list_frame_in_dir = sorted(os.listdir(frame_dir))

            start_index = list_frame_in_dir.index(frame_name)
            end_index = len(list_frame_in_dir) - 1
            if end_index - start_index > 40: 
                end_index = start_index + 40 

            if start_index < 40: 
                start_index = 0 
            else: 
                start_index -= 40
            list_frame = [frame_dir + '/' + image for image in sorted(os.listdir(frame_dir))[start_index:end_index:2]]
            
            #full list
            list_frames.append(list_frame)
            files.append((_, frame_path, frame_name, video_info, video_path, s2t_content))

        #front-end support
        scroll = True
        return render_template('index.html', files=files, query=text, image=image, count=str(len(files)) +' files found.' , s2t=s2t, list_frames = json.dumps(list_frames), model=model_port, scroll=scroll, translate=translate, his=list(history.keys()), logic=logic, k = k, history_query=history_query)
    else:
        history.clear()
        history_query.clear()
        dir_submission = '/workspace/competitions/AIC_2024/SIU_Pumpkin/submission/'
        list_sub_name = sorted(list(os.listdir(dir_submission)))
        mypath = "/dataset/AIC2023/original_dataset/1/frames/Keyframes_L11/keyframes/L11_V001/"
        files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        files.sort()    
        scores = [('zero', f) for f in files]
        scroll = False
        return render_template('index.html', scores=scores, path_a=path_a_lst, path_b=path_b_lst, path_c=path_c_lst, original_path="1/frames/Keyframes_L11/keyframes/L11_V001", video_path=path_c, scroll=scroll, translate="false", logic='and', k=200, history_query='')
    

if __name__ == "__main__":
    # #METADATA 0
    # print('dataset 0...')
    s2t = '/dataset/AIC2024/pumkin_dataset/1/speech_to_text/transcript_all_segmented.json'
    # dict_fps = '/dataset/AIC2024/pumkin_dataset/0/util_files/video_fps_0.json'


    # with open(s2t, encoding='utf-8-sig') as json_file:
    #     dict_s2t = json.load(json_file)
    # with open(dict_fps, encoding='utf-8-sig') as json_file:
    #     dict_fps = json.load(json_file)

    #METADATA 1
    print('dataset 1...')

    s2t_1 = '/dataset/AIC2024/pumkin_dataset/1/speech_to_text/transcript_all_segmented.json'
    dict_fps_1 = '/dataset/AIC2024/pumkin_dataset/1/util_files/video_fps_test.json'

    with open(s2t_1, encoding='utf-8-sig') as json_file:
        dict_s2t_1 = json.load(json_file)    
    with open(dict_fps_1, encoding='utf-8-sig') as json_file:
        dict_fps_1 = json.load(json_file) 
    
    #  #METADATA 2
    # print('dataset 2... ')

    # s2t_2 = '/dataset/AIC2023/pumkin_dataset/2/speech_to_text/transcript_all_formated.json'
    # dict_fps_2 = '/dataset/AIC2023/pumkin_dataset/2/util_files/video_fps_2.json'

    # with open(s2t_2, encoding='utf-8-sig') as json_file:
    #     dict_s2t_2 = json.load(json_file)    
    # with open(dict_fps_2, encoding='utf-8-sig') as json_file:
    #     dict_fps_2 = json.load(json_file) 

    #TOTAL 
    # dict_s2t = dict_s2t | dict_s2t_1 | dict_s2t_2
    # dict_fps = dict_fps | dict_fps_1 | dict_fps_2
    
    dict_s2t = dict_s2t_1
    dict_fps = dict_fps_1
    
    #HISTORY SAVING 
    history = {}
    history_query = []
    video_info_to_watch = [('','')]

    #WEB RUN
    app.run("0.0.0.0", port=8501)

 