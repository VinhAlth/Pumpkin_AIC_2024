from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import subprocess
from tqdm import tqdm
import os
import numpy as np
import json
import ujson
import time
import unicodedata

DATASET_PATH_TEAM = "/dataset/AIC2024/pumkin_dataset/"
DATASET_INDEX = "/dataset/AIC2024/pumkin_dataset/utils/index_autoshot"

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

# {
#     "L13_V001.mp4": {
#         "00000.jpg": [
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

def merge_and_remove_accents(word_list):
    # Merge the array of words into a sentence
    sentence = ' '.join(word_list)
    # Normalize the text to decompose accented characters
    nfkd_form = unicodedata.normalize('NFKD', sentence)
    # Filter out combining characters (marks) and return the plain text
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)]).replace('đ', 'd')

def merge_and_remove_accents_no_list(sentence):
    # Normalize the text to decompose accented characters
    nfkd_form = unicodedata.normalize('NFKD', sentence)
    # Filter out combining characters (marks) and return the plain text
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)]).replace('đ', 'd')

def get_split(video_name):
    #print(video_name)
    if int(video_name[1:3])<=12:
        return "0"
    elif int(video_name[1:3])<=24:
        return "1"
    return "2"


def get_videos_from_split(theme):
    
    allowed_video_list = []
    theme = int(theme)
    #general news
    if theme==0:
        allowed_video_list = ['L01','L02','L03','L04','L05','L06','L07',
                              'L08','L09','L10','L11','L12','L13','L14',
                              'L15','L16','L17','L18','L19','L20','L21','L22']
    #bike racing
    elif theme==1:
        allowed_video_list = ['L23']
    #dragon dancing
    elif theme==2:
        allowed_video_list = ['L24']
    #online courses
    elif theme==3:
        allowed_video_list = ['L25']
    #cooking lessons
    elif theme==4:
        allowed_video_list = ['L26']
    #tourism
    elif theme==5:
        allowed_video_list = ['L27', 'L28', 'L29', 'L30']                   
    #print(allowed_video_list)
    directory = "/dataset/AIC2024/original_dataset/"
    mp4_files = []
    # Walk through the directory and its subdirectories
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            if file.endswith('.mp4') and file[0:3] in allowed_video_list:
                mp4_files.append(os.path.basename(os.path.join(dirpath, file)))
    #print(mp4_files)
    return mp4_files


def frame_to_index(video, frame):
    #open video index file
    with open(DATASET_INDEX + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    #print(video, frame)
    return index_dict[str(int(frame))]

def get_frame_dict(video):
    with open(DATASET_INDEX + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    return index_dict

def get_index_dict(video):
    with open(DATASET_INDEX + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    index_dict = {value: key for key, value in index_dict.items()}
    return index_dict

def merge_scores(list_res_A, list_res_B):
    # Iterate over list_res_B
    for record_B in list_res_B:
        max_temp_score = 0.0
        # Iterate over list_res_A to find matching records
        for record_A in list_res_A:
            # Check if video_name matches and keyframe_id difference is less than 1000
            if (record_A['video_name'] == record_B['video_name'] and 
                int(record_B['keyframe_id']) - int(record_A['keyframe_id']) > 0 and
                int(record_B['keyframe_id']) - int(record_A['keyframe_id']) < 1000):
                # Add the score from A to the current score sum
                max_temp_score = max(float(record_A['score']), max_temp_score)
        
        # Update the score in B
        record_B['score'] = float(record_B['score']) + max_temp_score
    
    #resort the score
    sorted_list = sorted(list_res_B, key=lambda x: x['score'], reverse=True)
        
    return sorted_list

class QDRANT:
    def __init__(self, collection_name = None):
        self.client = QdrantClient(url="http://0.0.0.0:6333", 
                                   port=None, 
                                   prefer_grpc=True, 
                                   timeout=600)
        print("QDRANT Connection Success")
        self.collection_name = collection_name

    
    
    def addDatabase(self, collection_name, feature_size, FEATURES_PATH, SPLIT_NAME):
        
        self.collection_name = collection_name
        self.size = feature_size
    
        
        if self.client.collection_exists(collection_name=collection_name) == True:
            print("Collection existed, deleting...")
            self.client.delete_collection(collection_name=self.collection_name)
        
            
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=feature_size, 
                                        distance=Distance.COSINE,
                                        quantization_config=models.BinaryQuantization(
                                            binary=models.BinaryQuantizationConfig(
                                            always_ram=True,
                                            ),
                                        )
                                        ),
            on_disk_payload=True,
            shard_number=96,
            optimizers_config=models.OptimizersConfigDiff(default_segment_number=64)
        )
        print("Collection Created!")
        
        print("Inserting Data...")
        
        struct_id = 0
        insert_points = []
        for idx_folder, folder_path in enumerate(FEATURES_PATH):
            # print(folder_path)
            for feat_npy in tqdm(sorted(os.listdir(folder_path))):
                # print(feat_npy)
                video_name = feat_npy.split('.')[0]
                feats_arr = np.load(os.path.join(folder_path, feat_npy))
                # frame_path = DATASET_PATH_TEAM + str(idx_folder) +"/frames/pyscenedetect/Keyframes_" + str(video_name).split("_")[0]  + "/keyframes/"  + str(video_name.split(".")[0])
                frame_path = DATASET_PATH_TEAM + str(idx_folder) +"/frames/" + SPLIT_NAME  + "/Keyframes_" + str(video_name).split("_")[0]  + "/keyframes/"  + str(video_name.split(".")[0])
                frame_list = sorted(os.listdir(frame_path))
                
                for idx, feat in enumerate(feats_arr):
                    feat_reshaped = feat.reshape(1,-1).astype('float32')[0]
                    insert_points.append(
                        PointStruct(
                            id=struct_id, 
                            vector=feat_reshaped, 
                            payload={
                                "idx_folder": idx_folder,
                                "video_name": video_name + ".mp4",
                                "frame_name": int(frame_list[idx].replace(".jpg","")),
                                "s2t": merge_and_remove_accents(dict_s2t[video_name + ".mp4"][frame_list[idx]]),
                                "duplicate_status": dict_duplicate[video_name][frame_list[idx].replace(".jpg","")],
                                "fps": dict_fps[video_name]
                        })
                    )
                    struct_id+=1

        operation_info = self.client.upsert(
            collection_name = self.collection_name,
            wait = False,
            points = insert_points)
        
        print("Dataset Insert Completed")
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="video_name",
            field_schema=models.KeywordIndexParams(
                type="keyword",
                on_disk=True,
            ),
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="frame_name",
            field_schema=models.IntegerIndexParams(
                type=models.IntegerIndexType.INTEGER,
                on_disk=True,
            ),
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="s2t",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=15,
                lowercase=True,
            ),
        )
        
        
        print("Create payload index complete")
        
        return operation_info
    
    def scroll_video(self, k, video_filter, time_in, time_out, s2t_filter=None):
        field_condition = [
                            models.FieldCondition(
                                key="video_name",
                                match=models.MatchValue(value=video_filter + '.mp4'),
                            ),
                            models.FieldCondition(
                                key="frame_name",
                                range=models.Range(
                                    gte = None if time_in == "" else int(time_in),
                                    lte = None if time_out == "" else int(time_out)
                                ),
                            ),
                        ]
        if s2t_filter!=None:
            field_condition.append(
                models.FieldCondition(
                key="s2t",
                match=models.MatchText(text=merge_and_remove_accents_no_list(s2t_filter)),
                )
            )
        FILTER_RESULTS = models.Filter(must = field_condition)
        
        SCROLL_RESULT = self.client.scroll(
            collection_name = self.collection_name,
            scroll_filter=FILTER_RESULTS,
            with_payload=True,
            with_vectors=False,
            limit=int(k)
        )
        return_result = []
        for item in SCROLL_RESULT[0]:
            for idx, field in enumerate(item):
                if idx == 0:
                    key = str(field[1])
                elif idx == 1:
                    idx_folder = str(field[1]['idx_folder'])
                    video_name = str(field[1]['video_name'])
                    keyframe_id = str(field[1]['frame_name']).zfill(5)
                    s2t = str(field[1]['s2t'])
                    duplicate_status = str(field[1]['duplicate_status']),
                    fps = str(field[1]['fps'])
                    
            result = {"key": key,
                      "idx_folder": idx_folder,
                      "video_name":video_name,
                      "keyframe_id": keyframe_id,
                      "s2t": s2t,
                      "duplicate_status": duplicate_status,
                      "fps": fps,
                      "score": 0.273
                      }
            
            return_result.append(result)
        
        return return_result
    
    def search(self, query, k, video_filter, time_in, time_out, batchs=[0,1,2], s2t_filter=None):
        
        list_video_allowed = []
        for batch in batchs:
            video_this_batch = get_videos_from_split(batch)
            for item in video_this_batch:
                list_video_allowed.append(item)
        
        field_condition = [
                models.FieldCondition(
                    key="video_name",
                    match=models.MatchAny(any=list_video_allowed)
                    )
                ]
        
        if s2t_filter!=None:
            field_condition.append(
                models.FieldCondition(
                    key="s2t",
                    match=models.MatchText(text=merge_and_remove_accents_no_list(s2t_filter)),
                )
            )
        
        BATCH_FILTER = models.Filter(
            must = field_condition
        )
            
        if video_filter == "":
            search_results = self.client.query_points(
                collection_name=self.collection_name, 
                query=query,
                query_filter=BATCH_FILTER,
                timeout=600,
                limit=int(k),
                #score_threshold=0.2
            ).points
        else:
            field_condition = [
                            models.FieldCondition(
                                key="video_name",
                                match=models.MatchValue(value=video_filter + '.mp4'),
                            ),
                            models.FieldCondition(
                                key="frame_name",
                                range=models.Range(
                                    gte = None if time_in == "" else int(time_in),
                                    lte = None if time_out == "" else int(time_out)
                                ),
                            ),
                        ]
            
            if s2t_filter!=None:
                field_condition.append(models.FieldCondition(
                    key="s2t",
                    match=models.MatchText(text=merge_and_remove_accents_no_list(s2t_filter)),
                    )
                )
            
            FILTER_RESULTS = models.Filter(
                        must = field_condition
                    )
            search_results = self.client.query_points(
                collection_name=self.collection_name, 
                query=query,
                query_filter=FILTER_RESULTS,
                timeout=600,
                limit=int(k),
                #score_threshold=0.2
            ).points
            
        return_result = []
        
        for item in search_results:
            
            for idx, field in enumerate(item):
                if idx == 0:
                    key = str(field[1])
                elif idx == 2:
                    score = str(field[1])
                elif idx == 3:
                    idx_folder = str(field[1]['idx_folder'])
                    video_name = str(field[1]['video_name'])
                    keyframe_id = str(field[1]['frame_name']).zfill(5)
                    s2t = str(field[1]['s2t'])
                    duplicate_status = str(field[1]['duplicate_status']),
                    fps = str(field[1]['fps'])
                    
            result = {"key": key,
                      "idx_folder": idx_folder,
                      "video_name":video_name,
                      "keyframe_id": keyframe_id,
                      "s2t": s2t,
                      "duplicate_status": duplicate_status,
                      "fps": fps,
                      "score": score
                      }
            return_result.append(result)
        
        return return_result

    
    def deleteDatabase(self):
        self.client.delete_collection(collection_name=self.collection_name)
        

    def getCount(self):
        for idx, item in enumerate(self.client.get_collection(collection_name=self.collection_name)):
            #print(item)
            if idx==4:
                return int(item[1])
        return 0
    

    def find_duplicate(self, query, threshold):
        
        SEARCH_RESULTS = self.client.query_points(
            collection_name=self.collection_name, 
            query=query,
            timeout=600,
            limit=200,
            score_threshold=threshold
        ).points
        
        return_result = []
        
        for item in SEARCH_RESULTS:
            
            
            for idx, field in enumerate(item):
                if idx == 0:
                    key = str(field[1])
                elif idx == 2:
                    score = str(field[1])
                elif idx == 3:
                    idx_folder = str(field[1]['idx_folder'])
                    video_name = str(field[1]['video_name'])
                    keyframe_id = str(field[1]['frame_name']).zfill(5)
                    s2t = str(field[1]['s2t'])
                    duplicate_status = str(field[1]['duplicate_status']),
                    fps = str(field[1]['fps'])
                    
            result = {"key": key,
                      "idx_folder": idx_folder,
                      "video_name":video_name,
                      "keyframe_id": keyframe_id,
                      "s2t": s2t,
                      "duplicate_status": duplicate_status,
                      "fps": fps,
                      "score": score
                      }

            same_shot = False
            for previous_item in return_result:
                if previous_item['video_name']==video_name and abs(int(previous_item['keyframe_id'])-int(keyframe_id))<=1000:
                    same_shot = True
                    break
            if not same_shot:
                return_result.append(result)
            
        SEARCH_RESULTS = return_result
        
        return SEARCH_RESULTS       
    
    def search_temporal(self, queryList, k, video_filter, time_in, time_out, re_sort = None, batchs=[0,1,2], s2t_filter=None):
        
        list_video_allowed = []
        for batch in batchs:
            video_this_batch = get_videos_from_split(batch)
            for item in video_this_batch:
                list_video_allowed.append(item)
        
        field_condition = [
                models.FieldCondition(
                    key="video_name",
                    match=models.MatchAny(any=list_video_allowed)
                )
            ]
        
        if s2t_filter!=None:
            field_condition.append(models.FieldCondition(
                key="s2t",
                match=models.MatchText(text=merge_and_remove_accents_no_list(s2t_filter)),
                )
            )
        
        BATCH_FILTER = models.Filter(
            must = field_condition
        )
        
        if video_filter == "":
            SEARCH_RESULTS = self.client.query_points(
                collection_name=self.collection_name, 
                query=queryList[0],
                query_filter=BATCH_FILTER, 
                timeout=600,
                limit=int(k)*len(queryList),
                #score_threshold=0.2
            ).points
        else:
            field_condition = [
                            models.FieldCondition(
                                key="video_name",
                                match=models.MatchValue(value=video_filter + '.mp4'),
                            ),
                            models.FieldCondition(
                                key="frame_name",
                                range=models.Range(
                                    gte = None if time_in == "" else int(time_in),
                                    lte = None if time_out == "" else int(time_out)
                                ),
                            ),
                        ]
            
            if s2t_filter!=None:
                field_condition.append(models.FieldCondition(
                    key="s2t",
                    match=models.MatchText(text=merge_and_remove_accents_no_list(s2t_filter)),
                    )
                )
            
            FILTER_RESULTS = models.Filter(
                        must = field_condition
                    )
            SEARCH_RESULTS = self.client.query_points(
                collection_name=self.collection_name, 
                query=queryList[0],
                query_filter=FILTER_RESULTS,
                timeout=600,
                limit=int(k)*len(queryList),
                #score_threshold=0.2
            ).points

        return_result = []
        
        for item in SEARCH_RESULTS:
            
            for idx, field in enumerate(item):
                if idx == 0:
                    key = str(field[1])
                elif idx == 2:
                    score = str(field[1])
                elif idx == 3:
                    idx_folder = str(field[1]['idx_folder'])
                    video_name = str(field[1]['video_name'])
                    keyframe_id = str(field[1]['frame_name']).zfill(5)
                    s2t = str(field[1]['s2t'])
                    duplicate_status = str(field[1]['duplicate_status']),
                    fps = str(field[1]['fps'])
                    
            result = {"key": key,
                      "idx_folder": idx_folder,
                      "video_name":video_name,
                      "keyframe_id": keyframe_id,
                      "s2t": s2t,
                      "duplicate_status": duplicate_status,
                      "fps": fps,
                      "score": score
                      }
            return_result.append(result)
            
        SEARCH_RESULTS = return_result
        PREVIOUS_SEARCH_RESULTS = SEARCH_RESULTS
        for idx, query in enumerate(queryList):
            if idx==0:
                continue
            
            return_result = []
            FILTER_RESULTS = []
            
            for result in SEARCH_RESULTS:
                split_name = result['idx_folder']
                video_name = result['video_name'].replace('.mp4','')
                frame = result['keyframe_id']
                FILTER_RESULTS.append(
                    models.Filter(
                        must = [
                            models.FieldCondition(
                                key="video_name",
                                match=models.MatchValue(value=video_name + '.mp4'),
                            ),
                            models.FieldCondition(
                                key="frame_name",
                                range=models.Range(
                                    gte = int(frame),
                                    lte = int(frame)+1000
                                ),
                            ),
                        ]
                    )
                )
        
            FILTER_RESULTS = models.Filter(
                should = FILTER_RESULTS
            )
            
            SEARCH_RESULTS = self.client.query_points(
                collection_name=self.collection_name, 
                query=query,
                query_filter=FILTER_RESULTS,
                limit=int(k)*(len(queryList)-idx),
                timeout=600,
                #score_threshold=0.2
            ).points
            
            return_result = []
            
            for item in SEARCH_RESULTS:
                for idx, field in enumerate(item):
                    if idx == 0:
                        key = str(field[1])
                    elif idx == 2:
                        score = str(field[1])
                    elif idx == 3:
                        idx_folder = str(field[1]['idx_folder'])
                        video_name = str(field[1]['video_name'])
                        keyframe_id = str(field[1]['frame_name']).zfill(5)
                        s2t = str(field[1]['s2t'])
                        duplicate_status = str(field[1]['duplicate_status']),
                        fps = str(field[1]['fps'])
                        
                result = {"key": key,
                        "idx_folder": idx_folder,
                        "video_name":video_name,
                        "keyframe_id": keyframe_id,
                        "s2t": s2t,
                        "duplicate_status": duplicate_status,
                        "fps": fps,
                        "score": score
                        } 
                return_result.append(result)
            
            SEARCH_RESULTS = return_result
            SEARCH_RESULTS = merge_scores(PREVIOUS_SEARCH_RESULTS, SEARCH_RESULTS)
            PREVIOUS_SEARCH_RESULTS = SEARCH_RESULTS
            
        if re_sort!=None and re_sort!=(len(queryList)-1):
            re_sort = int(re_sort)
            
            return_result = []
            FILTER_RESULTS = []
            
            for result in SEARCH_RESULTS:
                split_name = result['idx_folder']
                video_name = result['video_name'].replace('.mp4','')
                frame = result['keyframe_id']
                FILTER_RESULTS.append(
                    models.Filter(
                        must = [
                            models.FieldCondition(
                                key="video_name",
                                match=models.MatchValue(value=video_name + '.mp4'),
                            ),
                            models.FieldCondition(
                                key="frame_name",
                                range=models.Range(
                                    gte = int(frame),
                                    lte = int(frame)
                                ),
                            ),
                        ]
                    )
                )
        
            FILTER_RESULTS = models.Filter(
                should = FILTER_RESULTS
            )
            
            SEARCH_RESULTS = self.client.query_points(
                collection_name=self.collection_name, 
                query=queryList[re_sort],
                query_filter=FILTER_RESULTS,
                limit=int(k),
                timeout=600,
                #score_threshold=0.2
            ).points
            
            return_result = []
            
            for item in SEARCH_RESULTS:
                for idx, field in enumerate(item):
                    if idx == 0:
                        key = str(field[1])
                    elif idx == 2:
                        score = str(field[1])
                    elif idx == 3:
                        idx_folder = str(field[1]['idx_folder'])
                        video_name = str(field[1]['video_name'])
                        keyframe_id = str(field[1]['frame_name']).zfill(5)
                        s2t = str(field[1]['s2t'])
                        uplicate_status = str(field[1]['duplicate_status']),
                        fps = str(field[1]['fps'])
                        
                result = {"key": key,
                        "idx_folder": idx_folder,
                        "video_name":video_name,
                        "keyframe_id": keyframe_id,
                        "s2t": s2t,
                        "duplicate_status": duplicate_status,
                        "fps": fps,
                        "score": score
                        } 
                return_result.append(result)
            
            SEARCH_RESULTS = return_result
        


        return SEARCH_RESULTS

# import time

# # feat_test = np.load("cat.npy").reshape(1,-1).astype('float32')[0]
# # st = time.time()
# # search_result = qdrant_database.search(feat_test,180)
# # print(search_result)
# # print(time.time()-st)

# # feat_test = np.load("nurse.npy").reshape(1,-1).astype('float32')[0]
# # st = time.time()
# # search_result = qdrant_database.search(feat_test,180)
# # print(search_result)
# # print(time.time()-st)


# # feat_test = np.load("dog.npy").reshape(1,-1).astype('float32')[0]
# # st = time.time()
# # search_result = qdrant_database.search(feat_test,180)
# # print(search_result)
# # print(time.time()-st)

# feat_test = np.load("cat.npy").reshape(1,-1).astype('float32')[0]
# st = time.time()
# search_result = qdrant_database.search(feat_test,3)
# for item in search_result:
#     print(item)
# qdrant_database.deleteDatabase()