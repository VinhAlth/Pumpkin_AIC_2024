import faiss
from tqdm import tqdm
import numpy as np
import os
import json
import ujson

DATASET_PATH_TEAM = "/dataset/AIC2024/pumkin_dataset/"
DATASET_INDEX = "/dataset/AIC2024/pumkin_dataset/utils/index"
# valid_program = '/dataset/AIC2024/pumkin_dataset/1/util_files/valid_frame_test_h12_1.json'
# # soccer = '/dataset/AIC2023/pumkin_dataset/soccer_compile.json'

# with open(valid_program, encoding='utf-8-sig') as json_file:
#     valid_program = json.load(json_file)

# with open(soccer, encoding='utf-8-sig') as json_file:
#     soccer = json.load(json_file)


def frame_to_index(index_by_video_path, video, frame):
    #open video index file
    with open(index_by_video_path + "/" + video + ".json", 'r') as infile:
        index_dict = ujson.load(infile)
    return index_dict[str(frame)]


# Function load extracted feature into faiss    
def indexing_methods_faiss_temp(clip_features_path, size, temporal_list):
    # faiss_db = faiss.IndexFlatL2(size)
    faiss_db = faiss.IndexFlatIP(size)
    db = []
    for idx_folder, folder_path in enumerate(clip_features_path):
        # print(folder_path)
        for feat_npy in tqdm(sorted(os.listdir(folder_path))):
            # print(feat_npy)
            video_name = feat_npy.split('.')[0]
            # print(video_name)
            if video_name in temporal_list:
                feats_arr = np.load(os.path.join(folder_path, feat_npy))
                for idx, feat in enumerate(feats_arr):
                #Lưu mỗi records với 3 trường thông tin là video_name, keyframe_id, feature_of_keyframes
                    if temporal_list[video_name][0]<= idx <= temporal_list[video_name][1]: 
                        #instance = (video_name, idx, idx_folder)
                        instance = (video_name, idx, idx_folder)
                        db.append(instance)
                        faiss_db.add(feat.reshape(1,-1).astype('float32'))
    db = dict(enumerate(sorted(db)))
    print(f'total: {len(db)}')
    return db, faiss_db


def indexing_rerank(MODEL_B_FEATURES_PATH, SEARCH_RESULTS_A, size):
    faiss_db = faiss.IndexFlatIP(size)

    db = []
    
    for result in SEARCH_RESULTS_A:

        split_name = result['idx_folder']
        video_name = result['video_name'].replace('.mp4','')
        frame = int(frame_to_index(DATASET_INDEX, video_name, str(int(result['keyframe_id']))))
        instance = (video_name, frame, split_name)
        db.append(instance)
        feat = np.load(MODEL_B_FEATURES_PATH[int(split_name)] + '/' + video_name + '.npy')[frame]
        faiss_db.add(feat.reshape(1,-1).astype('float32'))
    
    db = dict(enumerate(sorted(db)))
    print(f'total: {len(db)}')
    return db, faiss_db

def indexing_methods_faiss(clip_features_path, size):
    # faiss_db = faiss.IndexFlatL2(size)
    faiss_db = faiss.IndexFlatIP(size)

    db = []
    for idx_folder, folder_path in enumerate(clip_features_path):
        #print("IDEX:", folder_path)
        for feat_npy in tqdm(sorted(os.listdir(folder_path))):
            # print(feat_npy)
            video_name = feat_npy.split('.')[0]
            feats_arr = np.load(os.path.join(folder_path, feat_npy))
            for idx, feat in enumerate(feats_arr):
            #Lưu mỗi records với 3 trường thông tin là video_name, keyframe_id, feature_of_keyframes
                #instance = (video_name, idx, idx_folder)
                instance = (video_name, idx, idx_folder)
                db.append(instance)
                # print(feat.shape)
                faiss_db.add(feat.reshape(1,-1).astype('float32'))
    db = dict(enumerate(sorted(db)))
    print(f'total: {len(db)}')
    return db, faiss_db

# def indexing_methods_faiss_soccer(clip_features_path, size):
#     # faiss_db = faiss.IndexFlatL2(size)
#     global soccer

#     faiss_db = faiss.IndexFlatIP(size)

    
#     db = []
#     for idx_folder, folder_path in enumerate(clip_features_path):
#         # print(folder_path)
#         for feat_npy in tqdm(sorted(os.listdir(folder_path))):
#             # print(feat_npy)
#             video_name = feat_npy.split('.')[0]
#             feats_arr = np.load(os.path.join(folder_path, feat_npy))
#             frame_path = DATASET_PATH_TEAM + str(idx_folder) +"/frames/low_res/Keyframes_" + str(video_name).split("_")[0]  + "/keyframes/"  + str(video_name.split(".")[0])
#             frame_list = sorted(os.listdir(frame_path))
#             for idx, feat in enumerate(feats_arr):
#             #Lưu mỗi records với 3 trường thông tin là video_name, keyframe_id, feature_of_keyframes
#                 if([str(idx_folder),video_name,frame_list[idx].replace('.jpg','')] in soccer): 
#                     instance = (video_name, idx, idx_folder)
#                     db.append(instance)
#                     # print(feat.shape)
#                     faiss_db.add(feat.reshape(1,-1).astype('float32'))
#     db = dict(enumerate(sorted(db)))
#     print(f'total: {len(db)}')
#     return db, faiss_db

def indexing_methods_faiss_filter(clip_features_path, size, filter_file, SPLIT_NAME):
    valid_program = None
    
    with open(filter_file, encoding='utf-8-sig') as json_file:
        valid_program = json.load(json_file)
        
    # faiss_db = faiss.IndexFlatL2(size)
    faiss_db = faiss.IndexFlatIP(size)

    
    db = []
    for idx_folder, folder_path in enumerate(clip_features_path):
        # print(folder_path)
        for feat_npy in tqdm(sorted(os.listdir(folder_path))):
            # print(feat_npy)
            video_name = feat_npy.split('.')[0]
            feats_arr = np.load(os.path.join(folder_path, feat_npy))
            # frame_path = DATASET_PATH_TEAM + str(idx_folder) +"/frames/pyscenedetect/Keyframes_" + str(video_name).split("_")[0]  + "/keyframes/"  + str(video_name.split(".")[0])
            frame_path = DATASET_PATH_TEAM + str(idx_folder) +"/frames/" + SPLIT_NAME  + "/Keyframes_" + str(video_name).split("_")[0]  + "/keyframes/"  + str(video_name.split(".")[0])
            frame_list = sorted(os.listdir(frame_path))
            for idx, feat in enumerate(feats_arr):
            #Lưu mỗi records với 3 trường thông tin là video_name, keyframe_id, feature_of_keyframes
                #if valid_program[str((video_name,frame_list[idx].replace('.jpg','')))] == True: 
                    # instance = (video_name, idx, idx_folder)
                instance = (video_name, idx, idx_folder)
                db.append(instance)
                # print(feat.shape)
                faiss_db.add(feat.reshape(1,-1).astype('float32'))
    db = dict(enumerate(sorted(db)))
    print(f'total: {len(db)}')
    return db, faiss_db

def search_in_B_filtered_by_A():
    # Sample vectors
    vectors_A = np.random.rand(5, 10).astype('float32')  # 5 vectors in A
    vectors_B = np.random.rand(5, 10).astype('float32')  # 5 vectors in B
    vector_query = np.random.rand(1, 10).astype('float32')  # The query vector

    # 1) Setup FAISS A 
    faiss_db_A = faiss.IndexFlatIP(10)
    faiss_db_A.add(vectors_A)

    # 2) Setup FAISS B 
    faiss_db_B = faiss.IndexFlatIP(10)
    faiss_db_B.add(vectors_B)

    # 3) Search in A to get top-2 results
    D_A, I_A = faiss_db_A.search(vector_query, k=2)

    # 4) Search in B to get all results
    D_B, I_B = faiss_db_B.search(vector_query, k=len(vectors_B))
    
    # 5) Filter I_B by I_A (keep order of I_B)
    return 
    
    
   
