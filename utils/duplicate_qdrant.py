import sys
import os
import ujson
import numpy as np
import torch
import pathlib
from pathlib import Path
from PIL import Image


os.environ['HF_HOME'] = '/workspace/competitions/AIC_2024/SIU_Pumpkin/cache/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="5"
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

from libs.vector_database.QDRANT.qdrant import QDRANT
from models.Siglip import SIGLIP



model = SIGLIP()
qdrant = QDRANT("SIGLIP")

def preprocessing_image(image):
    global model
    image_feat_arr = model.get_image_features(image) 
    image_feat_arr = image_feat_arr.reshape(1,-1).astype('float32') #=> float32
    return image_feat_arr[0]

query = preprocessing_image("/dataset/AIC2024/pumkin_dataset/0/frames/autoshot/Keyframes_L08/keyframes/L08_V006/02593.jpg") 


# # process_path = "/dataset/AIC2024/pumkin_dataset/0/frames/autoshot/"

# # for image in Path(process_path).glob("**/*.jpg"):
# #     if not image.is_file():  # Skip directories
# #         continue
# #     image = str(image)
# #     json_folder = Path(image.replace("/frames/autoshot/","/duplicate_json/")).parents[0]
# #     json_folder.mkdir(parents=True, exist_ok=True)    
# #     json_name = str(image).replace("/frames/autoshot/","/duplicate_json/").replace(".jpg",".json")
# #     query = preprocessing_image(image)
# #     with open(json_name, 'w') as save_json:
# #         ujson.dump(qdrant.find_duplicate(query, 0.96), save_json, indent=4)
# #     #picture.save(str(low_res_image).replace(".jpg",".avif"), "AVIF", optimize=True, quality=10)
# #     print(json_name)
    
    
process_path = "/dataset/AIC2024/pumkin_dataset/1/frames/autoshot/"
switched_processed = False
for image in Path(process_path).glob("**/*.jpg"):
    if not image.is_file():  # Skip directories
        continue
    if "L19_V027/19158" in str(image):
        switched_processed = True
    if switched_processed==False:
        continue
    image = str(image)
    json_folder = Path(image.replace("/frames/autoshot/","/duplicate_json/")).parents[0]
    json_folder.mkdir(parents=True, exist_ok=True)    
    json_name = str(image).replace("/frames/autoshot/","/duplicate_json/").replace(".jpg",".json")
    query = preprocessing_image(image)
    if os.path.exists(json_name):
        os.remove(json_name)
    with open(json_name, 'w') as save_json:
        ujson.dump(qdrant.find_duplicate(query, 0.96), save_json, indent=4)
    #picture.save(str(low_res_image).replace(".jpg",".avif"), "AVIF", optimize=True, quality=10)
    print(json_name)

# process_path = "/dataset/AIC2024/pumkin_dataset/2/frames/autoshot/"

# for image in Path(process_path).glob("**/*.jpg"):
#     if not image.is_file():  # Skip directories
#         continue
#     image = str(image)
#     json_folder = Path(image.replace("/frames/autoshot/","/duplicate_json/")).parents[0]
#     json_folder.mkdir(parents=True, exist_ok=True)    
#     json_name = str(image).replace("/frames/autoshot/","/duplicate_json/").replace(".jpg",".json")
#     query = preprocessing_image(image)
#     with open(json_name, 'w') as save_json:
#         ujson.dump(qdrant.find_duplicate(query, 0.96), save_json, indent=4)
#     #picture.save(str(low_res_image).replace(".jpg",".avif"), "AVIF", optimize=True, quality=10)
#     print(json_name)

# process_path = "/dataset/AIC2024/pumkin_dataset/0/frames/autoshot/"

# dict_duplicate = {}

# for image in Path(process_path).glob("**/*.jpg"):
#     if not image.is_file():  # Skip directories
#         continue
#     image = str(image)
#     json_folder = Path(image.replace("/frames/autoshot/","/duplicate_json/")).parents[0]
#     json_folder.mkdir(parents=True, exist_ok=True)    
#     json_name = str(image).replace("/frames/autoshot/","/duplicate_json/").replace(".jpg",".json")
    
#     #print(json_name)
    
#     with open(json_name, 'r') as open_json:
#         list_dup = ujson.load(open_json)
    
    
#     parts = json_name.split('/')

#     # Extract L01_V001 and 00024 from their respective positions
#     video_name = parts[-2]  # Second last part
#     frame_name = parts[-1].split('.')[0]
        
#     if len(list_dup)>1:
#         if video_name not in dict_duplicate:
#             dict_duplicate[video_name] = {}
#         dict_duplicate[video_name][frame_name] = 1
#     else:
#         if video_name not in dict_duplicate:
#             dict_duplicate[video_name] = {}        
#         dict_duplicate[video_name][frame_name] = 0
        
#     # if os.path.exists(json_name):
#     #     continue
#     # query = preprocessing_image(image)
#     # with open(json_name, 'w') as save_json:
#     #     ujson.dump(qdrant.find_duplicate(query, 0.96), save_json, indent=4)
#     # #picture.save(str(low_res_image).replace(".jpg",".avif"), "AVIF", optimize=True, quality=10)
#     print(json_name)

# with open('/dataset/AIC2024/pumkin_dataset/0/utils/duplicate_0_all.json','w') as save_duplicate:
#     ujson.dump(dict_duplicate, save_duplicate, indent=4)


# process_path = "/dataset/AIC2024/pumkin_dataset/1/frames/autoshot/"

# dict_duplicate = {}

# for image in Path(process_path).glob("**/*.jpg"):
#     if not image.is_file():  # Skip directories
#         continue
#     image = str(image)
#     json_folder = Path(image.replace("/frames/autoshot/","/duplicate_json/")).parents[0]
#     json_folder.mkdir(parents=True, exist_ok=True)    
#     json_name = str(image).replace("/frames/autoshot/","/duplicate_json/").replace(".jpg",".json")

#     with open(json_name, 'r') as open_json:
#         list_dup = ujson.load(open_json)
    
    
#     parts = json_name.split('/')

#     # Extract L01_V001 and 00024 from their respective positions
#     video_name = parts[-2]  # Second last part
#     frame_name = parts[-1].split('.')[0]
        
#     if len(list_dup)>1:
#         if video_name not in dict_duplicate:
#             dict_duplicate[video_name] = {}
#         dict_duplicate[video_name][frame_name] = 1
#     else:
#         if video_name not in dict_duplicate:
#             dict_duplicate[video_name] = {}        
#         dict_duplicate[video_name][frame_name] = 0
#     print(json_name)

# with open('/dataset/AIC2024/pumkin_dataset/1/utils/duplicate_1_all.json','w') as save_duplicate:
#     ujson.dump(dict_duplicate, save_duplicate, indent=4)