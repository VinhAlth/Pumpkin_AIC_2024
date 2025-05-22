import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/models")
from BLIP2 import BLIP2Coco

# # from models.BLIPFeaturesExtractor import Blip
# from models.BLIP2FeaturesExtractor import BLIP2
import os
import numpy as np
from tqdm import tqdm
import argparse
KEY_FRAME_PATH = '/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5/' 

FEATURE_PATH = '/dataset/AIC2024/pumkin_dataset/0/blip2/'  



def main():
    model =  BLIP2Coco()  
    print("Starting")
    for i in range(1, 13):
        if i>6:
            continue
        video_list_path = f"{KEY_FRAME_PATH}Keyframes_L{i:02d}/keyframes/"
        # if(_ == 'Keyframes_L35' or _ == 'Keyframes_L36'): 
        #     print(_)
        for video_name in os.listdir(video_list_path): 
            frame_list = video_list_path + video_name + '/'
            video_feature = []
            print(video_name)
            for frame_name in tqdm(sorted(os.listdir(frame_list))):
                if ".csv" in frame_name: 
                    continue
                frame_path = frame_list + "/" + frame_name
                frame_feature = model.get_image_features(frame_path) #############
                video_feature.append(frame_feature)
            video_feature = np.array(video_feature)
            np.save(FEATURE_PATH + video_name + ".npy", video_feature)

if __name__ == '__main__':
    main()