import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/models")
from LaionB88K import LAION

# # from models.BLIPFeaturesExtractor import Blip
# from models.BLIP2FeaturesExtractor import BLIP2
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/nhihtc/pretrained_weights'
import numpy as np
from tqdm import tqdm
KEY_FRAME_PATH = '/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5/' 

FEATURE_PATH = '/dataset/AIC2024/pumkin_dataset/0/laion/'  
# taskset -c 51-75 python keyframes_laion32.py

def main():
    model =  LAION()  
    print("Starting")
    for i in range(1, 13):
        if i>6:
            continue
        video_list_path = f"{KEY_FRAME_PATH}Keyframes_L{i:02d}/keyframes/"
        print(video_list_path)
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
