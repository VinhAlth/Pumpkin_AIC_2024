import os 
import numpy as np
import urllib.parse
import json
import sys
import ujson
import bisect
import pillow_avif
import requests
from tqdm import tqdm
import ujson

sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

from models.BLIP2 import BLIP2Coco
model = BLIP2Coco()

def preprocessing_text(text):
    global model
    text_feat_arr = model.get_text_features(text) 
    text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32') #=> float32
    return text_feat_arr

query = "nurse"
np.save(query + ".npy", preprocessing_text(query))

query = "dog"
np.save(query + ".npy", preprocessing_text(query))

query = "cat"
np.save(query + ".npy", preprocessing_text(query))