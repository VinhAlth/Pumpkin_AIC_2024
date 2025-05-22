import torch
from PIL import Image
import requests
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/nhihtc/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152


import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin")
import mobileclip

from models.Nomicai import NOMICAI
model = NOMICAI()

def preprocessing_text(text):
    global model
    text_feat_arr = model.get_text_features(text) 
    text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32') #=> float32
    return text_feat_arr

# query = "nurse"
# np.save(query + ".npy", preprocessing_text(query))

# query = "dog"
# np.save(query + ".npy", preprocessing_text(query))

query = "cat"
np.save(query + ".npy", preprocessing_text(query))