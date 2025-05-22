import torch
from PIL import Image
import requests
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/nhihtc/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152


import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin")

from models.Longclip import LONGCLIP
model = LONGCLIP()

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
np.save("/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/vector_database/QDRANT/" + query + "_longclip.npy", preprocessing_text(query))