import torch
from PIL import Image
import requests
import numpy as np
import os
os.environ['HF_HOME'] = '/workspace/competitions/AIC_2024/SIU_Pumpkin/cache/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from scipy.spatial.distance import cosine

import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin")

from models.Siglip import SIGLIP
model = SIGLIP()

def preprocessing_text(text):
    global model
    text_feat_arr = model.get_text_features(text) 
    text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32') #=> float32
    return text_feat_arr[0]

# query = "nurse"
# np.save(query + ".npy", preprocessing_text(query))

# query = "dog"
# np.save(query + ".npy", preprocessing_text(query))

query = "2022"
feat_1 = preprocessing_text(query)
# np.save("/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/vector_database/QDRANT/" + query + "_siglip.npy", preprocessing_text(query))
query = "two thousand twenty two"
feat_2 = preprocessing_text(query)
cosine_similarity = 1 - cosine(feat_1, feat_2)
print(cosine_similarity)

