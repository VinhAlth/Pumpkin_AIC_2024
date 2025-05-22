import os

os.environ['HF_HOME'] = '/workspace/competitions/AIC_2024/SIU_Pumpkin/cache/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import numpy as np

import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin")

from models.Eva import EVA
model = EVA()

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
np.save("/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/vector_database/QDRANT/" + query + "_eva.npy", preprocessing_text(query))