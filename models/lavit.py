import os
import random
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/LaVIT/LaVIT/")
from models import build_model
from PIL import Image
import ml_dtypes

def demo():
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)

    # The local directory you save the LaVIT pre-trained weight, 
    # it will automatically download the checkpoint from huggingface
    model_path = '/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/LaVIT_weight/'

    # Using BFloat16 during inference
    model_dtype = 'bf16'  # Or set to fp16 to enable float16 inference

    # Inference using GPU-ID
    device_id = 6
    torch.cuda.set_device(device_id)
    device = torch.device('cuda')

    # Building LaVIT for understanding and load its weight from huggingface
    model = build_model(model_path=model_path, model_dtype=model_dtype,
                device_id=device_id, use_xformers=False, understanding=True)
    model = model.to(device)    

    
    
    # Image Captioning
    image_path = '/dataset/AIC2024/pumkin_dataset/1/frames/pyscenedetect/Keyframes_L11/keyframes/L11_V001/00030.jpg'
    caption = model.generate({"image": image_path})[0]
    print(caption)
    # an old photo of a horse and buggy in front of a building

    # Visual Question Answering
    image_path = '/dataset/AIC2024/pumkin_dataset/1/frames/pyscenedetect/Keyframes_L11/keyframes/L11_V001/00030.jpg'
    question = "How many minute?"
    answer = model.predict_answers({"image": image_path, "text_input": question}, max_len=10)[0]
    print("The answer is: ", answer)
    # The answer is: orange juice


class LAVIT:
    def __init__(self, gpu_id=2) -> None:
        seed = 1234
        random.seed(seed)
        torch.manual_seed(seed)

        # The local directory you save the LaVIT pre-trained weight, 
        # it will automatically download the checkpoint from huggingface
        model_path = '/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/LaVIT_weight/'

        # Using BFloat16 during inference
        model_dtype = 'bf16'  # Or set to fp16 to enable float16 inference

        # Inference using GPU-ID
        device_id = gpu_id
        torch.cuda.set_device(device_id)
        self.device = torch.device('cuda')

        # Building LaVIT for understanding and load its weight from huggingface
        self.model = build_model(model_path=model_path, model_dtype=model_dtype,
                    device_id=device_id, use_xformers=False, understanding=True)
        self.model = self.model.to(self.device)    
    
    def get_image_features(self, image_path : str):
        image_feat = self.model.process_image(image_path)
        with self.model.maybe_autocast():
            image_feat = self.model.compute_dynamic_visual_embeds(image_feat)
            
        return image_feat[0].cpu().detach().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
    
    def get_text_features(self, text):
        self.model.llama_tokenizer.padding_side = "left"
        prompt_tokens = self.model.llama_tokenizer(
            text, padding="longest", return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        with self.model.maybe_autocast():
            text_feat = self.model.llama_model.get_input_embeddings()(prompt_tokens.input_ids)
        
        #text_feat = text_feat.mean(dim=1)
        return text_feat.cpu().detach().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
    
lavit = LAVIT(6)
print(lavit.get_image_features('/dataset/AIC2024/pumkin_dataset/1/frames/pyscenedetect/Keyframes_L11/keyframes/L11_V001/00030.jpg'))
print(lavit.get_text_features("a cat"))

import time


st = time.time()
image_feature_1 = lavit.get_image_features('/dataset/AIC2024/pumkin_dataset/1/frames/pyscenedetect/Keyframes_L11/keyframes/L11_V001/00030.jpg')
print("1st image FE: ", time.time()-st)
print(image_feature_1.shape)

st = time.time()
image_feature_2 = lavit.get_image_features('/dataset/AIC2024/pumkin_dataset/1/frames/pyscenedetect/Keyframes_L11/keyframes/L11_V001/00540.jpg')
print("2nd image FE: ", time.time()-st)
print(image_feature_2.shape)

query = ["a cat jumping","60s","firework"]

from numpy import dot
from numpy.linalg import norm

def cosine(a, b):
    # a = a.flatten()
    # b = b.flatten()
    # cos_sim = dot(a, b)/(norm(a)*norm(b))
    a_normalized = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_normalized = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos_sim = np.matmul(a_normalized, b_normalized.T)
    total_cos_sim= np.mean(cos_sim)
    return total_cos_sim
print("\nCOSINE WITH IMAGE 1")
for _, text in enumerate(query):
    st = time.time()
    text_feature = lavit.get_text_features(text).squeeze()
    print(f"text FE {_+1}: ", time.time()-st)
    print(text, text_feature.shape)
    similarity = cosine(image_feature_1, text_feature)
    print(similarity)

print("\nCOSINE WITH IMAGE 2")
for _, text in enumerate(query):
    st = time.time()
    text_feature = lavit.get_text_features(text).squeeze(0)
    print(f"text FE {_+1}: ", time.time()-st)
    print(text, text_feature.shape)
    similarity = cosine(image_feature_2, text_feature)
    print(similarity)