import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/nhihtc/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

class SIGLIP:
    def __init__(self):
        print("google/siglip-so400m-patch14-384")
        self.device = "cuda"
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").eval().to(self.device)
        #self.model = torch.compile(self.model)

    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_path)
            
        inputs = self.processor(images=image, padding="max_length", return_tensors="pt", truncation = True).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

    def get_text_features(self, text: str) -> np.array:     
        inputs = self.processor(text=text, padding="max_length", return_tensors="pt", truncation = True).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = self.model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy()

