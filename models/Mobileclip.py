import torch
from PIL import Image

import requests
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/nhihtc/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/ml-mobileclip/mobileclip")
import mobileclip
class MOBILECLIP:
    def __init__(self):
        print("mobileclip_blt")
        self.device = "cuda"
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='/workspace/competitions/AIC_2024/SIU_Pumpkin/libs/ml-mobileclip/mobileclip/checkpoint/mobileclip_blt.pt')
        self.model.to(self.device)
        self.tokenizer = mobileclip.get_tokenizer('mobileclip_b')

    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_path)
        inputs = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

    def get_text_features(self, text: str) -> np.array:       
        inputs = self.tokenizer(text).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy()

