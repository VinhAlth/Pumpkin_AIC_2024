import torch
from PIL import Image
import open_clip
import requests
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/nhihtc/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

class LAION:
    def __init__(self):
        print("laion/CLIP-ViT-g-14-laion2B-s34B-b88K")
        self.device = "cuda"
        self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K', device=self.device)
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_path)
        inputs = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features = self.model.encode_image(inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

    def get_text_features(self, text: str) -> np.array:       
        inputs = self.tokenizer(text).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = self.model.encode_text(inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy()

