import os
os.environ['HF_HOME'] = '/workspace/competitions/AIC_2024/SIU_Pumpkin/cache/pretrained_weights'
import torch
from PIL import Image

import requests
import numpy as np

from transformers import AutoModel, AutoConfig
from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer

import sys
sys.path.append("/workspace/nhihtc/work/AIC")

class EVA:
    def __init__(self):
        print("AAI/EVA-CLIP-8B")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = AutoModel.from_pretrained("BAAI/EVA-CLIP-8B", torch_dtype=torch.float16, trust_remote_code=True).to('cuda').eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("BAAI/EVA-CLIP-8B")

    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True, truncation = True).pixel_values.to('cuda')
        with torch.no_grad(),  torch.amp.autocast('cuda'):
            image_features = self.model.encode_image(inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

    def get_text_features(self, text: str) -> np.array:       
        inputs = self.tokenizer(text,  return_tensors="pt", padding=True, truncation = True).input_ids.to('cuda')
        with torch.no_grad(),  torch.amp.autocast('cuda'):
            text_features = self.model.encode_text(inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy()

