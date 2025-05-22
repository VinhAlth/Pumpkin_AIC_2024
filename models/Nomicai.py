import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/nhihtc/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

class NOMICAI:
    def __init__(self):
        print("nomicai")
        self.device = "cuda"
        self.processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        self.vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
        self.vision_model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
        self.text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        self.text_model.eval().to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
  
    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_path)
            
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        img_emb = self.vision_model(**inputs).last_hidden_state
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
        image_features = img_embeddings
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

    def get_text_features(self, text: str) -> np.array:     
        encoded_input = self.tokenizer(f"clustering: {text}", padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.text_model(**encoded_input)

        text_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
        text_features = F.normalize(text_embeddings, p=2, dim=1).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy()

