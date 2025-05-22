import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/nhihtc/pretrained_weights'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
sys.path.append("/workspace/nhihtc/work/AIC")

class BLIP2Coco:
    def __init__(self):
        print('BLIP COCO.....')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="coco", is_eval=True, device=self.device)
        self.sample_image = Image.new('RGB',(10,10))
        self.sample_image = self.vis_processors["eval"](self.sample_image).unsqueeze(0).to(self.device)
    
    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_path)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        sample = {"image": image, "text_input": ""}
        image_features = self.model.extract_features(sample, mode="image")
        #image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        image_features  = image_features.image_embeds_proj[:,0,:]
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

    def get_text_features(self, text : str) -> np.array:
        text_input = self.txt_processors["eval"](text)
        sample = {"image": self.sample_image, "text_input": text_input}
        text_features = self.model.extract_features(sample, mode="text")
        #text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features.text_embeds_proj[:,0,:]
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().detach().numpy()
    
# blip = BLIP2Coco()
# query = ['cat', 'dog', 'fish', 'plane']
# # image_path = [torch.load('response_vector_cat.pt'), torch.load('response_vector.pt'), torch.load('response_vector_fish.pt'), torch.load('response_vector_plane.pt')]
# image_path = ["/workspace/nhihtc/cat.jpg", "/workspace/nhihtc/dog.webp", "/workspace/nhihtc/fish.webp", "/workspace/nhihtc/plane.jpg"]
# for i in range(4):
#     print('image about', query[i])
#     for j in range(4):
#         image_features = blip.get_image_features(image_path[i])
#         text_features = blip.get_text_features(query[j])
#         similarity = (image_features @ text_features.t())
#         print(query[j],":",similarity)
