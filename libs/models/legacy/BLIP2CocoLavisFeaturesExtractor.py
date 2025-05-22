from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
import numpy as np
import requests


class BLIP2Coco:
    def __init__(self):
        print('BLIP COCO.....')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="coco", is_eval=True, device=self.device)
        self.sample_image = Image.new('RGB',(10,10))
        self.sample_image = self.vis_processors["eval"](self.sample_image).unsqueeze(0).to(self.device)
    
    def get_image_features(self, image_path : str) -> np.array:
        if (image_path.startswith("http")):
            image = Image.open(requests.get(image_path, stream=True).raw)
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
    
# blip = BLIP()
# image_features = blip.get_image_features("/workspace/competitions/AIC_2023/SIU_Pumpkin/utils/test/000000039769.jpg")
# text_features = blip.get_text_features("a photo of two cats")
# print(image_features)
# print(text_features)
# similarity = torch.matmul(text_features, image_features.t())
# print(similarity)