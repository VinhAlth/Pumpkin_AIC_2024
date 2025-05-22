import os
import pathlib
from pathlib import Path
from PIL import Image
import pillow_avif

# dataset_0 = "/dataset/AIC2023/pumkin_dataset/0/frames/pyscenedetect_t_5/"
# dataset_1 = "/dataset/AIC2023/pumkin_dataset/1/frames/pyscenedetect_t_5/"
# dataset_2 = "/dataset/AIC2023/pumkin_dataset/2/frames/pyscenedetect_t_5/"
dataset_test = "/dataset/AIC2024/pumkin_dataset/2/frames/autoshot/"
# low_res_path = "/dataset/AIC2023/pumkin_dataset/0/frames/low_res/"

def compress_image_folder(high_res_path):
    for image in Path(high_res_path).glob("**/*.jpg"):
        if not image.is_file():  # Skip directories
            continue
        image = str(image)
        picture = Image.open(image)
        low_res_folder = Path(image.replace("autoshot","low_res_autoshot")).parents[0]
        low_res_folder.mkdir(parents=True, exist_ok=True)
        low_res_image = str(image).replace("autoshot","low_res_autoshot")
        picture.save(str(low_res_image).replace(".jpg",".avif"), "AVIF", optimize=True, quality=10)
        
# compress_image_folder(dataset_0)
# compress_image_folder(dataset_1)
compress_image_folder(dataset_test)