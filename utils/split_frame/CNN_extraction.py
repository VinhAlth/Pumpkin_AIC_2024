import os
import pathlib
from pathlib import Path
from imagededup.methods import CNN
import json
import numpy as np

method_object = CNN()

frames_path = "/dataset/AIC2024/pumkin_dataset/2/frames/autoshot/"
save_path = "/dataset/AIC2024/pumkin_dataset/2/imagededup_autoshot/"

for frame in Path(frames_path).rglob('*.jpg'):
    print(frame)
    save_npy_path = Path(save_path + str(Path(str(Path(*frame.parts[frame.parts.index('autoshot'):])).replace(".jpg",""))))
    save_npy_path.parent.mkdir(parents=True, exist_ok=True)
    save_npy = method_object.encode_image(image_file=frame)
    np.save(save_npy_path, save_npy)