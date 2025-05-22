from pathlib import Path
import ujson
import math
import colorsys
import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")
from utils.sort_by_color import get_dominant_color

import os

# frames_path_0 = "/dataset/AIC2024/pumkin_dataset/0/frames/autoshot/"
# save_path_0 = "/dataset/AIC2024/pumkin_dataset/0/colorsys_autoshot/"

# frames_path_1 = "/dataset/AIC2024/pumkin_dataset/1/frames/autoshot/"
# save_path_1 = "/dataset/AIC2024/pumkin_dataset/1/colorsys_autoshot/"


frames_path_2 = "/dataset/AIC2024/pumkin_dataset/2/frames/autoshot/"
save_path_2 = "/dataset/AIC2024/pumkin_dataset/2/colorsys_autoshot/"


# def get_lum(r, g, b):
#     return math.sqrt(0.241 * r + 0.691 * g + 0.068 * b)

# def step(r, g, b, repetitions=1):
#     lum = get_lum(r, g, b)

#     h, s, v = colorsys.rgb_to_hsv(r, g, b)
#     h2 = int(h * repetitions)
#     lum2 = int(lum * repetitions)
#     v2 = int(v * repetitions)

#     if h2 % 2 == 1:
#         v2 = repetitions - v2
#         lum = repetitions - lum

#     return (h2, lum, v2)

# print("ONE")
# for frame in Path(frames_path_0).rglob('*.jpg'):
#     #print(frame)
#     save_json_path = Path(save_path_0 + str(Path(str(Path(*frame.parts[frame.parts.index('autoshot'):])).replace(".jpg",""))))
#     if os.path.exists(save_json_path):
#         continue    
#     save_json_path.parent.mkdir(parents=True, exist_ok=True)
#     dominant_color = get_dominant_color(frame)
#     try:
#         step_sort = {'rgb': dominant_color, 'hsv': colorsys.rgb_to_hsv(*dominant_color), 'hls': colorsys.rgb_to_hls(*dominant_color)}
#         with open(str(save_json_path) + ".json", 'w') as file:
#             ujson.dump(step_sort, file, indent=4)
#     except:
#         print("Failed")
        
# print("TWO")
# for frame in Path(frames_path_1).rglob('*.jpg'):
#     print(frame)
#     save_json_path = Path(save_path_1 + str(Path(str(Path(*frame.parts[frame.parts.index('autoshot'):])).replace(".jpg",""))))
#     if os.path.exists(save_json_path):
#         continue
#     save_json_path.parent.mkdir(parents=True, exist_ok=True)
#     dominant_color = get_dominant_color(frame)
#     try:
#         step_sort = {'rgb': dominant_color, 'hsv': colorsys.rgb_to_hsv(*dominant_color), 'hls': colorsys.rgb_to_hls(*dominant_color)}
#         with open(str(save_json_path) + ".json", 'w') as file:
#             ujson.dump(step_sort, file, indent=4)
#     except:
#         print("Failed")

print("THREE")
for frame in Path(frames_path_2).rglob('*.jpg'):
    print(frame)
    save_json_path = Path(save_path_2 + str(Path(str(Path(*frame.parts[frame.parts.index('autoshot'):])).replace(".jpg",""))))
    save_json_path.parent.mkdir(parents=True, exist_ok=True)
    dominant_color = get_dominant_color(frame)
    try:
        step_sort = {'rgb': dominant_color, 'hsv': colorsys.rgb_to_hsv(*dominant_color), 'hls': colorsys.rgb_to_hls(*dominant_color)}
        with open(str(save_json_path) + ".json", 'w') as file:
            ujson.dump(step_sort, file, indent=4)
    except:
        print("Failed")