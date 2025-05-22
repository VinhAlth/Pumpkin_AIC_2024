#!/usr/bin/env python3

from imagedominantcolor import DominantColor
import os
import time
import json
from operator import itemgetter
import colorsys
import math

path = input("Enter directory you'd like to sort: ")

os.chdir(os.path.expanduser(path))

print(os.getcwd())

def get_lum(r, g, b):
    return math.sqrt(0.241 * r + 0.691 * g + 0.068 * b)

def step(r, g, b, repetitions=1):
    lum = get_lum(r, g, b)

    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)

    if h2 % 2 == 1:
        v2 = repetitions - v2
        lum = repetitions - lum

    return (h2, lum, v2)

files = [f for f in os.listdir('.') if os.path.isfile(f)]
print('files', files)
color_list = []
for index, file in enumerate(files):
    if not file.startswith('.'):
        color_thief = DominantColor(file)
        print('Getting color for:', file, '(' + str(index + 1) + '/' + str(len(files)) + ')')
        st = time.time()
        dominant_color = color_thief.rgb
        print(dominant_color)
        print("Retrieve color time: ", time.time() - st)
        # palette = color_thief.get_palette(color_count=6)
        st = time.time()
        color_list.append({'file': file, 'rgb': dominant_color, 'hsv': colorsys.rgb_to_hsv(*dominant_color), 'hls': colorsys.rgb_to_hls(*dominant_color)})
        print("Convert: ", time.time() - st)
    else:
        print('Skipping:', file, '(' + str(index + 1) + '/' + str(len(files)) + ')')

sorted_colors = sorted(color_list, key=lambda k: k['hls'])
sorted_lum = sorted(color_list, key=lambda k: get_lum(*k['rgb']))
sorted_step = sorted(color_list, key=lambda k: step(k['rgb'][0], k['rgb'][1], k['rgb'][2], 8))

output_file_name = '.color_' + str(time.time())
with open(output_file_name, 'w+') as output_file:
    json.dump(sorted_step, output_file)

for index, file in enumerate(sorted_step):
    os.rename(file['file'], str(index) + '. ' + file['file'])
