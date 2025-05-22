from imagededup.methods import CNN
from imagededup.utils import plot_duplicates
import pillow_avif

def clusteration():
    method_object = CNN()
    duplicates = method_object.find_duplicates(image_dir='/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5/Keyframes_L01/keyframes/L01_V001',
                                               min_similarity_threshold=0.9,
                                               #max_distance_threshold=16,
                                               scores=True,
                                               outfile="CNN_output.json",
                                               recursive=True)
    # for image in duplicates:
    #     plot_duplicates('/workspace/competitions/AIC_2023/SIU_Pumpkin/utils/shot_clustering/Keyframes_L11', duplicates, image, outfile='/workspace/competitions/AIC_2023/SIU_Pumpkin/utils/shot_clustering/test/' + image)
    
clusteration()

import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def group_images(duplicates):
    visited = set()
    groups = []

    def dfs(image, group):
        if image not in visited:  # Ensure image is a string
            visited.add(image)
            group.add(image)
            for dup_info in duplicates.get(image, []):
                dup = dup_info[0]  # Extracting the duplicate file name
                if isinstance(dup, str):  # Ensure dup is a string
                    dfs(dup, group)

    for image in duplicates:
        if image not in visited:  # Ensure image is a string
            group = set()
            dfs(image, group)
            groups.append(sorted(group))  # Sorting for consistent output

    return groups

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Load the JSON data
file_path = 'CNN_output.json'
duplicates = load_json(file_path)

# Group the images
image_groups = group_images(duplicates)

# Save the grouped images as a new JSON file
output_file_path = 'grouped_images.json'
save_json(image_groups, output_file_path)

print(f"Grouped images saved to {output_file_path}")