import os
import json
from pathlib import Path

def create_json_structure(root_dir):
    json_data = {}

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                jpg_path = Path(root) / file
                video_name = jpg_path.parent.name + '.mp4'  # Assuming parent folder is the video name
                if video_name not in json_data:
                    json_data[video_name] = {}
                json_data[video_name][file] = []  # Empty list as required

    return json_data

# Specify your root directory here
root_directory = "/dataset/AIC2024/pumkin_dataset/1/frames/autoshot"

# Generate the JSON structure
result_json = create_json_structure(root_directory)

# Save the JSON structure to a file
output_json_path = "/dataset/AIC2024/pumkin_dataset/1/speech_to_text/transcript_all_segmented_empty.json"
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(result_json, json_file, ensure_ascii=False, indent=4)

print(f"JSON file created at: {output_json_path}")