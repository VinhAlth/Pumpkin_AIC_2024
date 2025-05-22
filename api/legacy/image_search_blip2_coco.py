import flask
from flask import jsonify, request
import os 
import json
import requests
import sys 
import urllib.parse
from googletrans import Translator
from PIL import Image

import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

# Gọi lên chạy 
app = flask.Flask("API Image Search")
app.config["DEBUG"] = True

@app.route('/predict', methods=['POST', 'GET'])

def updateCurrentCode():
    global KEYFRAME_FOLDER_PATH
    image ="" #duong dan anh

    if request.method == "POST":
        image = request.json['image_url']
        k = request.json['k']
    else:
        image = request.args.get('image_url')
        k = request.args.get('k')

    
    print(f"image: {image}")
    # preprocessing text 
    faiss_url = f'http://0.0.0.0:8502/faiss_search?query={image}&k={k}&dataset=1'
    faiss_response = requests.get(faiss_url)

    search_results = faiss_response.json()
    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8505, debug=False)