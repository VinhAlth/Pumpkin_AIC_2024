import flask
from flask import jsonify, request
import urllib.parse
import requests
import json
import sys
sys.path.append("/workspace/competitions/AIC_2024/SIU_Pumpkin/")

# Gọi lên chạy 
app = flask.Flask("API Text Search")
app.config["DEBUG"] = True

@app.route('/predict', methods=['POST', 'GET'])
def updateCurrentCode():
    global KEYFRAME_FOLDER_PATH
    text ="" #câu truy vấn => duong dan anh

    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']
    else:
        text = request.args.get('text')
        k = request.args.get('k')

    
    print(f"text: {text}")
    
    # Call FAISS API using GET
    faiss_url = f'http://0.0.0.0:8502/faiss_search?query={text}&k={k}'
    faiss_response = requests.get(faiss_url)

    search_results = faiss_response.json()
    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8504, debug=False)