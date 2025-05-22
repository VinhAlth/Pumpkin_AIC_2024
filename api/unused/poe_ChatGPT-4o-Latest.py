from poe_api_wrapper import PoeApi
import time  


import os
import json
import math
import requests
import flask
from flask import Flask, request
import re
import ujson
import ast

class POE:
    def __init__(self):
        tokens = {
            'p-b': 'hMIE-gpIXICD5g0VPkfuZw%3D%3D',
            'p-lat': 'AFLfsKu%2BtvkJtHWDiJ9SsDqg9aaK42p4kung%2FzOQug%3D%3D',
            'formkey': '22b268d6b6d642286cbdb18294b10fb966',
            '__cf_bm': 'MwcccvasIOVWO9PDfF1GzxYWhfa1_fY6XOe5nQpzKac-1726134953-1.0.1.1-Ik3CNXIZALAugKOp80kcK49vfbeBRUOLFkZD3FyZw6GuUVmcvmINkP9LvQKo8humxT.3bFnS_alVuyjqTSbtRQ',
            'cf_clearance': 'WdnV87BpTAAOsKUk3esUrIivXAcI7824zoRTIJf.4GM-1726134994-1.2.1.1-uFwJ62yhtJ9TvSDK3G1tJrfu4roefL.mMclNsu0p8eGoMMhFhrtooJGonF2ipjKs55yrbQEYZnNTAFTphpRY.wCGuEGD88KpV6ZHcxBl1rkz.s4Mx8aZaWKcj1F6uapcHEpC2pRt19YmeP6zzUoR4Q7sFdMDt6uL5jn54kL82oll.1nL5ciE1Ndnx251bciskFTFDdblbgpM0LBK9Fx_LGqphuuV56BJuIl9rbDmGoUuc8RVgVgg6j6jgrFMEdNBVfcx3C.1k9hwYuwS8kXACttjkHcN0PN8PgVkxj3XAUJpbyCXdPIc4pu4f3z22k1.aD.WgL6x7L3p3.0fVxyPy1w4lqQwx.f2p6zx6wFV5Q0IM98U_725OWR66h9ptocr'
            }
        self.client = PoeApi(tokens=tokens) 
        self.chatCode = False

    def _del(self):
        if self.chatCode:
            self.client.delete_chat("AIC2024", chatCode=self.chatCode)
        return 
    
    def image_list_to_answer(self, query: str, images_path: list = []):
        if self.chatCode:
            for chunk in self.client.send_message("AIC2024", query, file_path=images_path, chatCode=self.chatCode):
                pass
            return chunk["text"]
        else:
            for chunk in self.client.send_message("AIC2024", query, file_path=images_path):
                pass
            self.chatCode = chunk["chatCode"]
            return chunk["text"]

# start_time = time.time()
# bot = chatbot()
# end_time = time.time()
# execution_time = end_time - start_time  # Tính toán thời gian thực thi  
# print(f"Thời gian chạy: {execution_time:.6f} giây")  

# start_time = time.time()
# print(bot.run("Bao gạo giảm bao nhiêu phần trăm và số trên máy clarus là gì?", ["/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5/Keyframes_L11/keyframes/L11_V020/01456.jpg", "/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5/Keyframes_L04/keyframes/L04_V029/04711.jpg"]))
# end_time = time.time()
# execution_time = end_time - start_time  # Tính toán thời gian thực thi  
# print(f"Thời gian chạy: {execution_time:.6f} giây")  

# start_time = time.time()
# print(bot._del())
# end_time = time.time()
# execution_time = end_time - start_time  # Tính toán thời gian thực thi  
# print(f"Thời gian chạy: {execution_time:.6f} giây")

# Gọi lên chạy 
app = flask.Flask("API Chatbot")
app.config["DEBUG"] = True

@app.route('/generate')  # Specify POST method here
def generate():
    
    query = None
    frame_list = None
    results = ""
    
    if request.method == "POST":
        data = request.get_json()  # Get JSON data from request body
        query = data.get('query')
        frame_list = data.get('frame_list', [])
    elif request.method == "GET":
        query = request.args.get('query')
        frame_list = request.args.get('frame_list')
    
    if query!=None and frame_list!=None:
        frame_list = ast.literal_eval(frame_list)
        st = time.time()
        poe = POE()
        print("Init time: ", time.time()-st)
        st = time.time()
        results = poe.image_list_to_answer(query, frame_list)
        print("Process time: ", time.time()-st)
    
    
    response = flask.jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response