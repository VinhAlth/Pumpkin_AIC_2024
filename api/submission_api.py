# gunicorn -w 1 -k gevent --threads 10 -b 0.0.0.0:8508 submission_api:app

import flask
from flask import request
import time

# Gọi lên chạy 
app = flask.Flask("API Translation")
app.config["DEBUG"] = True

from datetime import datetime
import ujson

ground_truth = [#0
                {"video_name":"L00_V000", "start_frame":0, "end_frame":0, "qa": 0},
                #1
                {"video_name":"L20_V015", "start_frame":28264, "end_frame":28496, "qa": None},
                #2
                {"video_name":"L24_V040", "start_frame":350, "end_frame":1131, "qa": None},
                #3
                {"video_name":"L11_V006", "start_frame":27799, "end_frame":28072, "qa": 14},
                #4
                {"video_name":"L09_V010", "start_frame":31686, "end_frame":31772, "qa": None},
                #5
                {"video_name":"L08_V016", "start_frame":8494, "end_frame":8818, "qa": None},
                #6
                {"video_name":"L17_V006", "start_frame":36890, "end_frame":37955, "qa": None},
                #7
                {"video_name":"L14_V012", "start_frame":20406, "end_frame":20491, "qa": None},
                #8
                {"video_name":"L01_V021", "start_frame":13903, "end_frame":14517, "qa": None},
                #9
                {"video_name":"L07_V014", "start_frame":21850, "end_frame":23125, "qa": None},
                #10
                {"video_name":"L19_V019", "start_frame":31534, "end_frame":31640, "qa": 4},
                #11
                {"video_name":"L06_V029", "start_frame":8817, "end_frame":9420, "qa": None},
                #12
                {"video_name":"L03_V009", "start_frame":14963, "end_frame":15495, "qa": None},
                #13
                {"video_name":"L22_V023", "start_frame":18093, "end_frame":18667, "qa": None},
                #14
                {"video_name":"L08_V012", "start_frame":18895, "end_frame":20384, "qa": None},
                #15
                {"video_name":"L10_V009", "start_frame":5750, "end_frame":5923, "qa": 2024},
                #16
                {"video_name":"L22_V012", "start_frame":35004, "end_frame":35629, "qa": 8},
                #17
                {"video_name":"L22_V012", "start_frame":35004, "end_frame":35629, "qa": None}]




round_start_time = time.time()
current_query_index = 0

score_team = {"pumpkin": 0,
              "kitty": 0,
              "champion": 0}

submitted_this_round = {"pumpkin": False,
                        "kitty": False,
                        "champion": False}

wrong_this_round = {"pumpkin": 0,
                    "kitty": 0,
                    "champion": 0}


#api.siu.edu.vn/aic/6/start_round?start_round_index=3
#api.siu.edu.vn/aic/6/start_round?start_round_index=0
#api.siu.edu.vn/aic/6/start_round?start_round_index=4

@app.route('/start_round', methods=['GET'])
def start_round():
    
    global current_query_index
    global ground_truth
    global round_start_time
    global wrong_this_round
    global score_team
    global submitted_this_round
    
    
    start_round_index = None
    
    if request.method == "GET":
        start_round_index = request.args.get('start_round_index')
    
    current_query_index = int(start_round_index)
    
    #reset a couple of things?
    round_start_time = time.time()
    submitted_this_round["pumpkin"] = False  
    submitted_this_round["kitty"] = False  
    submitted_this_round["champion"] = False  
    wrong_this_round["pumpkin"] = 0
    wrong_this_round["champion"] = 0
    wrong_this_round["kitty"] = 0
    
    #return
    response = flask.jsonify(ground_truth[current_query_index])
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response


@app.route('/get_status', methods=['GET'])
def get_status():
    global current_query_index
    global ground_truth
    global round_start_time
    global wrong_this_round
    global score_team
    global submitted_this_round
    cur_status = {"Current Query" : current_query_index,
                              "Round Start Time" : round_start_time,
                              "Wrong This Round" : wrong_this_round,
                              "Score Teams" : score_team,
                              "Submitted": submitted_this_round}
    log_name = "log" + str(datetime.now())+".json"
    with open("logs/" + log_name, 'w') as write_log:
        ujson.dump(cur_status, write_log, indent=4)
    response = flask.jsonify(cur_status)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response

@app.route('/submit', methods=['POST'])
def submit():
    try:
        global current_query_index
        global ground_truth
        global round_start_time
        global wrong_this_round
        global score_team
        global submitted_this_round
        
        team_id = None
        video_name = None
        frame_name = None
        qa = None
        
        if request.method == "POST":
            team_id = request.json['team_id']
            video_name = request.json['video_name']
            frame_name = request.json['frame_name']
            if ground_truth[current_query_index]['qa']!=None:
                qa = request.json['qa']

        
        return_status = "WRONG"
        score = 0
        
        if video_name == ground_truth[current_query_index]['video_name'] and int(frame_name)>=int(ground_truth[current_query_index]['start_frame']) and int(frame_name)<=int(ground_truth[current_query_index]['end_frame']):
            if ground_truth[current_query_index]['qa']==None:
                return_status = "CORRECT"
                if submitted_this_round[team_id]==False:
                    score = max(100-int(time.time()-round_start_time)-10*int(wrong_this_round[team_id]),10) 
                    score_team[team_id]+=score
                    submitted_this_round[team_id] = True
            elif int(qa) == int(ground_truth[current_query_index]['qa']):
                return_status = "CORRECT"
                if submitted_this_round[team_id]==False:
                    score = max(100-int(time.time()-round_start_time)-10*int(wrong_this_round[team_id]),10) 
                    score_team[team_id]+=score
                    submitted_this_round[team_id] = True
            
        if return_status!="CORRECT":
            wrong_this_round[team_id]+=1
                
        response = flask.jsonify(return_status + " " + str(score))
    except:
         response = flask.jsonify("Invalid request")
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8505, debug=False)