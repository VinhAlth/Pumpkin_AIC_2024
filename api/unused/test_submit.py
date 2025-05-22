import flask
from flask import request
import time

# Gọi lên chạy 
app = flask.Flask("API Translation")
app.config["DEBUG"] = True


ground_truth = [{"video_name":"L19_V011", "start_frame":26353, "end_frame":26676},
                ]
round_start_time = time.time()
current_query_index = 0

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    try:
        global current_query_index
        global ground_truth
        global round_start_time
        
        team_id = None
        video_name = None
        frame_name = None
        
        if request.method == "POST":
            team_id = request.json['team_id']
            video_name = request.json['video_name']
            frame_name = request.json['frame_name']

        
        return_status = "WRONG"
        
        if video_name == ground_truth[current_query_index]['video_name'] and int(frame_name)>=int(ground_truth[current_query_index]['start_frame']) and int(frame_name)<=int(ground_truth[current_query_index]['end_frame']):
            return_status = "CORRECT"
            

            
        response = flask.jsonify(return_status)
    except:
        response = flask.jsonify("Invalid Form")
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8505, debug=False)