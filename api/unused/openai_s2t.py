from openai import OpenAI
import json
from datetime import datetime
import time
import flask
from flask import request
import subprocess
import os

class S2T:
    def __init__(self) -> None:

        self.client = OpenAI(api_key='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')


    def get_transcript(self, audio_path):

        audio_file= open(audio_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        
        return transcription.text

    def __call__(self, audio_path):
        
        transcript = self.get_transcript(audio_path)
        
        return transcript

#convert .mp3 and .wav audio to correct format for model (16000hz, mono channel, .wav) and save as new file
def convert_audio(audio_path):
    now = datetime.now()
    curtime = now.strftime("-%d%m%Y-%H%M%S")
    if "mp3" in audio_path:
        new_audio_path = audio_path.replace(".mp3","-16k" + curtime + ".wav")
    else:
        new_audio_path = audio_path.replace(".wav","-16k" + curtime + ".wav")
    #subprocess.call(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", new_audio_path])
    #subprocess.call(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-map", " 0:a:0", "-b:a", "96k", "-ac", "1", new_audio_path])
    #data, samplerate = soundfile.read(audio_path)
    #soundfile.write(new_audio_path, data, samplerate, subtype='PCM_16')
    #return new_audio_path
    return audio_path

#---API---

app = flask.Flask("API for Speech To Text")
app.config["DEBUG"] = False

@app.route('/s2t', methods=['POST', 'GET'])
def updateCurrentCode():
    global s2t
    audio_path = None
    new_audio_path = None


    default_path = "/workspace/competitions/AIC_2024/SIU_Pumpkin/utils/temp/"
    audio_path = request.args.get("audio_path")
    #print(audio_path)
    audio_path = os.path.join(default_path, audio_path)
    new_audio_path = convert_audio(audio_path)
    text = s2t(new_audio_path)
        
    response = flask.jsonify(text)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Content-type','application/json; charset=utf-8')
    response.success = True
    return response

if __name__ == '__main__':
    #subprocess.call(["nvidia-settings", "-a", "[gpu:0]/GPUPowerMizerMode=1"])
    s2t = S2T()
    app.run(host="0.0.0.0", port=8507, debug=False)
    #from waitress import serve
    #serve(app, host='0.0.0.0', port=8073)