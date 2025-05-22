from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os, zipfile
from transformers.file_utils import cached_path, hf_bucket_url
from datasets import load_dataset
import soundfile as sf
import torch
torch.cuda.empty_cache()
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import subprocess
from datetime import datetime


import sys
sys.path.append('/workspace/competitions/AIC_2024/SIU_Pumpkin/')

import flask
from flask import request

DEVICE = "cuda"

cache_dir = './cache/'

from huggingface_hub import login
login(token="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")

model_checkpoint = "phamgiakiet273/wav2vec2-base-vi-vlsp530h"


processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)
model = Wav2Vec2ForCTC.from_pretrained(model_checkpoint)
lm_file = cache_dir + 'vi_lm_4grams.bin'

def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    vocab_list[tokenizer.pad_token_id] = ""
    vocab_list[tokenizer.unk_token_id] = ""
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet,
                                   language_model=LanguageModel(lm_model))
    return decoder

ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, lm_file)


def map_to_array(batch):
    speech, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    print(batch)
    return batch

def speech_to_text(audio_input):
    ds = []
    if '/' in audio_input:
        ds = map_to_array({
            "file": audio_input
        })
        ds = ds['speech']
    else:
        ds = audio_input
    input_values = processor(
        ds, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values.to(DEVICE)
    model.to(DEVICE)
    logits = model(input_values).logits[0]
    hotwords = [""]
    beam_search_output = ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500, hotwords = hotwords, hotword_weight=5.0)
    return beam_search_output

def convert_audio(audio_path):
    now = datetime.now()
    curtime = now.strftime("-%d%m%Y-%H%M%S")
    if "mp3" in audio_path:
        new_audio_path = audio_path.replace(".mp3","-16k" + curtime + ".wav")
    else:
        new_audio_path = audio_path.replace(".wav","-16k" + curtime + ".wav")
    subprocess.call(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-map", " 0:a:0", "-b:a", "96k", "-ac", "1", new_audio_path])
    return new_audio_path

#---API---

app = flask.Flask("API for Speech To Text")
app.config["DEBUG"] = False

@app.route('/audio_path', methods=['POST', 'GET'])
def updateCurrentCode():
    
    if request.method == 'GET':
        default_path = "/cache/recording/"
        audio_path = request.args.get("audio_path")
        audio_path = os.path.join(default_path, audio_path)
        new_audio_path = convert_audio(audio_path)

    elif request.method == 'POST':
        default_path = "/cache/recording/"
        data = request.get_json()
        audio_path = data.get('audio_path')
        audio_path = os.path.join(default_path, audio_path)
        new_audio_path = convert_audio(audio_path)
    
    text = speech_to_text(new_audio_path)

    response = flask.jsonify(text)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Content-type','application/json; charset=utf-8')
    response.success = True
    return response

if __name__ == '__main__':
    speech_to_text("/workspace/ai_intern/kietpg/adapter_run-16k.wav")
    app.run(host="0.0.0.0", port=8505, debug=False)