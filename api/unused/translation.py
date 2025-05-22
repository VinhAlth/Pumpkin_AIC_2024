import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import flask
from flask import request


tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")
device_vi2en = torch.device("cuda")
model_vi2en.to(device_vi2en) 

def translate_vi2en(vi_texts: str) -> str:
    input_ids = tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(device_vi2en)
    output_ids = model_vi2en.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    en_texts = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    return en_texts

# Gọi lên chạy 
app = flask.Flask("API Translation")
app.config["DEBUG"] = True

@app.route('/preprocess')
def preprocess():
    text = ""
    
    if request.method == "POST":
        text = request.json['text']
        k = request.json['k']
    else:
        text = request.args.get('text')
        k = request.args.get('k')
    
    text.rstrip('.')
    
    text_en = translate_vi2en(text)
    print(f"trans: {text_en}")
    for idx, text in enumerate(text_en):
        text_en[idx] = text_en[idx].rstrip('.')
        
    response = flask.jsonify(text_en)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response

# if __name__ == '__main__':
   
#     app.run(host="0.0.0.0", port= 8505, debug=False)


# worker -> so phien lam viec cua api
# chay = flask: 1 worker 1 thread
# xu ly duoc 1 request, trong 1 request do chi xu ly duoc 1 dong 1 luc
# threads -> so luong lam viec cua pi

# -> tang worker tang thread

# flask -> experiment
# ?? -> production : stability, scalabity

# apache / nginx/ tornado
# WSGI server

# gunicorn

# worker: sync, async, gevent

# pip install gunicorn -> 
# gip install gevent

# gunicorn -w 1 -k gevent --threads 16 -b 0.0.0.0:8505 translation:app
# gunicorn -w 2 -k gevent --threads 16 -b 0.0.0.0:8505 translation:app
# -> tat ca model, code no se duoc nhan 2 len

# # Hien thi len UI
# -> worker: 10-20, threads 8-16
# Xu ly tim kiem
# mo hinh A thuong xuyen su dung: 2 worker
# mo hinh B, C it su dung hon: 1 work moi mo hinh
# Cong cu khac (dich, s2t)