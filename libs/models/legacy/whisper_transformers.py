from transformers import pipeline
import torch
import numpy

pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current PyTorch device is set to", pytorch_device)


audio_path = "/dataset/AIC2024/pumkin_dataset/0/speech_to_text/L01_V001/speech169-187.wav"

pipe = pipeline(model="openai/whisper-large-v2", device=pytorch_device)
predictions = pipe(audio_path, chunk_length_s=20, stride_length_s=(5), generate_kwargs={"task": "transcribe"})

predictions = predictions["text"]
print(predictions)