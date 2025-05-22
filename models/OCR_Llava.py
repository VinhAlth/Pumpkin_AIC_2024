from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nOutput ONLY the OCR content of the image, skipping the red running line at the bottom and the logo as well as the time at the top right. ASSISTANT:"
img_path = "/dataset/AIC2024/pumkin_dataset/0/frames/autoshot/Keyframes_L08/keyframes/L08_V020/04097.jpg"
image = Image.open(img_path)

inputs = processor(images=image, text=prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=15)
res = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(res)