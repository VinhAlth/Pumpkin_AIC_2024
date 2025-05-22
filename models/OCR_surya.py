from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

IMAGE_PATH = "/dataset/AIC2024/pumkin_dataset/0/frames/autoshot/Keyframes_L08/keyframes/L08_V020/04097.jpg"

image = Image.open(IMAGE_PATH)
langs = ["vi"] # Replace with your languages - optional but recommended
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
print(predictions)