from typing import Any
import easyocr
# reader = easyocr.Reader(['vi']) # this needs to run only once to load the model into memory
# result = reader.readtext()
# print(result[0][])

class OCR:
    def __init__(self, threshold=0.6):
        self.reader = easyocr.Reader(['vi'])
        self.threshold = threshold
        
    def __call__(self, image_path) -> Any:
        result = self.reader.readtext(image_path)
        full_text = ""
        for object in result:
            text = object[1]
            score = object[2]
            if score>=self.threshold:
                full_text+=text + " "
        return full_text
    
ocr = OCR()
print(ocr("/dataset/AIC2024/pumkin_dataset/0/frames/autoshot/Keyframes_L08/keyframes/L08_V020/04097.jpg"))