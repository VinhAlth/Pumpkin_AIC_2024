from paddleocr import PaddleOCR
from PIL import Image

class PADDLE:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi')
        
    def __call__(self, image_path):
        result = self.ocr.ocr(image_path, cls=True)
        return result
    
paddle = PADDLE()
img_path = "/dataset/AIC2024/pumkin_dataset/0/frames/autoshot/Keyframes_L12/keyframes/L12_V004/05555.jpg"
result = paddle(img_path)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)