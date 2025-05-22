from imagededup.methods import CNN
import json
method_object = CNN()
# import time
# st = time.time()
# encodings = method_object.encode_image(image_file='/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5/Keyframes_L01/keyframes/L01_V001/00000.jpg')
# print(time.time()-st)
# st = time.time()
# encodings = method_object.encode_image(image_file='/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5/Keyframes_L01/keyframes/L01_V001/00000.jpg')
# print(time.time()-st)
# st = time.time()
encodings = method_object.encode_image(image_file='/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5/Keyframes_L01/keyframes/L01_V001/00000.jpg')
#print(time.time()-st)
print(encodings)
# save_encondings = {key: value.tolist() for key, value in encodings.items()}

# with open("/dataset/AIC2024/pumkin_dataset/utils/CNN_output.json", 'w') as json_file:
#     json.dump(save_encondings, json_file, indent=4)  # 'indent=4' is optional, it makes the file more readable

