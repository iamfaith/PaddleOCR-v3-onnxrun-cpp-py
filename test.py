import paddlehub as hub
import cv2

ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)       # mkldnn加速仅在CPU下有效
result = ocr.recognize_text(images=[cv2.imread('python/images/2.jpg')], visualization=True)

print(result)