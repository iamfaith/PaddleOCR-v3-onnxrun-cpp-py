import torch
# net = torch.load('/home/faith/PaddleOCR-v3-onnxrun-cpp-py/yolov8/yolov8n.pt', map_location='cpu') # 加载预训练模型

from ultralytics import YOLO

model = YOLO('/home/faith/PaddleOCR-v3-onnxrun-cpp-py/yolov8/yolov8n.pt') 
model.export(format="onnx", imgsz=[640,640])
# print(net)
# net.eval() # 设置为评估模式
# dummpy_input = torch.randn(1, 3, 640, 640) # 创建一个虚拟输入张量
# torch.onnx.export(net, dummpy_input, 'yolov8n.onnx', export_params=True, input_names=['input'], output_names=['output']) # 导出模型为ONNX格式
