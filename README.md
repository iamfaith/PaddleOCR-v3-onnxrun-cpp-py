# PaddleOCR-v3-onnxrun-cpp-py
使用ONNXRuntime部署PaddleOCR-v3, 包含C++和Python两个版本的程序。

从百度PaddlePaddle团队的PaddleOCR项目里导出的onnx文件，使用onnxruntime部署，
从而摆脱对深度学习框架PaddlePaddle的依赖。起初想用opnecv部署的，可是opencv的
dnn模块读取onnx文件出错了，无赖只能使用onnxruntime部署。
本套程序里包含dbnet文件检测，文字方向分类，crnn文字识别三个模块，
onnx文件大小不超过15M




你好，这是 Bing。我可以帮你找到一些关于 PP-OCRv4 模型转换为 onnxruntime 的信息。😊

PP-OCRv4 是一个基于飞桨框架的 OCR 模型，它可以对图片中的文字进行检测和识别¹。要将 PP-OCRv4 模型转换为 onnxruntime，你可以参考以下几个步骤：

- 首先，你需要安装 paddle2onnx 工具，它可以将飞桨模型转换为 ONNX 格式²。你可以使用以下命令安装 paddle2onnx：

```bash
python -m pip install paddle2onnx
```

- 然后，你需要下载 PP-OCRv4 模型的部署文件，包括 inference.pdmodel 和 inference.pdiparams¹。你可以使用以下命令下载 PP-OCRv4 模型：

```bash
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_PP-OCRv4_det_infer.tar && tar xf ch_PP-OCRv4_det_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_PP-OCRv4_rec_infer.tar && tar xf ch_PP-OCRv4_rec_infer.tar
```

- 接下来，你可以使用 paddle2onnx 命令行工具将 PP-OCRv4 模型转换为 ONNX 格式²。你需要指定模型的输入形状和数据类型，以及 ONNX 的版本号。你可以使用以下命令转换 PP-OCRv4 模型：

```bash
# 转换检测模型
paddle2onnx --model_dir ch_PP-OCRv4_det_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ch_PP-OCRv4_det.onnx --opset_version 11 --enable_onnx_checker True

# 转换识别模型
paddle2onnx --model_dir ch_PP-OCRv4_rec_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ch_PP-OCRv4_rec.onnx --opset_version 11 --enable_onnx_checker True
```

- 最后，你可以使用 onnxruntime 来加载和运行 ONNX 模型³。你需要安装 onnxruntime 和 opencv 库，然后编写相应的代码来读取图片，调用模型，处理输出，等等。你可以参考以下链接中的示例代码⁴⁵。

希望这些信息对你有帮助。如果你还有其他问题，欢迎继续和我聊天。😊

Source: Conversation with Bing, 11/24/2023
(1) GitHub - triwinds/ppocr-onnx: 利用 onnxruntime 及 PaddleOCR 提供的模型 .... https://github.com/triwinds/ppocr-onnx.
(2) 飞桨模型转 ONNX 模型-使用文档-PaddlePaddle深度学习平台. https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/advanced/model_to_onnx_cn.html.
(3) 如何将Pytorch模型转ONNX格式并使用OnnxRuntime推理-云社区-华为云. https://bbs.huaweicloud.com/blogs/detail/180532.
(4) 从pytorch转换到onnx - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/422290231.
(5) 在C++上利用onnxruntime （CUDA）和 opencv 部署模型onnx. https://blog.csdn.net/qq_42995327/article/details/121051991.
(6) undefined. https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.tar.gz.





paddle2onnx --model_dir inference_model --model_filename ppocrv3_rec.pdmodel --params_filename ppocrv3_rec.pdiparams --save_file ch_PP-OCRv3_rec.onnx --opset_version 11 --enable_onnx_checker True


 paddle2onnx --model_dir inference_model --model_filename ppocrv3_det.pdmodel --params_filename ppocrv3_det.pdiparams --save_file ppocrv3_det.onnx --opset_version 11 --enable_onnx_checker True


'/home/faith/.paddlehub/modules/ch_pp_ocrv3_det/inference_model/ppocrv3_det.pdmodel'



 hub run ch_pp-ocrv3 --input_path 'python/images/B5PQ23.png'

 https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/models_list.md