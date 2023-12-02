from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.text import TextExecutor #python -m pip install paddlenlp==2.5.2
import paddle
text_executor = TextExecutor()

# from paddlespeech.cli.text import TextExecutor
# text_executor = TextExecutor()
asr = ASRExecutor()

# ./main -m ../../ggml-large-v3-q5_0.bin -f /home/faith/PaddleOCR-v3-onnxrun-cpp-py/asr/zh.wav --print-colors --prompt "请用简体中文输出"
# import paddlespeech.s2t.models.u2.u2.U2Model
result = asr(audio_file = "/home/faith/PaddleOCR-v3-onnxrun-cpp-py/asr/zh.wav")
print(result)
result = text_executor(
    text=result,
    task='punc',
    model='ernie_linear_p7_wudao',
    lang='zh',
    config=None,
    ckpt_path=None,
    punc_vocab=None,
    device=paddle.get_device())
print('Text Result: \n{}'.format(result))


# result = text_executor(
#         text=result,
#         task='punc',
#         model='ernie_linear_p3_wudao')


# paddle2onnx --model_dir encoder --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file encoder.onnx --opset_version 12 --enable_onnx_checker True

########### export encoder
########## 1 x 80
        # # ## before embedding 80 -> 512
        # model = paddle.jit.to_static(
        #     self.model.encoder,
        #     input_spec=[
        #         paddle.static.InputSpec(
        #             shape=[1, None, 80], dtype=paddle.float32), paddle.static.InputSpec(
        #             shape=[1], dtype=paddle.int64), 16, paddle.static.InputSpec(
        #             shape=[None, None, 1], dtype=paddle.bool)
        #     ])

        # # Save in static graph model.
        # paddle.jit.save(model, os.path.join("/home/faith/PaddleOCR-v3-onnxrun-cpp-py/asr/", "inference"))