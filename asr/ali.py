import os
import logging
import torch
import soundfile

# /home/faith/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
def onnx():
    from funasr_onnx import Paraformer
    from pathlib import Path

    model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" # work
    
                
    
    model = Paraformer(model_dir, batch_size=1, quantize=True)

    wav_path = ['{}/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav'.format(Path.home())]
    wav_path = ["/home/faith/PaddleOCR-v3-onnxrun-cpp-py/asr/zh.wav"]

    result = model(wav_path)
    print(result)


def onnx_stream():
    from funasr_onnx.paraformer_online_bin import Paraformer
    model_dir = '/home/faith/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx' #https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/files
                # 'damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online'  # https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/files
    model = Paraformer(model_dir, batch_size=1, quantize=True)
    
    speech, sample_rate = soundfile.read("/home/faith/PaddleOCR-v3-onnxrun-cpp-py/asr/zh.wav")
    print(speech.shape, sample_rate)
    speech_length = speech.shape[0]

    sample_offset = 0
    chunk_size = [5, 10, 5] #[5, 10, 5] 600ms, [8, 8, 4] 480ms
    stride_size =  chunk_size[1] * 960
    param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size}
    final_result = ""

    for sample_offset in range(0, speech_length, min(stride_size, speech_length - sample_offset)):
        if sample_offset + stride_size >= speech_length - 1:
            stride_size = speech_length - sample_offset
            param_dict["is_final"] = True
        rec_result = model(audio_in=speech[sample_offset: sample_offset + stride_size],
                                        param_dict=param_dict)
        print(rec_result)
        # if len(rec_result) != 0:
        #     final_result += rec_result['text'] + " "
        #     print(rec_result)
    print(final_result)
    
onnx_stream()

def modelscope():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.utils.logger import get_logger

    #  /home/faith/.cache/modelscope/
    logger = get_logger(log_level=logging.CRITICAL)
    logger.setLevel(logging.CRITICAL)

    os.environ["MODELSCOPE_CACHE"] = "./"
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
        model_revision='v1.0.6',
        mode="paraformer_streaming"
    )

    model_dir = os.path.join(os.environ["MODELSCOPE_CACHE"], "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
    # speech, sample_rate = soundfile.read(os.path.join(model_dir, "example/asr_example.wav"))
    speech, sample_rate = soundfile.read("/home/faith/PaddleOCR-v3-onnxrun-cpp-py/asr/zh.wav")
    speech_length = speech.shape[0]

    sample_offset = 0
    chunk_size = [5, 10, 5] #[5, 10, 5] 600ms, [8, 8, 4] 480ms
    stride_size =  chunk_size[1] * 960
    param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size}
    final_result = ""

    for sample_offset in range(0, speech_length, min(stride_size, speech_length - sample_offset)):
        if sample_offset + stride_size >= speech_length - 1:
            stride_size = speech_length - sample_offset
            param_dict["is_final"] = True
        rec_result = inference_pipeline(audio_in=speech[sample_offset: sample_offset + stride_size],
                                        param_dict=param_dict)
        if len(rec_result) != 0:
            final_result += rec_result['text'] + " "
            print(rec_result)
    print(final_result)