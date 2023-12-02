import paddle
from paddlespeech.cli.kws import KWSExecutor

kws_executor = KWSExecutor()
result = kws_executor(
    # audio_file='./hey_snips.wav',
    audio_file='/home/faith/PaddleOCR-v3-onnxrun-cpp-py/keyword_spotting/hey_snips.wav',
    threshold=0.8,
    model='mdtc_heysnips',
    config=None,
    ckpt_path=None,
    device=paddle.get_device())
print('KWS Result: \n{}'.format(result))