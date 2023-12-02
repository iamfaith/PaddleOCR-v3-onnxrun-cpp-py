
# https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/text_to_speech/README_cn.md
# conda install libgcc
#  strings /opt/nvidia/nsight-systems/2022.4.2/host-linux-x64/libstdc++.so.6 | grep GLIBCXX_3.4.29
import paddle
from paddlespeech.cli.tts import TTSExecutor
tts_executor = TTSExecutor()
# wav_file = tts_executor(
#     text='今天的天气不错啊',
#     output='output.wav',
#     am='fastspeech2_csmsc',
#     am_config=None,
#     am_ckpt=None,
#     am_stat=None,
#     spk_id=0,
#     phones_dict=None,
#     tones_dict=None,
#     speaker_dict=None,
#     voc='pwgan_csmsc',
#     voc_config=None,
#     voc_ckpt=None,
#     voc_stat=None,
#     lang='zh',
#     device=paddle.get_device())
# print('Wave file has been generated: {}'.format(wav_file))

from paddlespeech.cli.tts import TTSExecutor
tts_executor = TTSExecutor()
wav_file = tts_executor(
    text='你这个大马趴大臭蹲',
    output='output.wav',
    am='fastspeech2_csmsc',
    voc='hifigan_csmsc',
    lang='zh',
    use_onnx=True,
    cpu_threads=16)