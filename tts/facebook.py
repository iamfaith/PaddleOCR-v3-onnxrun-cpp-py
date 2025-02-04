#%%
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd


models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/tts_transformer-zh-cv7_css10",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
# model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

text = "您好，这是试运行。"

sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction(task, models, generator, sample)

ipd.Audio(wav, rate=rate)