import os, pathlib
import numpy as np
from . import config
from .Stream import Stream_SSD

BASE_DIR = pathlib.Path(__file__)
stride = config.stride
sample_rate = 16000
user_voice_path = os.path.join(BASE_DIR.parent, 'users_voices.json')
stream_ssd_calss = Stream_SSD(user_voice_path)
# stream_ssd_calss.create_and_train_model(os.path.join(BASE_DIR.parent,'supervised_speaker_diarization', 'saved_model' ))
stream_ssd_calss.load_clf_model(os.path.join(BASE_DIR.parent,'saved_model'))

def speech_diarization(audio_byte):
    speaker = stream_ssd_calss.recognize(audio_byte)
    return speaker