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
    audio_bufers = np.frombuffer(audio_byte, dtype=np.int16)
    speaker_lable_list = []
    for index in range(0, int(len(audio_bufers) / (sample_rate * stride))):
        audio_bufer = audio_bufers[int(index * stride * sample_rate) : int((index+1) * (stride * sample_rate))]
        audio_byte = audio_bufer.tobytes()
        speech_lable = stream_ssd_calss.recognize(audio_byte)
        speaker_lable_list.append(speech_lable)
    speaker = max(set(speaker_lable_list), key=speaker_lable_list.count) if speaker_lable_list != [] else 'non-speech'
    return speaker