#Copyright 2022 Hamidreza Sadeghi. All rights reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


import numpy as np
import torch
import soundfile as sf
import json
import webrtcvad
from python_speech_features import mfcc
from supervised_speaker_diarization.utils.speech_utils import is_speech
from supervised_speaker_diarization.utils.dvector_utils import load_dvector_and_wav2mel
from supervised_speaker_diarization.config import win_len, win_step_training
 



# def find_non_speech(segs_list, duration):
#     out = []
#     start = 0
#     for seg in segs_list:
#         if seg['label'] == 'SPEECH':
#             if seg['segment']['start'] != start:
#                 out += [(start, seg['segment']['start'])]
#             start = seg['segment']['end'] 
#     if seg['segment']['end'] != duration:
#         out += [(seg['segment']['end'], duration)]
#     return out




def json_to_dict(json_file_path):
    # Opening JSON file
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    return data




def audio_data_preparation(path, wav2mel, dvector, vad, stride = 0.03, win_len = 1.0):
    audio_input, sample_rate = sf.read(path)
    duration = len(audio_input)/ sample_rate
    steps = int(duration//stride)

    num_samples = int(sample_rate*win_len)
    num_new_samples = int(sample_rate*stride)

    index = lambda i: i*num_new_samples

    out = []
    non_speech = []
    for i in range(steps):
        try:
            _sig = audio_input[index(i):index(i) + num_samples]
            is_s = is_speech(_sig, sample_rate, 10/1000, vad)
            vec = torch.tensor(_sig[np.newaxis, :])
            vec = vec.to(torch.float32)
            mel_tensor = wav2mel(vec, sample_rate)
            emb_tensor = dvector.embed_utterance(mel_tensor) 
            if len(_sig) == num_samples:
                if is_s:
                    out += [emb_tensor.cpu().detach().numpy()]
                else:
                    non_speech += [emb_tensor.cpu().detach().numpy()]

        except:
            pass
    

    return out, non_speech




def prepare_train_data(json_file_path):
    users_voices = json_to_dict(json_file_path)
    data = users_voices['data']

    id2label = dict()
    label2id = dict()

    for i in range(len(data)):
        data[i]['id'] = i
        id2label[i] = data[i]['label']
        label2id[data[i]['label']] = i


    classes = [i for i in range(len(data))]
    non_speech = []
    speaker_features = dict()
    wav2mel, dvector = load_dvector_and_wav2mel()
    vad = webrtcvad.Vad(3)


    for i in classes:
        file_path = data[i]['voice_data_path']
        spf, ns = audio_data_preparation(file_path, wav2mel, dvector, vad, stride = win_step_training, win_len = win_len)
        speaker_features[i] = spf
        non_speech += ns

    
    return speaker_features, non_speech, id2label, label2id




def create_input_and_label(speaker_features, non_speech, use_non_speech = False):
    X = []
    y = []
    if use_non_speech:
        for key in speaker_features.keys():
            X += speaker_features[key]
            y += [key]*len(speaker_features[key])
        X = np.array(X + non_speech) 
        y = np.array(y + (key+1)*len(non_speech))

    else:
        for key in speaker_features.keys():
            X += speaker_features[key]
            y += [key]*len(speaker_features[key])
        X = np.array(X)
        y = np.array(y)

    
    return X, y