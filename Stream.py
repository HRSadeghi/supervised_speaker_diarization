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


from supervised_speaker_diarization.model import build_model, train_model
from supervised_speaker_diarization.config import win_len, win_step_inference
from supervised_speaker_diarization.utils.speech_utils import is_speech
from supervised_speaker_diarization.utils.dvector_utils import load_dvector_and_wav2mel
import numpy as np
import webrtcvad
import soundfile as sf
import torch
import pickle
import os



class Stream_SSD:
  def __init__(self, json_file_path):
    self.json_file_path = json_file_path
    self.wav2mel, self.dvector = load_dvector_and_wav2mel()
    self.vad = webrtcvad.Vad(3)



  def create_and_train_model(self, save_path=''):
    self.clf = build_model()
    self.id2label, self.label2id = train_model(self.clf, self.json_file_path)
    if save_path != '':
      with open(os.path.join(save_path, 'clf_model.pkl'), 'wb') as outp:
        pickle.dump(self.clf, outp, pickle.HIGHEST_PROTOCOL)
      with open(os.path.join(save_path, 'id2label.pkl'), 'wb') as outp:
        pickle.dump(self.id2label, outp, pickle.HIGHEST_PROTOCOL)

  def load_clf_model(self, save_path):
    with open(os.path.join(save_path, 'clf_model.pkl'), 'rb') as inp:
      self.clf = pickle.load(inp)
    with open(os.path.join(save_path, 'id2label.pkl'), 'rb') as inp:
      self.id2label = pickle.load(inp)

  def recognize(self, byte_array, sample_rate = 16000):
    numpydata = np.frombuffer(byte_array, dtype=np.int16) / 2**15


    # if not is_speech(numpydata, sample_rate, 10./1000, vad):
    #   return 'non-speech'
      

    # fus = mfcc(numpydata, sample_rate, win_len, win_step, num_cep, nfft = nfft)
    # pred = self.clf.predict(fus)
    # counts = np.bincount(pred)
    # i = np.argmax(counts)

    # return self.id2label[i]
    if not is_speech(numpydata, sample_rate, 10./1000, self.vad):
      return 'non-speech'
        
      

    vec = torch.tensor(numpydata[np.newaxis, :])
    vec = vec.to(torch.float32)
    if vec.shape[1] != 0:
      try:
        mel_tensor = self.wav2mel(vec, sample_rate)  
        emb_tensor = self.dvector.embed_utterance(mel_tensor)

        fus = emb_tensor.cpu().detach().numpy()[np.newaxis, :]
      except:
        return 'non-speech'

      p = self.clf.predict(fus)[0]
      return self.id2label[p]
    else:
      return 'non-speech'
