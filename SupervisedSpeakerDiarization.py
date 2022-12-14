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




class SupervisedSpeakerDiarization:
  def __init__(self, json_file_path):
    self.json_file_path = json_file_path
    self.wav2mel, self.dvector = load_dvector_and_wav2mel()
    self.vad = webrtcvad.Vad(3)


  def create_and_train_model(self):
    self.clf = build_model()
    self.id2label, self.label2id = train_model(self.clf, self.json_file_path)



  def recognize(self, wav_file_path):
    (sig, rate) = sf.read(wav_file_path)
    duration = len(sig)/ rate
    steps = int(duration//win_step_inference)

    num_samples = int(rate*win_len)
    num_new_samples = int(rate*win_step_inference)


    for i in range(int(duration)):
      _sig = sig[num_samples*i:num_samples*(i+1)]
      print(_sig)
      if not is_speech(_sig, rate, 10./1000, self.vad):
          print('non-speech')
          continue
      
      try:
        vec = torch.tensor(_sig[np.newaxis, :])
        vec = vec.to(torch.float32)
        if vec.shape[1] != 0:
            mel_tensor = self.wav2mel(vec, rate)  
            emb_tensor = self.dvector.embed_utterance(mel_tensor)

            fus = emb_tensor.cpu().detach().numpy()[np.newaxis, :]

            p = self.clf.predict(fus)[0]
            #p2 = clf2.predict(fus)[0]
            #p3 = clf3.predict(fus)[0]
            #p4 = clf4.predict(fus)[0]
            
            #p = most_common([p1, p2, p3, p4])
            print(self.id2label[p])
      except:
        print('non-speech')

    # for j in np.arange(0, duration, stride):
    #     _sig = sig[round(rate*j):round(rate*(j+stride))]
    #     if not is_speech(_sig, rate, 10./1000, vad):
    #         print('non-speech')
    #         continue

    #     fus = mfcc(_sig, rate, win_len, win_step, num_cep, nfft = nfft)
    #     pred = self.clf.predict(fus)
    #     counts = np.bincount(pred)
    #     i = np.argmax(counts)

    #     print(self.id2label[i])