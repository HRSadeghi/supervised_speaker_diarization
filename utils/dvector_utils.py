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


import torch
import numpy as np
import torchaudio
import soundfile as sf
from supervised_speaker_diarization.config import wav2mel_path, dvector_path




def load_dvector_and_wav2mel():
    wav2mel = torch.jit.load(wav2mel_path).eval()
    dvector = torch.jit.load(dvector_path).eval()
    return wav2mel, dvector