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




def is_speech(audio, fs, frame_size, vad):
    def audioSlice(audio, fs, frame_size):
        framesamp = int(frame_size*fs)
        hopsamp = int(frame_size*fs)
        X = np.array([audio[i:i+framesamp] for i in range(0, len(audio)-framesamp, hopsamp)])
        return X
    out = [vad.is_speech(np.int16(fr*2**15).tobytes(), fs) for fr in audioSlice(audio, fs, frame_size)]
    out = np.int16(out)
    counts = np.bincount(out)
    i = np.argmax(counts)
    return bool(i)
