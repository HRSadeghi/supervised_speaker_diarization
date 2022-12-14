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



# Preprocessing setup
win_len = 1.0
win_step_inference = 0.3 # 0.3s
win_step_training = 0.03 # 0.03s



# Pyannote hyper parameters
HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}



# dvector setup


wav2mel_path = "/home/sadeghi/Desktop/supervised_speaker_diarization/models/wav2mel.pt"
dvector_path = "/home/sadeghi/Desktop/supervised_speaker_diarization/models/dvector.pt"




