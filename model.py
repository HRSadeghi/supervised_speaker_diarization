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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from supervised_speaker_diarization.utils.preprocess_utils import prepare_train_data, create_input_and_label





def build_model():
  clf = OneVsRestClassifier(make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True, kernel='rbf', C=1)))
  return clf




def train_model(clf, json_file_path):
  speaker_features, non_speech, id2label, label2id = prepare_train_data(json_file_path)
  X, y = create_input_and_label(speaker_features, non_speech, use_non_speech = False)
  clf.fit(X, y)

  return id2label, label2id
