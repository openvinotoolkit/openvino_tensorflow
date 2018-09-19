# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import ngraph

# A simple script to run inference and training on resnet 50

model = ResNet50(weights='imagenet')

batch_size = 128
img = np.random.rand(batch_size, 224, 224, 3)
preds = model.predict(preprocess_input(img))
print('Predicted:', decode_predictions(preds, top=3)[0])
model.compile(tf.keras.optimizers.SGD(), loss='categorical_crossentropy')
preds = model.fit(
    preprocess_input(img), np.zeros((batch_size, 1000), dtype='float32'))
print('Ran a train round')
