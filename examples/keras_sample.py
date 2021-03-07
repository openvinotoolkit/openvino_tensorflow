# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import openvino_tensorflow

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
