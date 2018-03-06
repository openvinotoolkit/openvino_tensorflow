# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np

print("DEVICES: ")

from tensorflow.python import pywrap_tensorflow
pywrap_tensorflow.list_devices()

x = tf.placeholder(tf.float32, shape=(2, 3))
y = tf.placeholder(tf.float32, shape=(2, 3))

print("Python: Creating devices:" )
with tf.device("/device:NGRAPH:0"):
#with tf.device("/device:DYNAMIC_PLUGIN:0"):
#with tf.device("/device:XLA_DYNAMIC_PLUGIN:0"):
#with tf.device("/device:XLA_TEST_PLUGIN:0"):
    a = x + y

    with tf.Session() as sess:
	print("Python: Running with Session" )
        res = sess.run(a, feed_dict={x: np.ones((2, 3)), y: np.ones((2, 3))})
        np.testing.assert_allclose(res, np.ones((2, 3)) * 2.)
        print("result:", res)
