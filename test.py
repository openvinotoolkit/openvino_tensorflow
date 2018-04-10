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
"""nGraph TensorFlow installation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python import pywrap_tensorflow as py_tf
from tensorflow.python.framework import errors_impl

print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

with errors_impl.raise_exception_on_not_ok_status() as status:
    lib = py_tf.TF_LoadLibrary(
        '/home/avijitch/Projects-Mac/ngraph-tensorflow-bridge/build/experiments/libngraph_device.so',
        status)

print("Status: ", py_tf.TF_GetCode(status))

# Get the list of devices
tf_devices = device_lib.list_local_devices()

# Look for nGraph device
for dev in tf_devices:
    print("Device Name: ", dev.name)
    print("Device Type: ", dev.device_type)

# Now try out some computation
x = tf.placeholder(tf.float32, shape=(2, 3))
y = tf.placeholder(tf.float32, shape=(2, 3))

with tf.device("/device:NGRAPH_CPU:0"):
    # Look for nGraph device
    for dev in tf_devices:
        print("Device Name: ", dev.name)
        print("Device Type: ", dev.device_type)

    a = x + y

    config = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=True,
        inter_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        print("Python: Running with Session")
        res = sess.run(a, feed_dict={x: np.ones((2, 3)), y: np.ones((2, 3))})
        np.testing.assert_allclose(res, np.ones((2, 3)) * 2.)
        print("result:", res)
