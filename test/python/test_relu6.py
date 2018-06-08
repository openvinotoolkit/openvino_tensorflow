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
import getpass

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python import pywrap_tensorflow as py_tf
from tensorflow.python.framework import errors_impl

import tfgraphviz as tfg
import ngraph as ng

print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

# Get the list of devices
tf_devices = device_lib.list_local_devices()

# # Look for nGraph device
# for dev in tf_devices:
#     print("Device Name: ", dev.name)
#     print("Device Type: ", dev.device_type)



x = tf.placeholder(tf.float32, shape=(2, 3))
y = tf.placeholder(tf.float32, shape=(2, 3))
z = tf.placeholder(tf.float32, shape=(2, 3))

with tf.device("/device:NGRAPH_CPU:0"):
    #with tf.device("/device:CPU:0"):

    a = x + y + z + x + y + z + x
    c = z * a
    d = tf.nn.relu6(c)
    # d = y - c
    # b = a * y + d

    # Save the graphdef

    config = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=False,
        inter_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        print("Python: Running with Session")
        (result_a, result_c, result_d) = sess.run(
            (a, c, d),
            feed_dict={
                x: np.ones((2, 3)),
                y: np.ones((2, 3)),
                z: np.ones((2, 3)),
            })
        print("result:", result_d)

        # (result_a, result_b, _, _) = sess.run(
        #     (a, b, c, d),
        #     feed_dict={
        #         x: np.ones((2, 3)),
        #         y: np.ones((2, 3)),
        #         z: np.ones((2, 3))
        #     })
        # print("result:", result_b)

