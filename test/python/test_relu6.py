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

x = tf.placeholder(tf.float32, shape=(2, 3))
y = tf.placeholder(tf.float32, shape=(2, 3))
z = tf.placeholder(tf.float32, shape=(2, 3))

with tf.device("/device:NGRAPH:0"):

    a = x + y + z
    b = x + y + z
    c = b * a
    d = tf.nn.relu6(c)

    # input value and expected value
    x_np = np.ones((2, 3))
    y_np = np.ones((2, 3))
    z_np = np.ones((2, 3))
    a_np = x_np + y_np + z_np
    b_np = x_np + y_np + z_np 
    c_np = a_np * b_np
    expected = np.minimum(c_np,np.full(c_np.shape, 6.0)) 

    config = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=False,
        inter_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        print("Python: Running with Session")
        (result_a, result_c, result_d) = sess.run(
            (a, c, d),
            feed_dict={
                x: x_np,
                y: y_np,
                z: z_np,
            })
        print("result:", result_d)
        print("expected:", expected) 
        np.testing.assert_allclose(result_d, expected, atol=1e-5, verbose=True)

    
