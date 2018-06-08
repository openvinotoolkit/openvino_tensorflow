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

# Define LD_LIBRARY_PATH indicating where nGraph library is located fornow.
# Eventually this won't be needed as the library will be available in either
# the Python site-packages or some other means
import ctypes
lib = ctypes.cdll.LoadLibrary('libngraph_device.so')

with tf.device("/device:CPU:0"):
    x = tf.constant(42,dtype='float32')

with tf.device("/device:NGRAPH:0"):
    first_var = tf.placeholder(tf.float32,shape=())
    y = x + first_var

with tf.device("/device:CPU:0"):
    second_var = tf.placeholder(tf.float32,shape=())
    z = y + second_var

with tf.device("/device:NGRAPH:0"):
    w = z + y

with tf.device("/device:CPU:0"):
    third_var = tf.placeholder(tf.float32,shape=())
    q = w + third_var

config = tf.ConfigProto(
    allow_soft_placement=False,
    log_device_placement=True,
    inter_op_parallelism_threads=1)

with tf.Session(config=config) as sess:
    result = sess.run(q,feed_dict={first_var: 1, second_var: 1, third_var: 1})
    print(result)
