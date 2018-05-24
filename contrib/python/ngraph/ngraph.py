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

print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

# Define LD_LIBRARY_PATH indicating where nGraph library is located for now.
# Eventually this won't be needed as the library will be available in either
# the Python site-packages or some other means
# example: export LD_LIBRARY_PATH=/nfs/site/home/langjian/ngraph-tf:$LD_LIBRARY_PATH
import ctypes
if tf.VERSION=='1.8.0' or tf.VERSION=='1.9.0':
   lib = ctypes.cdll.LoadLibrary('libngraph_device.so')

