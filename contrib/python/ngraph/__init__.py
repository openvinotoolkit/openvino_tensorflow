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

import os
import sys
import time
import getpass
from platform import system
   
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python import pywrap_tensorflow as py_tf
from tensorflow.python.framework import errors_impl
 
print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)
 
import ctypes


ext = 'dylib' if system() == 'Darwin' else 'so'
 
# We need to revisit this later. We can automate that using cmake configure command.
if tf.VERSION >= '1.8.0':
   libpath = os.path.dirname(__file__)
   lib = ctypes.cdll.LoadLibrary(os.path.join(libpath,'libngraph_device.'+ext))
else:
   raise ValueError("Error: ngraph-tf requires tensorflow version >= 1.8.0!")

