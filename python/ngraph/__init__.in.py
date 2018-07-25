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
 
print("TensorFlow version installed: ", tf.VERSION, " (", tf.GIT_VERSION, ")" )
print("Version needed: ", "${TensorFlow_GIT_VERSION}" ) 
import ctypes

# if Tensorflow already had nGraph bundled in (the upstream candidate)
# then just return
found = False
tf_devices = device_lib.list_local_devices()
for dev in tf_devices:
    if dev.device_type == 'NGRAPH':
        found = True
        break

if not found:
    ext = 'dylib' if system() == 'Darwin' else 'so'
 
    # We need to revisit this later. We can automate that using cmake configure command.
    if tf.GIT_VERSION == "${TensorFlow_GIT_VERSION}":
        libpath = os.path.dirname(__file__)
        lib = ctypes.cdll.LoadLibrary(os.path.join(libpath,'libngraph_device.'+ext))
        print("Module nGraph loaded. Use '/device:NGRAPH:0' as device name")
    else:
        raise ValueError(
            "Error: Wrong TensorFlow version " + tf.GIT_VERSION +
            "\nNeeded: ${TensorFlow_GIT_VERSION}"
        )

