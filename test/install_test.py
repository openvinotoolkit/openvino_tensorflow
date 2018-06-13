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

import tensorflow as tf
from tensorflow.python.client import device_lib

import ctypes
lib = ctypes.cdll.LoadLibrary('libngraph_device.so')

def check_for_ngraph_device():
    # Get the list of devices
    tf_devices = device_lib.list_local_devices()

    found = False
    name = None
    # Look for nGraph device
    for dev in tf_devices:
        if dev.device_type == 'NGRAPH':
            name = dev.name
            found = True
            break
    return found, name


if __name__ == '__main__':
    print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

    [found, name] = check_for_ngraph_device()
    if not found:
        print("nGraph Device is not available")
        sys.exit(1)
    else:
        print("Device nGraph available")
        print("Name: ", name)
        sys.exit(0)
