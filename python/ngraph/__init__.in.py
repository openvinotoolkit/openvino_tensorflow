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
from tensorflow.python import pywrap_tensorflow as py_tf
from tensorflow.python.framework import errors_impl

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
 
import ctypes


__all__ = ['enable', 'disable', 'is_enabled', 'backends_len', 'list_backends',
    'set_backend', 'start_logging_placement', 'stop_logging_placement',
    'is_logging_placement']


ext = 'dylib' if system() == 'Darwin' else 'so'
ngraph = None
 

# We need to revisit this later. We can automate that using cmake configure
# command.
if tf.GIT_VERSION == "${TensorFlow_GIT_VERSION}":
    libpath = os.path.dirname(__file__)
    ngraph = ctypes.cdll.LoadLibrary(os.path.join(libpath,
                                                  'libngraph_device.' + ext))
else:
    raise ValueError(
        "Error: Wrong TensorFlow version " + tf.GIT_VERSION +
        "\nNeeded: ${TensorFlow_GIT_VERSION}")

def requested():
    return ops.get_default_graph()._attr_scope(
        {"_ngraph_requested": attr_value_pb2.AttrValue(b=True)})


ngraph.ngraph_is_enabled.restype = ctypes.c_bool
ngraph.ngraph_list_backends.restype = ctypes.c_bool
ngraph.ngraph_set_backend.restype = ctypes.c_bool
ngraph.ngraph_is_logging_placement.restype = ctypes.c_bool


def enable():
  ngraph.ngraph_enable()


def disable():
  ngraph.ngraph_disable()


def is_enabled():
  return ngraph.ngraph_is_enabled()


def backends_len():
  return ngraph.ngraph_backends_len()


def list_backends():
  len_backends = backends_len()
  result = (ctypes.c_string * len_backends)(*(None * len_backends))
  if not ngraph.ngraph_build_backends(result, len_backends):
    raise Exception("Backends fluctuated while listing")
  return result


def set_backend(backend):
  if not ngraph.ngraph_set_backend(backend):
    raise Exception("Backend " + backend + " unavailable.")


def start_logging_placement():
  ngraph.ngraph_start_logging_placement()


def stop_logging_placement():
  ngraph.ngraph_stop_logging_placement()


def is_logging_placement():
  return ngraph.ngraph_is_logging_placement()
