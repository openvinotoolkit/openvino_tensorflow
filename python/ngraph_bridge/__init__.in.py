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

import importlib
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

__all__ = [
    'enable', 'disable', 'is_enabled', 'backends_len', 'list_backends',
    'set_backend', 'is_supported_backend', 'get_currently_set_backend_name',
    'start_logging_placement', 'stop_logging_placement',
    'is_logging_placement', '__version__'
]

ext = 'dylib' if system() == 'Darwin' else 'so'

TF_VERSION = tf.VERSION
TF_GIT_VERSION = tf.GIT_VERSION
TF_VERSION_NEEDED = "${TensorFlow_VERSION}"
TF_GIT_VERSION_BUILT_WITH = "${TensorFlow_GIT_VERSION}"

# converting version representations to strings if not already
try:
    TF_VERSION = str(TF_VERSION, 'ascii')
except TypeError:  # will happen for python 2 or if already string
    pass

try:
    TF_VERSION_NEEDED = str(TF_VERSION_NEEDED, 'ascii')
except TypeError:
    pass

try:
    if TF_GIT_VERSION.startswith("b'"):  # TF version can be a bytes __repr__()
        TF_GIT_VERSION = eval(TF_GIT_VERSION)
    TF_GIT_VERSION = str(TF_GIT_VERSION, 'ascii')
except TypeError:
    pass

try:
    if TF_GIT_VERSION_BUILT_WITH.startswith("b'"):
        TF_GIT_VERSION_BUILT_WITH = eval(TF_GIT_VERSION_BUILT_WITH)
    TF_GIT_VERSION_BUILT_WITH = str(TF_GIT_VERSION_BUILT_WITH, 'ascii')
except TypeError:
    pass

# print("TensorFlow version installed: {0} ({1})".format(TF_VERSION,
#                                                        TF_GIT_VERSION))
# print("nGraph bridge built with: {0} ({1})".format(TF_VERSION_NEEDED,
#                                                    TF_GIT_VERSION_BUILT_WITH))

# We need to revisit this later. We can automate that using cmake configure
# command.
TF_INSTALLED_VER = TF_VERSION.split('.')
TF_NEEDED_VER = TF_VERSION_NEEDED.split('.')

ngraph_bridge_lib = None
if (TF_INSTALLED_VER[0] == TF_NEEDED_VER[0]) and \
   (TF_INSTALLED_VER[1] == TF_NEEDED_VER[1]) and \
   ((TF_INSTALLED_VER[2].split('-'))[0] == (TF_NEEDED_VER[2].split('-'))[0]):
    libpath = os.path.dirname(__file__)
    ngraph_bridge_lib = ctypes.cdll.LoadLibrary(
        os.path.join(libpath, 'libngraph_bridge.' + ext))
else:
    raise ValueError(
        "Error: Installed TensorFlow version {0}\nnGraph bridge built with: {1}"
        .format(TF_VERSION, TF_VERSION_NEEDED))


def requested():
    return ops.get_default_graph()._attr_scope({
        "_ngraph_requested":
        attr_value_pb2.AttrValue(b=True)
    })


ngraph_bridge_lib.ngraph_is_enabled.restype = ctypes.c_bool
ngraph_bridge_lib.ngraph_list_backends.restype = ctypes.c_bool
ngraph_bridge_lib.ngraph_set_backend.argtypes = [ctypes.c_char_p]
ngraph_bridge_lib.ngraph_set_backend.restype = ctypes.c_bool
ngraph_bridge_lib.ngraph_is_supported_backend.argtypes = [ctypes.c_char_p]
ngraph_bridge_lib.ngraph_is_supported_backend.restype = ctypes.c_bool
ngraph_bridge_lib.ngraph_get_currently_set_backend_name.restype = ctypes.c_bool
ngraph_bridge_lib.ngraph_is_logging_placement.restype = ctypes.c_bool
ngraph_bridge_lib.ngraph_tf_version.restype = ctypes.c_char_p
ngraph_bridge_lib.ngraph_lib_version.restype = ctypes.c_char_p
ngraph_bridge_lib.ngraph_tf_cxx11_abi_flag.restype = ctypes.c_int

try:
    importlib.import_module('plaidml.settings')
    # Importing plaidml.settings -- if it exists -- will have read the
    # user's settings and configured the runtime environment
    # appropriately.
except ImportError:
    pass


def enable():
    ngraph_bridge_lib.ngraph_enable()


def disable():
    ngraph_bridge_lib.ngraph_disable()


def is_enabled():
    return ngraph_bridge_lib.ngraph_is_enabled()


def backends_len():
    return ngraph_bridge_lib.ngraph_backends_len()


def list_backends():
    len_backends = backends_len()
    result = (ctypes.c_char_p * len_backends)()
    if not ngraph_bridge_lib.ngraph_list_backends(result, len_backends):
        raise Exception("Expected " + str(len_backends) +
                        " backends, but got some  other number of backends")
    list_result = list(result)
    # convert bytes to string required for py3 (encode/decode bytes)
    backend_list = []
    for backend in list_result:
        backend_list.append(backend.decode("utf-8"))
    return backend_list


def set_backend(backend):
    if not ngraph_bridge_lib.ngraph_set_backend(backend.encode("utf-8")):
        raise Exception("Backend " + backend + " unavailable.")


def is_supported_backend(backend):
    return ngraph_bridge_lib.ngraph_is_supported_backend(
        backend.encode("utf-8"))


def get_currently_set_backend_name():
    result = (ctypes.c_char_p * 1)()
    if not ngraph_bridge_lib.ngraph_get_currently_set_backend_name(result):
        raise Exception("Cannot get currently set backend")
    list_result = list(result)
    return list_result[0].decode("utf-8")


def start_logging_placement():
    ngraph_bridge_lib.ngraph_start_logging_placement()


def stop_logging_placement():
    ngraph_bridge_lib.ngraph_stop_logging_placement()


def is_logging_placement():
    return ngraph_bridge_lib.ngraph_is_logging_placement()

__version__ = \
  "nGraph bridge version: " + str(ngraph_bridge_lib.ngraph_tf_version()) + "\n" + \
  "nGraph version used for this build: " + str(ngraph_bridge_lib.ngraph_lib_version()) + "\n" + \
  "TensorFlow version used for this build: " + TF_GIT_VERSION_BUILT_WITH + "\n" \
  "CXX11_ABI flag used for this build: " + str(ngraph_bridge_lib.ngraph_tf_cxx11_abi_flag())

