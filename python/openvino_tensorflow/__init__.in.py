# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
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
tf.compat.v1.disable_eager_execution()

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import load_library

# This will turn off V1 API related warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import ctypes

__all__ = [
    'enable', 'disable', 'is_enabled', 'list_backends',
    'set_backend', 'get_backend',
    'start_logging_placement', 'stop_logging_placement',
    'is_logging_placement', '__version__', 'cxx11_abi_flag'
    'is_grappler_enabled', 'update_config',
    'set_disabled_ops', 'get_disabled_ops',
]

ext = 'dylib' if system() == 'Darwin' else 'so'

TF_VERSION = tf.version.VERSION
TF_GIT_VERSION = tf.version.GIT_VERSION
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
# print("Openvino_Tensorflow built with: {0} ({1})".format(TF_VERSION_NEEDED,
#                                                    TF_GIT_VERSION_BUILT_WITH))

# We need to revisit this later. We can automate that using cmake configure
# command.
TF_INSTALLED_VER = TF_VERSION.split('.')
TF_NEEDED_VER = TF_VERSION_NEEDED.split('.')

ovtf_classic_loaded = True
openvino_tensorflow_lib = None
if (TF_INSTALLED_VER[0] == TF_NEEDED_VER[0]) and \
   (TF_INSTALLED_VER[1] == TF_NEEDED_VER[1]) and \
   ((TF_INSTALLED_VER[2].split('-'))[0] == (TF_NEEDED_VER[2].split('-'))[0]):
    libpath = os.path.dirname(__file__)
    full_lib_path = os.path.join(libpath, 'libopenvino_tensorflow.' + ext)
    _ = load_library.load_op_library(full_lib_path)
    openvino_tensorflow_lib = ctypes.cdll.LoadLibrary(full_lib_path)
else:
    raise ValueError(
        "Error: Installed TensorFlow version {0}\n openvino_tensorflow built with: {1}"
        .format(TF_VERSION, TF_VERSION_NEEDED))

def requested():
    return ops.get_default_graph()._attr_scope({
        "_ovtf_requested":
        attr_value_pb2.AttrValue(b=True)
    })

if ovtf_classic_loaded:
    openvino_tensorflow_lib.is_enabled.restype = ctypes.c_bool
    openvino_tensorflow_lib.list_backends.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
    openvino_tensorflow_lib.list_backends.restype = ctypes.c_bool
    openvino_tensorflow_lib.set_backend.argtypes = [ctypes.c_char_p]
    openvino_tensorflow_lib.set_backend.restype = ctypes.c_bool
    openvino_tensorflow_lib.get_backend.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
    openvino_tensorflow_lib.get_backend.restype = ctypes.c_bool
    openvino_tensorflow_lib.is_logging_placement.restype = ctypes.c_bool
    openvino_tensorflow_lib.tf_version.restype = ctypes.c_char_p
    openvino_tensorflow_lib.ngraph_version.restype = ctypes.c_char_p
    openvino_tensorflow_lib.cxx11_abi_flag.restype = ctypes.c_int
    openvino_tensorflow_lib.is_grappler_enabled.restype = ctypes.c_bool
    openvino_tensorflow_lib.set_disabled_ops.argtypes = [ctypes.c_char_p]
    openvino_tensorflow_lib.get_disabled_ops.restype = ctypes.c_char_p

    def enable():
        openvino_tensorflow_lib.enable()

    def disable():
        openvino_tensorflow_lib.disable()

    def is_enabled():
        return openvino_tensorflow_lib.is_enabled()

    def list_backends():
        len_backends = openvino_tensorflow_lib.backends_len()
        result = (ctypes.c_char_p * len_backends)()
        if not openvino_tensorflow_lib.list_backends(result):
            raise Exception("Expected " + str(len_backends) +
                            " backends, but got some  other number of backends")
        list_result = list(result)
        # convert bytes to string required for py3 (encode/decode bytes)
        backend_list = []
        for backend in list_result:
            backend_list.append(backend.decode("utf-8"))
        return backend_list

    def set_backend(backend):
        if not openvino_tensorflow_lib.set_backend(backend.encode("utf-8")):
            raise Exception("Backend " + backend + " unavailable.")

    def get_backend():
        result = ctypes.c_char_p()
        if not openvino_tensorflow_lib.get_backend(ctypes.byref(result)):
            raise Exception("Cannot get currently set backend")
        return result.value.decode("utf-8")

    def start_logging_placement():
        openvino_tensorflow_lib.start_logging_placement()

    def stop_logging_placement():
        openvino_tensorflow_lib.stop_logging_placement()

    def is_logging_placement():
        return openvino_tensorflow_lib.is_logging_placement()

    def cxx11_abi_flag():
        return openvino_tensorflow_lib.cxx11_abi_flag()

    def is_grappler_enabled():
        return openvino_tensorflow_lib.is_grappler_enabled()

    def update_config(config, backend_name = "CPU", device_id = ""):
        #updating session config if grappler is enabled
        if(openvino_tensorflow_lib.is_grappler_enabled()):
            opt_name = 'ovtf-optimizer'
            # If the config already has ovtf-optimizer, then do not update it
            if config.HasField('graph_options'):
                if config.graph_options.HasField('rewrite_options'):
                    custom_opts = config.graph_options.rewrite_options.custom_optimizers
                    for i in range(len(custom_opts)):
                        if custom_opts[i].name == opt_name:
                            return config
            rewriter_options = rewriter_config_pb2.RewriterConfig()
            rewriter_options.meta_optimizer_iterations=(rewriter_config_pb2.RewriterConfig.ONE)
            rewriter_options.min_graph_nodes=-1
            ovtf_optimizer = rewriter_options.custom_optimizers.add()
            ovtf_optimizer.name = opt_name
            ovtf_optimizer.parameter_map["device_id"].s = device_id.encode()
            config.MergeFrom(tf.compat.v1.ConfigProto(graph_options=tf.compat.v1.GraphOptions(rewrite_options=rewriter_options)))
            # For reference, if we want to provide configuration support(backend parameters)
            # in a python script using the ovtf-optimizer
            # rewriter_options = rewriter_config_pb2.RewriterConfig()
            # rewriter_options.meta_optimizer_iterations=(rewriter_config_pb2.RewriterConfig.ONE)
            # rewriter_options.min_graph_nodes=-1
            # ovtf_optimizer = rewriter_options.custom_optimizers.add()
            # ovtf_optimizer.name = "ovtf-optimizer"
            # ovtf_optimizer.parameter_map["device_id"].s = device_id.encode()
            # ovtf_optimizer.parameter_map["max_batch_size"].s = b'64'
            # ovtf_optimizer.parameter_map["ice_cores"].s = b'12'
            # config.MergeFrom(tf.compat.v1.ConfigProto(graph_options=tf.compat.v1.GraphOptions(rewrite_options=rewriter_options)))
        return config

    def set_disabled_ops(unsupported_ops):
        openvino_tensorflow_lib.set_disabled_ops(unsupported_ops.encode("utf-8"))

    def get_disabled_ops():
        return openvino_tensorflow_lib.get_disabled_ops()

    __version__ = \
    "openvino_tensorflow version: " + str(OVTF_VERSION) + "\n" + \
    "nGraph version used for this build: " + str(openvino_tensorflow_lib.ngraph_version()) + "\n" + \
    "TensorFlow version used for this build: " + TF_GIT_VERSION_BUILT_WITH + "\n" \
    "CXX11_ABI flag used for this build: " + str(openvino_tensorflow_lib.cxx11_abi_flag()) + "\n" \
    "openvino_tensorflow built with Grappler: " + str(openvino_tensorflow_lib.is_grappler_enabled()) + "\n" \
