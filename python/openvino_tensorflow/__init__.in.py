# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import sys
import ast
import time
import getpass
from platform import system

import numpy as np
import tensorflow as tf

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import load_library

# This will turn off V1 API related warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import ctypes

cuda_visible_devices = ""
if (os.environ.get("OPENVINO_TF_DISABLE") != "1"):
    if ("CUDA_VISIBLE_DEVICES" in os.environ):
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

__all__ = [
    'enable', 'disable', 'is_enabled', 'list_backends',
    'set_backend', 'get_backend',
    'start_logging_placement', 'stop_logging_placement',
    'is_logging_placement', '__version__', 'cxx11_abi_flag'
    'is_grappler_enabled', 'update_config',
    'set_disabled_ops', 'get_disabled_ops',
    'enable_dynamic_fallback', 'disable_dynamic_fallback',
    'export_ir',
]

if system() == 'Darwin':
    ext = '.dylib'
elif system() == 'Windows':
    ext = '.dll'
else:
    ext = '.so'

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
        TF_GIT_VERSION = ast.literal_eval(TF_GIT_VERSION)
    TF_GIT_VERSION = str(TF_GIT_VERSION, 'ascii')
except TypeError:
    pass

try:
    if TF_GIT_VERSION_BUILT_WITH.startswith("b'"):
        TF_GIT_VERSION_BUILT_WITH = ast.literal_eval(TF_GIT_VERSION_BUILT_WITH)
    TF_GIT_VERSION_BUILT_WITH = str(TF_GIT_VERSION_BUILT_WITH, 'ascii')
except TypeError:
    pass

# We need to revisit this later. We can automate that using cmake configure
# command.
TF_INSTALLED_VER = TF_VERSION.split('.')
TF_NEEDED_VER = TF_VERSION_NEEDED.split('.')

ovtf_classic_loaded = True
openvino_tensorflow_lib = None
if (TF_INSTALLED_VER[0] == TF_NEEDED_VER[0]) and \
   (TF_INSTALLED_VER[1] == TF_NEEDED_VER[1]):
    libpath = os.path.dirname(__file__)
    if system() == 'Windows':
        full_lib_path = os.path.join(libpath, 'openvino_tensorflow' + ext)
    else:
      full_lib_path = os.path.join(libpath, 'libopenvino_tensorflow' + ext)
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
    openvino_tensorflow_lib.freeBackend.argtypes = []
    openvino_tensorflow_lib.freeBackend.restype = ctypes.c_void_p
    openvino_tensorflow_lib.freeBackendsList.argtypes = []
    openvino_tensorflow_lib.freeBackendsList.restype = ctypes.c_void_p
    openvino_tensorflow_lib.is_logging_placement.restype = ctypes.c_bool
    openvino_tensorflow_lib.tf_version.restype = ctypes.c_char_p
    openvino_tensorflow_lib.version.restype = ctypes.c_char_p
    openvino_tensorflow_lib.openvino_version.restype = ctypes.c_char_p
    openvino_tensorflow_lib.cxx11_abi_flag.restype = ctypes.c_int
    openvino_tensorflow_lib.is_grappler_enabled.restype = ctypes.c_bool
    openvino_tensorflow_lib.set_disabled_ops.argtypes = [ctypes.c_char_p]
    openvino_tensorflow_lib.get_disabled_ops.restype = ctypes.c_char_p
    openvino_tensorflow_lib.export_ir.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
    openvino_tensorflow_lib.export_ir.restype = ctypes.c_bool
    openvino_tensorflow_lib.freeClusterInfo.argtypes = []
    openvino_tensorflow_lib.freeClusterInfo.restype = ctypes.c_void_p
    openvino_tensorflow_lib.freeErrMsg.argtypes = []
    openvino_tensorflow_lib.freeErrMsg.restype = ctypes.c_void_p
    openvino_tensorflow_lib.load_tf_conversion_extensions.argtypes = [ctypes.c_char_p]
    
    def load_tf_conversion_extensions():
        import importlib
        lib_dir = os.path.dirname(importlib.util.find_spec("openvino_tensorflow").origin)
        if system() == "Windows":
            tf_conversion_extensions_lib_name = "${TF_CONVERSION_EXTENSIONS_LIB_NAME}" + ext
        else:
            tf_conversion_extensions_lib_name = "lib" + "${TF_CONVERSION_EXTENSIONS_LIB_NAME}" + ext
        tf_conversion_extensions_so_path = os.path.join(lib_dir, tf_conversion_extensions_lib_name)
        openvino_tensorflow_lib.load_tf_conversion_extensions(tf_conversion_extensions_so_path.encode("utf-8"))
    
    load_tf_conversion_extensions()

    def enable():
        openvino_tensorflow_lib.enable()
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def disable():
        openvino_tensorflow_lib.disable()
        if ("CUDA_VISIBLE_DEVICES" in os.environ):
            if (len(cuda_visible_devices) == 0):
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

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
        openvino_tensorflow_lib.freeBackendsList()
        return backend_list

    def set_backend(backend):
        if not openvino_tensorflow_lib.set_backend(backend.encode("utf-8")):
          raise Exception("Backend " + backend + " unavailable.")

    def get_backend():
        result = ctypes.c_char_p()
        if not openvino_tensorflow_lib.get_backend(ctypes.byref(result)):
            raise Exception("Cannot get currently set backend")
        backend_name = result.value.decode("utf-8")
        openvino_tensorflow_lib.freeBackend()
        return backend_name

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

    def enable_dynamic_fallback():
        openvino_tensorflow_lib.enable_dynamic_fallback()

    def disable_dynamic_fallback():
        openvino_tensorflow_lib.disable_dynamic_fallback()

    def export_ir(output_dir):
        cluster_info = ctypes.c_char_p()
        err_msg = ctypes.c_char_p()
        if not openvino_tensorflow_lib.export_ir(output_dir.encode("utf-8"), ctypes.byref(cluster_info), ctypes.byref(err_msg)):
            err_string = err_msg.value.decode("utf-8")
            openvino_tensorflow_lib.freeErrMsg()
            raise Exception("Cannot export IR files: "+err_string)
        cluster_string = cluster_info.value.decode("utf-8")
        openvino_tensorflow_lib.freeClusterInfo()

        return cluster_string

    __version__ = \
    "OpenVINO integration with TensorFlow version: " + str(openvino_tensorflow_lib.version()) + "\n" + \
    "OpenVINO version used for this build: " + str(openvino_tensorflow_lib.openvino_version()) + "\n" + \
    "TensorFlow version used for this build: " + "v" + TF_VERSION_NEEDED + "\n" \
    "CXX11_ABI flag used for this build: " + str(openvino_tensorflow_lib.cxx11_abi_flag()) + "\n"
