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

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.training import saver
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import nest
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import importer


# This will turn off V1 API related warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if (os.environ.get("OPENVINO_TF_DISABLE_REMAPPING") != "0"):
    tf.config.optimizer.set_experimental_options({'remapping': False})

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
    'is_logging_placement', '__version__', 'cxx11_abi_flag', 'update_config',
    'set_disabled_ops', 'get_disabled_ops',
    'enable_dynamic_fallback', 'disable_dynamic_fallback',
    'export_ir',
]

if system() == 'Darwin':
    ext = 'dylib'
elif system() == 'Windows':
    ext = 'dll'
else:
    ext = 'so'

TF_VERSION = tf.version.VERSION
TF_GIT_VERSION = tf.version.GIT_VERSION
TF_VERSION_NEEDED = "${TensorFlow_VERSION}"
TF_GIT_VERSION_BUILT_WITH = "${TensorFlow_GIT_VERSION}"
TF_MAJOR_VERSION = int(TF_VERSION.split(".")[0])
TF_MINOR_VERSION = int(TF_VERSION.split(".")[1])

rewriter_config = rewriter_config_pb2.RewriterConfig()
rewriter_config.meta_optimizer_iterations = (rewriter_config_pb2.RewriterConfig.ONE)
ovtf_optimizer = rewriter_config.custom_optimizers.add()
ovtf_optimizer.name = "ovtf-optimizer"

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
        full_lib_path = os.path.join(libpath, 'openvino_tensorflow.' + ext)
    else:
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
    openvino_tensorflow_lib.freeBackend.argtypes = []
    openvino_tensorflow_lib.freeBackend.restype = ctypes.c_void_p
    openvino_tensorflow_lib.freeBackendsList.argtypes = []
    openvino_tensorflow_lib.freeBackendsList.restype = ctypes.c_void_p
    openvino_tensorflow_lib.is_logging_placement.restype = ctypes.c_bool
    openvino_tensorflow_lib.tf_version.restype = ctypes.c_char_p
    openvino_tensorflow_lib.version.restype = ctypes.c_char_p
    openvino_tensorflow_lib.openvino_version.restype = ctypes.c_char_p
    openvino_tensorflow_lib.cxx11_abi_flag.restype = ctypes.c_int
    openvino_tensorflow_lib.set_disabled_ops.argtypes = [ctypes.c_char_p]
    openvino_tensorflow_lib.get_disabled_ops.restype = ctypes.c_char_p
    openvino_tensorflow_lib.export_ir.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), 
                                                  ctypes.POINTER(ctypes.c_char_p)]
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
            tf_conversion_extensions_lib_name = "${TF_CONVERSION_EXTENSIONS_LIB_NAME}." + ext
        else:
            tf_conversion_extensions_lib_name = "lib" + "${TF_CONVERSION_EXTENSIONS_LIB_NAME}." + ext
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
                            " backends, but got some other number of backends")
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
        if not openvino_tensorflow_lib.export_ir(output_dir.encode("utf-8"), 
                ctypes.byref(cluster_info), ctypes.byref(err_msg)):
            err_string = err_msg.value.decode("utf-8")
            openvino_tensorflow_lib.freeErrMsg()
            raise Exception("Cannot export IR files: "+err_string)
        cluster_string = cluster_info.value.decode("utf-8")
        openvino_tensorflow_lib.freeClusterInfo()

        return cluster_string
    
    def optimize_graph_with_openvino_tf1(frozen_model_file,
                                        output_node_names,
                                        ):
        """
        Rewrites the tf.Graph of the frozen model with the OpenVINOGrapplerOptimizer.

        Example usage:

        >>> import openvino_tensorflow as ovtf
        >>> pb_file = "inception_v3_2016_08_28_frozen.pb"
        >>> output_names = ['InceptionV3/Predictions/Reshape_1']
        >>> model = ovtf.optimize_graph_with_openvino_tf1(pb_file, output_names)
        >>> with tf.compat.v1.Session() as sess:
              prob_tensor = tf.import_graph_def(model, name='', return_elements=output_names)
              preds = sess.run(prob_tensor, tf_inputs)
        
        Args:
          frozen_model_file: Path to the frozen model file containing the graphdef to optimize
          output_node_names: A list of output node names, which will be used as fetch nodes while 
                             creating the GrapplerItem object

        Raises:
          AssertionError: If the frozen model path is invalid
          AssertionError: If a backend other than CPU is used
        
        Returns:
          The optimized GraphDef
        """

        if not ((TF_MAJOR_VERSION >= 2) and (TF_MINOR_VERSION >= 8)):
            raise AssertionError("Only TF Versions >= 2.8.x are supported for the optimize_graph APIs")

        if not os.path.exists(frozen_model_file):
            raise AssertionError("Could not find frozen model path")
        
        openvino_tensorflow_lib.disable_rewrite_pass()

        if get_backend() != "CPU":
            raise AssertionError(("Offline TF Graph optimization with OpenVINOGrapplerOptimizer "
                                  "is only available for the CPU backend."
                                  "\n Consider removing the call to "
                                  "optimize_graph_with_openvino_tf1 to use OpenVINO"
                                  "on other backends."))

        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()

        with tf.compat.v1.gfile.GFile(frozen_model_file, "rb") as f:
          graph_def.ParseFromString(f.read())
        with graph.as_default():
          importer.import_graph_def(graph_def, name='')
        
        meta_graph_def = saver.export_meta_graph(graph_def=
                                                 graph.as_graph_def(add_shapes=True), graph=graph)

        fetch_collection = meta_graph_pb2.CollectionDef()
        for array in output_node_names:
            fetch_collection.node_list.value.append(array)
        
        # Grappler determines fetch ops from collection 'train_op'.
        meta_graph_def.collection_def[ops.GraphKeys.TRAIN_OP].CopyFrom(
            fetch_collection)

        grappler_session_config = config_pb2.ConfigProto()
        grappler_session_config.graph_options.rewrite_options.CopyFrom(rewriter_config)
        optimized_graph_def = tf_optimizer.OptimizeGraph(grappler_session_config, 
                                                         meta_graph_def, graph_id=b"tf_graph")

        return optimized_graph_def
    
    def optimize_graph_with_openvino_tf2(saved_model_dir,
                                        input_tensors=None,
                                        saved_model_signature=
                                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                                        saved_model_tag=tag_constants.SERVING,
                                        save_optimized_function_signature=False
                                        ):
        """
        Rewrites the tf.Graph of a TF2 SavedModel Function Signature with the 
        OpenVINOGrapplerOptimizer. Expects a sample input tensor with a fully defined shape and 
        dtype, which will be used to create the input feeds of GrapplerItem used for CostAnalysis.

        Converts all Variable ops into Const ops, and inlines supported compute heavy subgraphs 
        as encapsulated OpenVINO custom ops. Returns a single ConcreteFunction specialized to 
        input shape and dtype of the provided 'input_tensor'.

        Example usage:

        >>> import openvino_tensorflow as ovtf
        >>> model_path = "ssd_resnet101_v1_fpn_1024x1024"
        >>> image_numpy = np.array(np.random.rand(1, 1024,1024,3)).astype(np.uint8)
        >>> input_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.uint8)
        >>> model = ovtf.optimize_graph_with_openvino_tf2(model_path, input_tensor)
        >>> print(model)
        <ConcreteFunction pruned(args_0) at 0x>
        >>> results = model(input_tensor)
        
        Args:
          saved_model_dir: The SavedModel directory to load from.
          input_tensors: A tf.Tensor, a list or a dict of tf.Tensor or numpy arrays, whose shape and
            type will be used by OpenVINOGrapplerOptimizer for cost analysis. 
          saved_model_signature: SavedModel tag to load
          saved_model_tag: The SavedModel function signature key, whose graph will be optimized
          save_optimized_function_signature: Whether to save the new optimized function signature to
            the model at 'saved_model_dir'

        Raises:
          AssertionError: If the SavedModel path is invalid
          AssertionError: If a backend other than CPU is used

        Returns:
          The optimized TF ConcreteFunction object
        """

        #[TODO] Add support for taking direct tf.Graph or tf.function inputs
        
        if not ((TF_MAJOR_VERSION >= 2) and (TF_MINOR_VERSION >= 8)):
            raise AssertionError("Only TF Versions >= 2.8.x are supported for the optimize_graph APIs")

        if not os.path.exists(saved_model_dir):
          raise AssertionError("Could not find saved model path")

        if get_backend() != "CPU":
          raise AssertionError(("Offline TF Graph optimization with OpenVINOGrapplerOptimizer "
                                  "is only available for the CPU backend."
                                  "\n Consider removing the call to "
                                  "optimize_graph_with_openvino_tf2 to use OpenVINO"
                                  "on other backends."))
        
        openvino_tensorflow_lib.disable_rewrite_pass()

        # prepare tf function from saved_model
        # Load model with provided saved model tag
        try:
          # Try the provided tag or the default tag
          saved_model = load.load(saved_model_dir, saved_model_tag)
        except RuntimeError as e:
          # Catch RuntimeError if failed to load tag
          # Try skipping tag if the SavedModel contains a single MetaGraph, 
          # as for those exported from `tf.saved_model.save`.
          if saved_model_tag == tag_constants.SERVING:
              saved_model = load.load(saved_model_dir)
          else:
              raise RuntimeError(e)

        # form a concrete function with input tensor in it so grappler can do shape inference
        # Select desired saved model function signature
        try:
          # try the provided signature or the default signature
          print("Available Saved Model Signatures: ", saved_model.signatures)
          print("Selecting Signature: ", saved_model_signature)
            
          func = tf.function(saved_model.signatures[saved_model_signature])
          
        except KeyError as e:
          # If the provided signature doesn't work, 
          # let tf.function try inferring available signatures
          # If `None`, a separate function is instantiated for each inferred input signature
          if saved_model_signature == signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              func = tf.function(saved_model)
          else:
              raise RuntimeError(e)

        # Handle all types of possible input tensors
        if isinstance(input_tensors, dict):
          tensors = {name:(ops.convert_to_tensor(v) if not isinstance(v, tf.Tensor) else v) 
                     for name, v in input_tensors.items()}
          func = tf.function(func)
          args, kwargs = [], tensors
        elif isinstance(input_tensors, list):
          tensors = [ops.convert_to_tensor(v) if not isinstance(v, tf.Tensor) else v 
                     for v in input_tensors]
          input_signature = [tf.TensorSpec.from_tensor(v) for v in tensors]
          func = tf.function(func, input_signature=input_signature)
          args, kwargs = [], {}
        else:
          if not isinstance(input_tensors, tf.Tensor):
            tensors = ops.convert_to_tensor(input_tensors) 
          else:
            tensors = input_tensors
          input_signature = [tf.TensorSpec.from_tensor(tensors)]
          func = tf.function(func, input_signature=input_signature)
          args, kwargs = [], {}
        
        func = func.get_concrete_function(*args, **kwargs)
        
        # Converting var2consts for larger models might take a long time
        frozen_func = convert_to_constants.convert_variables_to_constants_v2(func, 
                                                lower_control_flow=False, aggressive_inlining=True)
        
        meta_graph_def = saver.export_meta_graph(graph_def=
                                                 frozen_func.graph.as_graph_def(add_shapes=True), 
                                                 graph=frozen_func.graph)

        fetch_collection = meta_graph_pb2.CollectionDef()
        for array in frozen_func.outputs:
          fetch_collection.node_list.value.append(array.name)
        
        # Grappler determines fetch ops from collection 'train_op'.
        meta_graph_def.collection_def[ops.GraphKeys.TRAIN_OP].CopyFrom(
            fetch_collection)

        grappler_session_config = config_pb2.ConfigProto()
        grappler_session_config.graph_options.rewrite_options.CopyFrom(rewriter_config)
        optimized_graph_def = tf_optimizer.OptimizeGraph(grappler_session_config, 
                                                         meta_graph_def, graph_id=b"tf_graph")
        
        # Swap original function with optimized function in TF's context
        for f in optimized_graph_def.library.function:
          while context.context().has_function(f.signature.name):
              context.context().remove_function(f.signature.name)

        optimized_func = wrap_function.function_from_graph_def(
            optimized_graph_def,
            [tensor.name for tensor in frozen_func.inputs],
            [tensor.name for tensor in frozen_func.outputs])

        optimized_func.graph.structured_outputs = nest.pack_sequence_as(
            func.graph.structured_outputs,
            optimized_func.graph.structured_outputs)

        optimized_func.graph.structured_input_signature = (
            func.structured_input_signature)

        # Rewrite the signature map using the optimized ConcreteFunction.
        signatures = {
            key: value for key, value in saved_model.signatures.items()
        }
        signatures["ovtf"] = optimized_func

        # Save the optimized function for later use
        # Sometimes this is useful when start-up overheads from this function call 
        # needs to be avoided
        if save_optimized_function_signature:
          save.save(saved_model, saved_model_dir)
          return optimized_func
        else:
          return optimized_func



    def convert_variables_to_constants_with_openvino_tf2(saved_model_dir,
                                                         input_tensors=None,
                                                         saved_model_signature=
                                                         signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                                                         saved_model_tag=tag_constants.SERVING
                                                         ):
        """
        Converts all Variable ops into Const ops, and inlines supported compute heavy subgraphs 
        as encapsulated OpenVINO custom ops. Returns a single ConcreteFunction specialized to 
        input shape and dtype of the provided 'input_tensor'.

        Example usage:

        >>> import openvino_tensorflow as ovtf
        >>> model_path = "ssd_resnet101_v1_fpn_1024x1024"
        >>> image_numpy = np.array(np.random.rand(1, 1024,1024,3)).astype(np.uint8)
        >>> input_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.uint8)
        >>> model = ovtf.convert_variables_to_constants_with_openvino_tf2(model_path, input_tensor)
        >>> print(model)
        <ConcreteFunction pruned(args_0) at 0x>
        >>> results = model(input_tensor)
        
        Args:
          saved_model_dir: The SavedModel directory to load from.
          input_tensors: A tf.Tensor, a list or a dict of tf.Tensor or numpy arrays, whose shape and
            type will be used by OpenVINOGrapplerOptimizer for cost analysis. 
          saved_model_signature: SavedModel tag to load
          saved_model_tag: The SavedModel function signature key, whose graph will be optimized

        Raises:
          AssertionError: If the SavedModel path is invalid
          AssertionError: If a backend other than CPU is used

        Returns:
          The optimized TF ConcreteFunction object
        """

        #[TODO] Add support for taking direct tf.Graph or tf.function inputs
        
        if not ((TF_MAJOR_VERSION >= 2) and (TF_MINOR_VERSION >= 8)):
            raise AssertionError("Only TF Versions >= 2.8.x are supported for the optimize_graph APIs")

        if not os.path.exists(saved_model_dir):
          raise AssertionError("Could not find saved model path")

        if get_backend() != "CPU":
          raise AssertionError(("Offline TF Graph optimization with OpenVINOGrapplerOptimizer "
                                  "is only available for the CPU backend."
                                  "\n Consider removing the call to "
                                  "optimize_graph_with_openvino_tf2 to use OpenVINO"
                                  "on other backends."))
        
        # prepare tf function from saved_model
        # Load model with provided saved model tag
        try:
          # Try the provided tag or the default tag
          saved_model = load.load(saved_model_dir, saved_model_tag)
        except RuntimeError as e:
          # Catch RuntimeError if failed to load tag
          # Try skipping tag if the SavedModel contains a single MetaGraph, 
          # as for those exported from `tf.saved_model.save`.
          if saved_model_tag == tag_constants.SERVING:
              saved_model = load.load(saved_model_dir)
          else:
              raise RuntimeError(e)

        # form a concrete function with input tensor in it so grappler can do shape inference
        # Select desired saved model function signature
        try:
          # try the provided signature or the default signature
          print("Available Saved Model Signatures: ", saved_model.signatures)
          print("Selecting Signature: ", saved_model_signature)
            
          func = tf.function(saved_model.signatures[saved_model_signature])
          
        except KeyError as e:
          # If the provided signature doesn't work, 
          # let tf.function try inferring available signatures
          # If `None`, a separate function is instantiated for each inferred input signature
          if saved_model_signature == signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              func = tf.function(saved_model)
          else:
              raise RuntimeError(e)

        # Handle all types of possible input tensors
        if isinstance(input_tensors, dict):
          tensors = {name:(ops.convert_to_tensor(v) if not isinstance(v, tf.Tensor) else v) 
                     for name, v in input_tensors.items()}
          func = tf.function(func)
          args, kwargs = [], tensors
        elif isinstance(input_tensors, list):
          tensors = [ops.convert_to_tensor(v) if not isinstance(v, tf.Tensor) else v 
                     for v in input_tensors]
          input_signature = [tf.TensorSpec.from_tensor(v) for v in tensors]
          func = tf.function(func, input_signature=input_signature)
          args, kwargs = [], {}
        else:
          if not isinstance(input_tensors, tf.Tensor):
            tensors = ops.convert_to_tensor(input_tensors) 
          else:
            tensors = input_tensors
          input_signature = [tf.TensorSpec.from_tensor(tensors)]
          func = tf.function(func, input_signature=input_signature)
          args, kwargs = [], {}
        
        func = func.get_concrete_function(*args, **kwargs)
        
        # Converting var2consts for larger models might take a long time
        frozen_func = convert_to_constants.convert_variables_to_constants_v2(func, 
                                                lower_control_flow=False, aggressive_inlining=True)

        return frozen_func


    __version__ = \
    "OpenVINO integration with TensorFlow version: " + str(openvino_tensorflow_lib.version()) \
    + "\n" + \
    "OpenVINO version used for this build: " + str(openvino_tensorflow_lib.openvino_version()) \
    + "\n" + \
    "TensorFlow version used for this build: " + "v" + TF_VERSION_NEEDED \
    + "\n" \
    "CXX11_ABI flag used for this build: " + str(openvino_tensorflow_lib.cxx11_abi_flag()) + "\n"

    def prepare_model_with_session(session, graph, input_names, output_names, input_shapes):
        in_dict = {}
        shape_idx = 0
        for in_name in input_names:
            input_operation = graph.get_operation_by_name(in_name)
            r = np.ndarray(input_shapes[shape_idx], dtype=input_operation.outputs[0].dtype.as_numpy_dtype)
            in_dict[input_operation.outputs[0]] = r
            shape_idx += 1
        out_list = []
        for out_name in output_names:
            output_operation = graph.get_operation_by_name(out_name)
            out_list.append(output_operation.outputs[0])
        num_iter = 1
        if (get_backend()=="CPU" and os.environ.get("OPENVINO_TF_DYNAMIC_FALLBACK") != "0" and os.environ.get("OPENVINO_TF_DISABLE_COST_ASSIGNMENT") != "1"):
            num_iter=3
        for i in range(num_iter):
            results = session.run(out_list, in_dict)


    def prepare_model(model, shape, data_type):
        t = tf.random.uniform(
            shape,
            minval=0,
            maxval=None,
            dtype=data_type,
            seed=None,
            name=None)
        num_iter = 1
        if (get_backend()=="CPU" and os.environ.get("OPENVINO_TF_DYNAMIC_FALLBACK") != "0" and os.environ.get("OPENVINO_TF_DISABLE_COST_ASSIGNMENT") != "1"):
            num_iter=3
        for i in range(num_iter):
            model(t)
