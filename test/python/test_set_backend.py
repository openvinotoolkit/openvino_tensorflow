# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

from __future__ import print_function
import pytest
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import openvino_tensorflow

from common import NgraphTest


# Test openvino_tensorflow config options
class TestSetBackend(NgraphTest):

    def test_set_backend(self):
        # store env variables
        # when testing on backends like GPU the tests are run with OPENVINO_TF_BACKEND
        # by storing and restoring the env_variables we run the tests independent of the backend set
        # currently we store and restore only the OPENVINO_TF_BACKEND
        env_var_map = self.store_env_variables(["OPENVINO_TF_BACKEND"])
        self.unset_env_variable("OPENVINO_TF_BACKEND")

        # test
        openvino_tensorflow.enable()
        backend_cpu = 'CPU'
        backend_interpreter = 'INTERPRETER'

        found_cpu = False
        found_interpreter = False
        # These will only print when running pytest with flag "-s"
        supported_backends = openvino_tensorflow.list_backends()
        print("Number of supported backends ", len(supported_backends))
        print(" ****** Supported Backends ****** ")
        for backend_name in supported_backends:
            print(backend_name)
            if backend_name == backend_cpu:
                found_cpu = True
            if backend_name == backend_interpreter:
                found_interpreter = True
        print(" ******************************** ")
        assert (found_cpu and found_interpreter) == True

        # Create Graph
        val = tf.compat.v1.placeholder(tf.float32)
        out1 = tf.abs(val)
        out2 = tf.abs(out1)

        # set INTERPRETER backend
        openvino_tensorflow.set_backend(backend_interpreter)
        current_backend = openvino_tensorflow.get_backend()
        assert current_backend == backend_interpreter

        # create new session to execute graph
        # If you want to re-confirm which backend the graph was executed
        # currently the only way is to enable OPENVINO_TF_VLOG_LEVEL=5
        with tf.compat.v1.Session() as sess:
            sess.run((out2,), feed_dict={val: ((1.4, -0.5, -1))})
        current_backend = openvino_tensorflow.get_backend()
        assert current_backend == backend_interpreter

        # set CPU backend
        openvino_tensorflow.set_backend(backend_cpu)
        current_backend = openvino_tensorflow.get_backend()
        assert current_backend == backend_cpu
        # create new session to execute graph
        with tf.compat.v1.Session() as sess:
            sess.run((out2,), feed_dict={val: ((1.4, -0.5, -1))})
        current_backend = openvino_tensorflow.get_backend()
        assert current_backend == backend_cpu

        # restore env_variables
        self.restore_env_variables(env_var_map)
