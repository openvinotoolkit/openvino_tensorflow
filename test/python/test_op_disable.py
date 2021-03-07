# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.ops import nn_ops
import os
import numpy as np
import openvino_tensorflow
import sys

from common import NgraphTest


class TestOpDisableOperations(NgraphTest):
    # Initially nothing is disabled
    def test_disable_op_0(self):
        assert openvino_tensorflow.get_disabled_ops() == b''

    # Note it is possible to set an invalid op name (as long as mark_for_clustering is not called)
    @pytest.mark.parametrize(("op_list",), (('Add',), ('Add,Sub',), ('',),
                                            ('_InvalidOp',)))
    def test_disable_op_1(self, op_list):
        openvino_tensorflow.set_disabled_ops(op_list)
        assert openvino_tensorflow.get_disabled_ops() == op_list.encode("utf-8")
        # Running get_disabled_ops twice to see nothing has changed between 2 consecutive calls
        assert openvino_tensorflow.get_disabled_ops() == op_list.encode("utf-8")
        # Clean up
        openvino_tensorflow.set_disabled_ops('')

    # Test to see that exception is raised if sess.run is called with invalid op types
    @pytest.mark.parametrize(("invalid_op_list",), (('Add,_InvalidOp',),
                                                    ('_nGraphEncapsulate',)))
    def test_disable_op_2(self, invalid_op_list):
        # This test is disabled for grappler because grappler fails silently and
        # TF continues to run with the unoptimized graph
        # Note, tried setting fail_on_optimizer_errors, but grappler still failed silently
        # TODO: enable this test for grappler as well.
        if (not openvino_tensorflow.is_grappler_enabled()):
            openvino_tensorflow.set_disabled_ops(invalid_op_list)
            a = tf.compat.v1.placeholder(tf.int32, shape=(5,))
            b = tf.constant(np.ones((5,)), dtype=tf.int32)
            c = a + b

            def run_test(sess):
                return sess.run(c, feed_dict={a: np.ones((5,))})

            assert (self.without_ngraph(run_test) == np.ones(5,) * 2).all()
            #import pdb; pdb.set_trace()
            try:
                # This test is expected to fail,
                # since all the strings passed to set_disabled_ops have invalid ops in them
                res = self.with_ngraph(run_test)
            except:
                # Clean up
                openvino_tensorflow.set_disabled_ops('')
                return
            assert False, 'Had expected test to raise error'

    def test_disable_op_env(self):
        op_list = 'Select,Where'
        openvino_tensorflow.set_disabled_ops(op_list)
        assert openvino_tensorflow.get_disabled_ops() == op_list.encode("utf-8")

        env_map = self.store_env_variables('OPENVINO_TF_DISABLED_OPS')
        env_list = 'Squeeze'
        self.set_env_variable('OPENVINO_TF_DISABLED_OPS', env_list)
        assert openvino_tensorflow.get_disabled_ops() == env_list.encode("utf-8")
        self.unset_env_variable('OPENVINO_TF_DISABLED_OPS')
        self.restore_env_variables(env_map)

        # Clean up
        openvino_tensorflow.set_disabled_ops('')
