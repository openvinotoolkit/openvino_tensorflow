# ==============================================================================
#  Copyright 2019-2020 Intel Corporation
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

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.ops import nn_ops
import os
import numpy as np
import ngraph_bridge
import sys

from common import NgraphTest


class TestOpDisableOperations(NgraphTest):
    # Initially nothing is disabled
    def test_disable_op_0(self):
        assert ngraph_bridge.get_disabled_ops() == b''

    # Note it is possible to set an invalid op name (as long as mark_for_clustering is not called)
    @pytest.mark.parametrize(("op_list",), (('Add',), ('Add,Sub',), ('',),
                                            ('_InvalidOp',)))
    def test_disable_op_1(self, op_list):
        ngraph_bridge.set_disabled_ops(op_list)
        assert ngraph_bridge.get_disabled_ops() == op_list.encode("utf-8")
        # Running get_disabled_ops twice to see nothing has changed between 2 consecutive calls
        assert ngraph_bridge.get_disabled_ops() == op_list.encode("utf-8")
        # Clean up
        ngraph_bridge.set_disabled_ops('')

    # Test to see that exception is raised if sess.run is called with invalid op types
    @pytest.mark.parametrize(("invalid_op_list",), (('Add,_InvalidOp',),
                                                    ('_nGraphEncapsulate',)))
    def test_disable_op_2(self, invalid_op_list):
        # This test is disabled for grappler because grappler fails silently and
        # TF continues to run with the unoptimized graph
        # Note, tried setting fail_on_optimizer_errors, but grappler still failed silently
        # TODO: enable this test for grappler as well.
        if (not ngraph_bridge.is_grappler_enabled()):
            ngraph_bridge.set_disabled_ops(invalid_op_list)
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
                ngraph_bridge.set_disabled_ops('')
                return
            assert False, 'Had expected test to raise error'

    def test_disable_op_env(self):
        op_list = 'Select,Where'
        ngraph_bridge.set_disabled_ops(op_list)
        assert ngraph_bridge.get_disabled_ops() == op_list.encode("utf-8")

        env_map = self.store_env_variables('NGRAPH_TF_DISABLED_OPS')
        env_list = 'Squeeze'
        self.set_env_variable('NGRAPH_TF_DISABLED_OPS', env_list)
        assert ngraph_bridge.get_disabled_ops() == env_list.encode("utf-8")
        self.unset_env_variable('NGRAPH_TF_DISABLED_OPS')
        self.restore_env_variables(env_map)

        # Clean up
        ngraph_bridge.set_disabled_ops('')
