# ==============================================================================
#  Copyright 2019 Intel Corporation
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
"""nGraph TensorFlow variable_update + static input
Var 
| \ 
|   \*
|   Encap 
|   /
Assign (or removed)
* The input to Encap is a static input from Variable
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import ctypes

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import json

import ngraph_bridge
import os

# If the below graph is run for many iterations
# The NGraphVar's NGTensor is updated every iteration
# NGraphVar's TFTensor is not updated as no TF Node needs it
# However StaticInputs are derived from the input TF Tensor (which is stale)
# giving functionally incorrect results
# TF MeanOp expects a static input
# * The input to Encap is a static input from Variable
#
#    Const     NGraphVar     Const
#      \       /   |   \     /
#      _\|  *|/_   |   _\| |/_
#         Mean     |    Add
#                  |     /
#                 \|/  |/_
#                NGraphAssign
#
# After Encapsulation
#
#            NGraphVar
#             /   |   \
#          *|/_   |   _\|
#     NGEncap1    |   NGEncap2
#                 |     /
#                \|/  |/_
#             NGraphAssign
#

from common import NgraphTest


class TestVariableStaticInputs(NgraphTest):

    def __run_test(self, sess):
        # Var is initialized by var_init
        var = tf.get_variable('var', [1], dtype=tf.int32)
        var_init = tf.constant([0])
        var_initialize = var.assign(var_init)

        # Computation of mean
        input1 = tf.constant(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]], name='input1')
        mean = tf.reduce_mean(input1, var)

        # For updating the Var
        const_var = tf.constant([1])
        var_add = tf.add(var, const_var)
        var_update = var.assign(var_add)

        # update to happen after mean computation
        with tf.control_dependencies([mean]):
            var_update = var.assign(var_add)

        with tf.control_dependencies([var_update]):
            update_op = tf.no_op('train_op')

        # Initialize Var
        var_init_value = sess.run((var_initialize))

        # Compute mean and var updates
        mean_values = []
        for i in range(3):
            (result_mean, result_up) = sess.run((mean, update_op))
            mean_values.append(result_mean)

        # Compute Final Values
        var_final_val = var.eval(sess)
        return var_init_value, mean_values, var_final_val

    def test_variable_static_input_variables_dont_share_buffer(self):
        # This test is not applicable for CPU as NGVariable's NG and TF Tensors
        # share buffer on CPU. To simulate other backend's non buffer sharing
        # property we can use this env flag NGRAPH_TF_NGVARIABLE_BUFFER_SHARING

        # set env variable to disable NGraphVariable's buffer sharing
        buffer_sharing_env = "NGRAPH_TF_NGVARIABLE_BUFFER_SHARING"
        env_var_map = self.store_env_variables([buffer_sharing_env])
        self.set_env_variable(buffer_sharing_env, "0")

        # Run on nGraph
        ng_var_init_val, ng_mean_values, ng_var_final = self.with_ngraph(
            self.__run_test)

        # Reset Graph
        # It is necessary to reset the graph because of the variables
        # TF thinks you want to reuse the variables
        tf.reset_default_graph()

        # Run on TF
        tf_var_init_val, tf_mean_values, tf_var_final = self.without_ngraph(
            self.__run_test)

        # Compare Values
        # initial Var value will match
        assert np.allclose(ng_var_init_val, tf_var_init_val)

        # 1st iteration mean value will match, 2nd and 3rd wont
        assert np.allclose(ng_mean_values[0], tf_mean_values[0])

        if ngraph_bridge.are_variables_enabled():
            assert (np.allclose(ng_mean_values[1], tf_mean_values[1]) == False)
            assert (np.allclose(ng_mean_values[2], tf_mean_values[2]) == False)

        # Final Var value will match
        assert np.allclose(ng_var_final, tf_var_final)

        # clean up
        self.unset_env_variable(buffer_sharing_env)
        self.restore_env_variables(env_var_map)

    # Everything works fine when buffer is shared
    def test_variable_static_input_variables_share_buffer(self):
        # set env variable to enable NGraphVariable's buffer sharing
        buffer_sharing_env = "NGRAPH_TF_NGVARIABLE_BUFFER_SHARING"
        env_var_map = self.store_env_variables([buffer_sharing_env])
        self.set_env_variable(buffer_sharing_env, "1")

        # Run on nGraph
        ng_var_init_val, ng_mean_values, ng_var_final = self.with_ngraph(
            self.__run_test)

        # Reset Graph
        # It is necessary to reset the graph because of the variables
        # TF thinks you want to reuse the variables
        tf.reset_default_graph()

        # Run on TF
        tf_var_init_val, tf_mean_values, tf_var_final = self.without_ngraph(
            self.__run_test)

        # Compare Values
        # initial Var value will match
        assert np.allclose(ng_var_init_val, tf_var_init_val)

        # mean value matches for all iterations
        assert np.allclose(ng_mean_values[0], tf_mean_values[0])
        assert np.allclose(ng_mean_values[1], tf_mean_values[1])
        assert np.allclose(ng_mean_values[2], tf_mean_values[2])

        # Final Var value will match
        assert np.allclose(ng_var_final, tf_var_final)

        # clean up
        self.unset_env_variable(buffer_sharing_env)
        self.restore_env_variables(env_var_map)
