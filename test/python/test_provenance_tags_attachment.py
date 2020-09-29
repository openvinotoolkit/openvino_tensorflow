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
"""nGraph TensorFlow bridge prod operations test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestProductOperations(NgraphTest):

    def test_provenance_for_no_effect_broadcast(self):
        # Creates a network: y = x + |x|
        #            ---------
        #          /           \
        # inp----->             + ---> out_node
        #          \           /
        #           ---abs----
        # The translation of the Add node, first broadcasts the 2 inputs
        # it receives and then creates an ngraph Add node.
        # Since the shapes of the inputs to the TF add node are same,
        # the broadcast builder will return the exact same inputs (lhs and rhs)
        # without creating new ng nodes
        # If we do not take care, we could be adding a tag to the ng abs node
        # when tagging the return of the broadcast builder

        # This test makes sure that TranslateGraph checks that the
        # builder returned nodes are different from its inputs,
        # and only in that case it adds provenance tags

        inp = tf.compat.v1.placeholder(
            tf.float32, shape=[1, 32, 32, 2], name='input')
        out_node = tf.add(tf.math.abs(inp, name="abs"), inp, name="add")
        self.with_ngraph(lambda sess: sess.run(
            out_node, feed_dict={inp: np.ones([1, 32, 32, 2])}))

    def test_provenance_for_broadcast_with_effect(self):
        # In this test, the broadcast actually produces new ng nodes
        # as opposed to test_provenance_for_no_effect_broadcast,
        # which is a dummy broadcast
        # so test that they are tagged appropriately
        inp0 = tf.compat.v1.placeholder(tf.float32, shape=[2, 2], name='input0')
        inp1 = tf.compat.v1.placeholder(tf.float32, shape=[2], name='input1')
        out_node0 = inp0 / inp1
        out_node1 = inp1 / inp0
        self.with_ngraph(lambda sess: sess.run([out_node0, out_node1],
                                               feed_dict={
                                                   inp0: np.ones([2, 2]),
                                                   inp1: np.ones([2])
                                               }))
