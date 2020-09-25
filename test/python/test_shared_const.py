# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
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
"""nGraph TensorFlow bridge pad operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import ngraph_bridge
from common import NgraphTest

np.random.seed(5)


class TestSharedConst(NgraphTest):

    def test_sharedconst1(self):
        # The const in this case is shared between Pad1 and Pad2
        # separated by Abs which is not encapsulated. The Pad op
        # is different in the sense that the shared const is a
        # static input to it and is not used by the op directly
        # but is used to create 2 new Consts.

        ngraph_bridge.set_disabled_ops('Abs')
        input_data1 = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))
        paddings = tf.compat.v1.constant([[2, 1], [2, 2]])

        pad1 = tf.pad(input_data1, paddings)
        abs2 = tf.abs(pad1)
        pad2 = tf.pad(abs2, paddings)

        inp = ((4, 2, 4), (4, 4, 1))
        pad = ((5, 3), (5, 5))

        def run_test(sess):
            return sess.run(pad2, feed_dict={input_data1: inp})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

        # Clean up
        ngraph_bridge.set_disabled_ops('')
