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
"""nGraph TensorFlow bridge pow operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf

from common import NgraphTest


@pytest.mark.skip(reason="new deviceless mode WIP")
class TestPowOperations(NgraphTest):

    @pytest.mark.parametrize(("lhs", "rhs"), ((1.4, 1.0), (-0.5, 2), (5, -1.1)))
    def test_pow_1d(self, lhs, rhs):
        val1 = tf.placeholder(tf.float32, shape=(1,))
        val2 = tf.placeholder(tf.float32, shape=(1,))
        expected = lhs**rhs

        with self.device:
            out = tf.pow(val1, val2)

            with self.session as sess:
                result = sess.run((out,),
                                  feed_dict={
                                      val1: (lhs,),
                                      val2: (rhs,)
                                  })
                assert result[0] == expected

    def test_pow_2d(self):
        lhs = ((1.5, -2.5, -3.5), (-4.5, -5.5, 6.5))
        rhs = ((5.0, 4.0, 3.0), (2.0, 1.0, 0.0))

        val1 = tf.placeholder(tf.float32, shape=(2, 3))
        val2 = tf.placeholder(tf.float32, shape=(2, 3))

        with self.device:
            out = tf.pow(val1, val2)

            with self.session as sess:
                result = sess.run(out, feed_dict={val1: lhs, val2: rhs})
                assert (result == np.power(lhs, rhs)).all()
