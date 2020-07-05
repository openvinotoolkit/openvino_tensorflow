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

from common import NgraphTest

np.random.seed(5)


class TestPadOperations(NgraphTest):

    def test_pad1(self):
        input_data = tf.compat.v1.placeholder(tf.int32, shape=(2, 3))
        paddings = tf.compat.v1.placeholder(tf.int32, shape=(2, 2))

        out = tf.pad(input_data, paddings)

        inp = ((4, 2, 4), (4, 4, 1))
        pad = ((5, 3), (5, 5))

        def run_test(sess):
            return sess.run(out, feed_dict={input_data: inp, paddings: pad})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_pad2(self):
        input_data = tf.compat.v1.placeholder(tf.int32, shape=(2, 3, 2))
        paddings = tf.compat.v1.placeholder(tf.int32, shape=(3, 2))

        inp = ([[[1, 1], [3, 6], [4, 5]],
                [[7, 8], [3, 3],
                 [7, 5]]])  # shape: (2,3,2), input-dim Dn (rank) = 3
        pad = ([[1, 0], [3, 2], [2, 1]])  # shape: Dn x 2

        out = tf.pad(
            input_data, paddings, mode='CONSTANT',
            constant_values=7)  # shape: (3,8,5)

        def run_test(sess):
            return sess.run(out, feed_dict={input_data: inp, paddings: pad})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_pad3(self):
        input_data = tf.compat.v1.placeholder(tf.int32, shape=(2, 3))
        paddings = tf.compat.v1.placeholder(tf.int32, shape=(2, 2))

        inp = ([[1, 2, 3], [4, 5, 6]])  # Shape (2,3), input-dim Dn (rank) = 2
        pad = ([[1, 1], [2, 2]])  # shape: Dn x 2

        out = tf.pad(input_data, paddings, mode='REFLECT')  # shape: (4,7)

        def run_test(sess):
            return sess.run(out, feed_dict={input_data: inp, paddings: pad})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_pad4(self):
        input_data = tf.compat.v1.placeholder(tf.int32, shape=(2, 3))
        paddings = tf.compat.v1.placeholder(tf.int32, shape=(2, 2))

        inp = ([[1, 2, 3], [4, 5, 6]])  # Shape (2,3), input-dim Dn (rank) = 2
        pad = ([[1, 1], [2, 2]])  # shape: Dn x 2

        out = tf.pad(input_data, paddings, mode='SYMMETRIC')  # shape: (4,7)

        def run_test(sess):
            return sess.run(out, feed_dict={input_data: inp, paddings: pad})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
