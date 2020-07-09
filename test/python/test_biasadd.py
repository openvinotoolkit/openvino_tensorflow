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
"""nGraph TensorFlow bridge BiasAdd operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest

np.random.seed(8)


class TestBiasAddOperations(NgraphTest):

    def test_BiasAdd1(self):
        input_data = (0, 1, 0, 1, 2, 1, 1, 0, 3, 1, 1, 0, 4, 4, 5, 4)
        input_data = np.reshape(input_data, (2, 2, 2, 2))
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(2, 2, 2, 2))

        bias_data = (100., -100.)
        bias_var = tf.compat.v1.placeholder(tf.float32, shape=(2))

        out = tf.nn.bias_add(input_var, bias_var, 'NHWC')

        def run_test(sess):
            return sess.run(
                out, feed_dict={
                    input_var: input_data,
                    bias_var: bias_data
                })

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_BiasAdd2(self):
        input_data = (0, 1, 0, 1, 2, 1, 1, 0, 3, 1, 1, 0, 4, 4, 5, 4)
        input_data = np.reshape(input_data, (2, 2, 2, 2))
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(2, 2, 2, 2))

        bias_data = (100., -100.)
        bias_var = tf.compat.v1.placeholder(tf.float32, shape=(2))

        out = tf.nn.bias_add(input_var, bias_var, 'NCHW')

        def run_test(sess):
            return sess.run(
                out, feed_dict={
                    input_var: input_data,
                    bias_var: bias_data
                })

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_BiasAdd3(self):
        input_data = (0, 1, 0, 1, 2, 1, 1, 0, 3, 1, 1, 0, 4, 4, 5, 4, 3, 5, 1,
                      2, 0, 4, 0, 1)
        input_data = np.reshape(input_data, (2, 3, 2, 2))
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(2, 3, 2, 2))

        bias_data = (100., -100., 50)  # channels = 3
        bias_var = tf.compat.v1.placeholder(tf.float32, shape=(3))

        out = tf.nn.bias_add(input_var, bias_var, 'NCHW')

        def run_test(sess):
            return sess.run(
                out, feed_dict={
                    input_var: input_data,
                    bias_var: bias_data
                })

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_BiasAdd4(self):
        input_data = (0, 1, 0, 1, 2, 1, 1, 0, 3, 1, 1, 0, 4, 4, 5, 4, 3, 5, 1,
                      2, 0, 4, 0, 1)
        input_data = np.reshape(input_data, (2, 2, 2, 3))
        input_var = tf.compat.v1.placeholder(tf.float32, shape=(2, 2, 2, 3))

        bias_data = (100., -100., 50)  # channels = 3
        bias_var = tf.compat.v1.placeholder(tf.float32, shape=(3))

        out = tf.nn.bias_add(input_var, bias_var, 'NHWC')

        def run_test(sess):
            return sess.run(
                out, feed_dict={
                    input_var: input_data,
                    bias_var: bias_data
                })

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
