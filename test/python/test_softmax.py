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
"""nGraph TensorFlow relu6 test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import numpy as np

from common import NgraphTest


class TestSoftmax(NgraphTest):

    def test_softmax_2d(self):
        x = tf.placeholder(tf.float32, shape=(2, 3))

        # input value and expected value
        x_np = np.random.rand(2, 3)
        one_only_on_dim = list(x_np.shape)
        dim = len(x_np.shape) - 1
        one_only_on_dim[dim] = 1
        y_np = np.exp(x_np)
        a_np = y_np / np.reshape(np.sum(y_np, dim), one_only_on_dim)
        expected = a_np
        a = tf.nn.softmax(x)
        sess_fn = lambda sess: sess.run((a), feed_dict={x: x_np})
        np.allclose(self.with_ngraph(sess_fn), self.without_ngraph(sess_fn))
        np.allclose(self.with_ngraph(sess_fn), expected)

    def test_softmax_3d(self):
        x = tf.placeholder(tf.float32, shape=(2, 3, 2))

        # input value and expected value
        x_np = np.random.rand(2, 3, 2)
        one_only_on_dim = list(x_np.shape)
        dim = len(x_np.shape) - 1
        one_only_on_dim[dim] = 1
        y_np = np.exp(x_np)
        a_np = y_np / np.reshape(np.sum(y_np, dim), one_only_on_dim)
        expected = a_np
        a = tf.nn.softmax(x)
        sess_fn = lambda sess: sess.run((a), feed_dict={x: x_np})
        np.allclose(self.with_ngraph(sess_fn), self.without_ngraph(sess_fn))
        np.allclose(self.with_ngraph(sess_fn), expected)

    def test_softmax_4d(self):
        x = tf.placeholder(tf.float32, shape=(2, 3, 2, 4))

        # input value and expected value
        x_np = np.random.rand(2, 3, 2, 4)
        one_only_on_dim = list(x_np.shape)
        dim = len(x_np.shape) - 1
        one_only_on_dim[dim] = 1
        y_np = np.exp(x_np)
        a_np = y_np / np.reshape(np.sum(y_np, dim), one_only_on_dim)
        expected = a_np
        a = tf.nn.softmax(x)
        sess_fn = lambda sess: sess.run((a), feed_dict={x: x_np})
        np.allclose(self.with_ngraph(sess_fn), self.without_ngraph(sess_fn))
        np.allclose(self.with_ngraph(sess_fn), expected)
