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
"""nGraph TensorFlow bridge ReluGrad operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import constant_op
from tensorflow.python.ops.gen_nn_ops import relu_grad

from common import NgraphTest

np.random.seed(5)


class TestReluGradOperations(NgraphTest):

    def test_relugrad_2d(self):
        gradients = tf.compat.v1.placeholder(tf.float32, [2, 3])
        features = tf.compat.v1.placeholder(tf.float32, [2, 3])
        out = relu_grad(gradients, features)
        g = np.random.rand(2, 3)
        f = np.random.rand(2, 3)
        sess_fn = lambda sess: sess.run(
            out, feed_dict={
                gradients: g,
                features: f
            })
        assert (np.allclose(
            self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)))

    def test_relugrad_1d(self):
        gradients = tf.compat.v1.placeholder(tf.float32, [100])
        features = tf.compat.v1.placeholder(tf.float32, [100])
        out = relu_grad(gradients, features)
        g = np.random.rand(100)
        f = np.random.rand(100)
        sess_fn = lambda sess: sess.run(
            out, feed_dict={
                gradients: g,
                features: f
            })
        assert (np.allclose(
            self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)))
