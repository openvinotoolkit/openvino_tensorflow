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
"""nGraph TensorFlow bridge SoftmaxCrossEntropyWithLogits operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.ops.gen_nn_ops import softmax_cross_entropy_with_logits
import numpy as np
from common import NgraphTest

np.random.seed(7)


# See TF C++ API https://www.tensorflow.org/versions/r1.15/api_docs/cc/class/tensorflow/ops/softmax-cross-entropy-with-logits.html
# Computes softmax cross entropy cost and gradients to backpropagate. Inputs are the logits, not probabilities.
# Both labels and fetaures are shape [BatchSize,NumClasses]
class TestSoftmaxCrossEntropyWithLogitsOperations(NgraphTest):

    def test_softmax_cross_entropy_with_logits_2d(self):
        num_classes = 10
        batch_size = 100
        total_size = batch_size * num_classes

        # labels should be a valid prob distribution across num_classes for each training sample
        # Compute the softmax transformation along the second axis (i.e. along the rows summing over num_classes)
        labels = tf.nn.softmax(np.random.rand(batch_size, num_classes), axis=1)

        # features/Logits could be any real number
        features = 200 * np.random.rand(batch_size, num_classes)

        out = softmax_cross_entropy_with_logits(features, labels)
        sess_fn = lambda sess: sess.run(out)

        expected = self.without_ngraph(sess_fn)
        result = self.with_ngraph(sess_fn)

        assert np.allclose(result[0], expected[0], rtol=0, atol=1e-02)  # loss
        assert np.allclose(
            result[1], expected[1], rtol=0, atol=1e-02)  # backprop

    def test_softmax_cross_entropy_with_logits_2d1d(self):
        num_classes = 10
        batch_size = 100
        total_size = batch_size * num_classes

        # labels should be a valid prob distribution across num_classes for each training sample
        # Compute the softmax transformation along the second axis (i.e. along the rows summing over num_classes)
        labels = tf.nn.softmax(np.random.rand(1, num_classes), axis=1)

        # features/Logits could be any real number
        features = 200 * np.random.rand(batch_size, num_classes)

        out = softmax_cross_entropy_with_logits(features, labels)
        sess_fn = lambda sess: sess.run(out)

        expected = self.without_ngraph(sess_fn)
        result = self.with_ngraph(sess_fn)

        assert np.allclose(result[0], expected[0], rtol=0, atol=1e-02)  # loss
        assert np.allclose(
            result[1], expected[1], rtol=0, atol=1e-02)  # backprop

    def test_softmax_cross_entropy_with_logits_1d2d(self):
        num_classes = 10
        batch_size = 100
        total_size = batch_size * num_classes

        # labels should be a valid prob distribution across num_classes for each training sample
        # Compute the softmax transformation along the second axis (i.e. along the rows summing over num_classes)
        labels = tf.nn.softmax(np.random.rand(batch_size, num_classes), axis=1)

        # features/Logits could be any real number
        features = 200 * np.random.rand(1, num_classes)

        out = softmax_cross_entropy_with_logits(features, labels)
        sess_fn = lambda sess: sess.run(out)

        expected = self.without_ngraph(sess_fn)
        result = self.with_ngraph(sess_fn)

        assert np.allclose(result[0], expected[0], rtol=0, atol=1e-02)  # loss
        assert np.allclose(
            result[1], expected[1], rtol=0, atol=1e-02)  # backprop
