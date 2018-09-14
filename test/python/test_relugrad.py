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
"""nGraph TensorFlow bridge ReluGrad operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops.gen_nn_ops import relu_grad

from common import NgraphTest


class TestReluGradOperations(NgraphTest):

    def test_relugrad_2d(self):
        gradients = constant_op.constant(
            self.generate_random_numbers(6, 1.0, 10.0), shape=[2, 3])
        features = constant_op.constant(
            self.generate_random_numbers(6, 0.0, 100.0), shape=[2, 3])

        # Run on nGraph
        with self.device:
            out = relu_grad(gradients, features)
            with self.session as sess:
                result = sess.run(out)

        # Run on CPU
        with self.cpu_device:
            out = relu_grad(gradients, features)
            with self.session as sess:
                expected = sess.run(out)

        assert (result == expected).all()

    def test_relugrad_1d(self):
        gradients = constant_op.constant(
            self.generate_random_numbers(100, 123.0, 345.0), shape=[100])
        features = constant_op.constant(
            self.generate_random_numbers(100, 567.0, 789.0), shape=[100])

        # Run on nGraph
        with self.device:
            out = relu_grad(gradients, features)
            with self.session as sess:
                result = sess.run(out)

        # Run on CPU
        with self.cpu_device:
            out = relu_grad(gradients, features)
            with self.session as sess:
                expected = sess.run(out)

        assert (result == expected).all()
