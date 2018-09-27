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
from tensorflow.python.ops.gen_math_ops import sigmoid_grad

from common import NgraphTest

import numpy as np


class TestSigmoidGradOperations(NgraphTest):

    def test_sigmoidgrad_2d(self):
        y = constant_op.constant(
            self.generate_random_numbers(30, 1.0, 10.0), shape=[3, 10])
        y_delta = constant_op.constant(
            self.generate_random_numbers(30, 0.0, 10.0), shape=[3, 10])

        out = sigmoid_grad(y, y_delta)

        def run_test(sess):
            return sess.run(out)

        assert np.allclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test))

    def test_sigmoidgrad_3d(self):
        y = constant_op.constant(
            self.generate_random_numbers(450, 1.0, 15.0), shape=[10, 3, 15])
        y_delta = constant_op.constant(
            self.generate_random_numbers(450, -3.0, 10.0), shape=[10, 3, 15])

        out = sigmoid_grad(y, y_delta)

        def run_test(sess):
            return sess.run(out)

        assert np.allclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test))

    def test_sigmoidgrad_5d(self):
        y = constant_op.constant(
            self.generate_random_numbers(240240, 1.0, 15.0),
            shape=[10, 11, 12, 13, 14])
        y_delta = constant_op.constant(
            self.generate_random_numbers(240240, -3.0, 10.0),
            shape=[10, 11, 12, 13, 14])

        out = sigmoid_grad(y, y_delta)

        def run_test(sess):
            return sess.run(out)

        assert np.allclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test))
