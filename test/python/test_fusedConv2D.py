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
"""nGraph TensorFlow bridge fusedConv2D tests.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from common import NgraphTest
from tensorflow.python.framework import dtypes

import numpy as np


class TestFusedConv2D(NgraphTest):
    INPUT_SIZES = [3, 1, 6, 2]
    FILTER_SIZES = [1, 1, 2, 2]
    BIAS_SIZES = [2]

    def test_fusedconv2d_bias(self):
        inp_values = np.random.rand(*self.INPUT_SIZES)
        filt_values = np.random.rand(*self.FILTER_SIZES)
        bias_values = np.random.rand(*self.BIAS_SIZES)

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            filt = array_ops.placeholder(dtypes.float32)
            bias = array_ops.placeholder(dtypes.float32)
            return sess.run(
                nn_ops.bias_add(
                    nn_ops.conv2d(
                        inp, filt, strides=[1, 1, 1, 1], padding="SAME"), bias),
                {
                    inp: inp_values,
                    filt: filt_values,
                    bias: bias_values,
                })

        assert np.allclose(
            self.without_ngraph(run_test), self.with_ngraph(run_test))

    def test_fusedconv2d_bias_relu(self):
        inp_values = np.random.rand(*self.INPUT_SIZES)
        filt_values = np.random.rand(*self.FILTER_SIZES)
        bias_values = np.random.rand(*self.BIAS_SIZES)

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            filt = array_ops.placeholder(dtypes.float32)
            bias = array_ops.placeholder(dtypes.float32)
            return sess.run(
                nn_ops.relu(
                    nn_ops.bias_add(
                        nn_ops.conv2d(
                            inp, filt, strides=[1, 1, 1, 1], padding="SAME"),
                        bias)), {
                            inp: inp_values,
                            filt: filt_values,
                            bias: bias_values,
                        })

        assert np.allclose(
            self.without_ngraph(run_test), self.with_ngraph(run_test))

    def test_fusedconv2d_batchnorm(self):
        inp_values = np.random.rand(*self.INPUT_SIZES)
        filt_values = np.random.rand(*self.FILTER_SIZES)
        scale_values = np.random.rand(*self.BIAS_SIZES)
        offset_values = np.random.rand(*self.BIAS_SIZES)
        mean_values = np.random.rand(*self.BIAS_SIZES)
        variance_values = np.random.rand(*self.BIAS_SIZES)

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            filt = array_ops.placeholder(dtypes.float32)
            scale = array_ops.placeholder(dtypes.float32)
            offset = array_ops.placeholder(dtypes.float32)
            mean = array_ops.placeholder(dtypes.float32)
            variance = array_ops.placeholder(dtypes.float32)
            bn, _, _ = nn_impl.fused_batch_norm(
                nn_ops.conv2d(inp, filt, strides=[1, 1, 1, 1], padding="SAME"),
                scale,
                offset,
                mean,
                variance,
                epsilon=0.02,
                is_training=False)
            return sess.run(
                bn, {
                    inp: inp_values,
                    filt: filt_values,
                    scale: scale_values,
                    offset: offset_values,
                    mean: mean_values,
                    variance: variance_values,
                })

        assert np.allclose(
            self.without_ngraph(run_test),
            self.with_ngraph(run_test),
            rtol=0,
            atol=5e-5)

    def test_fusedconv2d_batchnorm_relu(self):
        inp_values = np.random.rand(*self.INPUT_SIZES)
        filt_values = np.random.rand(*self.FILTER_SIZES)
        scale_values = np.random.rand(*self.BIAS_SIZES)
        offset_values = np.random.rand(*self.BIAS_SIZES)
        mean_values = np.random.rand(*self.BIAS_SIZES)
        variance_values = np.random.rand(*self.BIAS_SIZES)

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            filt = array_ops.placeholder(dtypes.float32)
            scale = array_ops.placeholder(dtypes.float32)
            offset = array_ops.placeholder(dtypes.float32)
            mean = array_ops.placeholder(dtypes.float32)
            variance = array_ops.placeholder(dtypes.float32)
            bn, _, _ = nn_impl.fused_batch_norm(
                nn_ops.conv2d(inp, filt, strides=[1, 1, 1, 1], padding="SAME"),
                scale,
                offset,
                mean,
                variance,
                epsilon=0.02,
                is_training=False)
            return sess.run(
                nn_ops.relu(bn), {
                    inp: inp_values,
                    filt: filt_values,
                    scale: scale_values,
                    offset: offset_values,
                    mean: mean_values,
                    variance: variance_values,
                })

        assert np.allclose(
            self.without_ngraph(run_test), self.with_ngraph(run_test))

    def test_fusedconv2d_squeeze_bias(self):
        inp_values = np.random.rand(*self.INPUT_SIZES)
        filt_values = np.random.rand(*self.FILTER_SIZES)
        bias_values = np.random.rand(*self.BIAS_SIZES)
        squeeze_dim = [1]

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            filt = array_ops.placeholder(dtypes.float32)
            bias = array_ops.placeholder(dtypes.float32)
            return sess.run(
                nn_ops.bias_add(
                    array_ops.squeeze(
                        nn_ops.conv2d(
                            inp, filt, strides=[1, 1, 1, 1], padding="SAME"),
                        squeeze_dim), bias), {
                            inp: inp_values,
                            filt: filt_values,
                            bias: bias_values,
                        })

        assert np.allclose(
            self.without_ngraph(run_test), self.with_ngraph(run_test))
