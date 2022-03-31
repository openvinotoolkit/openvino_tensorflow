# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow fusedConv2D tests.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform

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

    def get_relu_op(self, relutype):
        return {
            'relu': nn_ops.relu,
            'relu6': nn_ops.relu6,
            '': (lambda x: x)
        }[relutype]

    @pytest.mark.parametrize(("relutype",), (
        ('relu',),
        ('relu6',),
        ('',),
    ))
    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Only for Linux')
    def test_fusedconv2d_bias_relu(self, relutype):
        inp_values = np.random.rand(*self.INPUT_SIZES)
        filt_values = np.random.rand(*self.FILTER_SIZES)
        bias_values = np.random.rand(*self.BIAS_SIZES)

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            filt = array_ops.placeholder(dtypes.float32)
            bias = array_ops.placeholder(dtypes.float32)
            relu_op = self.get_relu_op(relutype)
            return sess.run(
                relu_op(
                    nn_ops.bias_add(
                        nn_ops.conv2d(
                            inp, filt, strides=[1, 1, 1, 1], padding="SAME"),
                        bias)), {
                            inp: inp_values,
                            filt: filt_values,
                            bias: bias_values,
                        })

        if not np.allclose(
                self.without_ngraph(run_test), self.with_ngraph(run_test)):
            raise AssertionError

    @pytest.mark.parametrize(("relutype",), (
        ('relu',),
        ('relu6',),
        ('',),
    ))
    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Only for Linux')
    def test_fusedconv2d_batchnorm(self, relutype):
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
            relu_op = self.get_relu_op(relutype)
            bn, _, _ = nn_impl.fused_batch_norm(
                nn_ops.conv2d(inp, filt, strides=[1, 1, 1, 1], padding="SAME"),
                scale,
                offset,
                mean,
                variance,
                epsilon=0.02,
                is_training=False)
            return sess.run(
                relu_op(bn), {
                    inp: inp_values,
                    filt: filt_values,
                    scale: scale_values,
                    offset: offset_values,
                    mean: mean_values,
                    variance: variance_values,
                })

        if not np.allclose(
                self.without_ngraph(run_test),
                self.with_ngraph(run_test),
                rtol=0,
                atol=5e-5):
            raise AssertionError

    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Only for Linux')
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

        if not np.allclose(
                self.without_ngraph(run_test), self.with_ngraph(run_test)):
            raise AssertionError
