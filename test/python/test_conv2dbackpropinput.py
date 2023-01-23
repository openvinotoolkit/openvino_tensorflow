# ==============================================================================
# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow depthwise_conv2d operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops

from common import NgraphTest

import numpy as np


class TestConv2DBackpropInput(NgraphTest):
    INPUT_SIZES_NCHW = [1, 2, 7, 6]
    INPUT_SIZES_NHWC = [1, 7, 6, 2]
    FILTER_IN_SIZES = [3, 3, 2, 2]
    OUT_BACKPROP_IN_SIZES = {"VALID": [1, 2, 3, 2], "SAME": [1, 2, 4, 3]}

    def make_filter_and_backprop_args(self, out_backprop_in_sizes):
        total_size_1 = 1
        total_size_2 = 1

        for s in out_backprop_in_sizes:
            total_size_1 *= s
        for s in self.FILTER_IN_SIZES:
            total_size_2 *= s

        x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
        x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]

        return x1, x2

    @pytest.mark.parametrize("padding", ("VALID", "SAME"))
    def test_nchw(self, padding):
        # The expected size of the backprop will depend on whether padding is VALID
        # or SAME.
        out_backprop_in_sizes = self.OUT_BACKPROP_IN_SIZES[padding]
        x1, x2 = self.make_filter_and_backprop_args(out_backprop_in_sizes)

        def run_test_ngraph(sess):
            t1 = constant_op.constant(self.INPUT_SIZES_NCHW)
            t2 = constant_op.constant(x2, shape=self.FILTER_IN_SIZES)
            t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
            inp = nn_ops.conv2d_backprop_input(
                t1,
                t2,
                t3,
                strides=[1, 1, 2, 2],
                padding=padding,
                data_format='NCHW')
            return sess.run(inp)

        # To validate on the CPU side we will need to run in NHWC, because the CPU
        # implementation of conv/conv backprop does not support NCHW. We will
        # transpose on the way in and on the way out.
        def run_test_tf(sess):
            t1 = constant_op.constant(self.INPUT_SIZES_NHWC)
            t2 = constant_op.constant(x2, shape=self.FILTER_IN_SIZES)
            t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
            t3 = tf.transpose(t3, [0, 2, 3, 1])
            inp = nn_ops.conv2d_backprop_input(
                t1,
                t2,
                t3,
                strides=[1, 2, 2, 1],
                padding=padding,
                data_format='NHWC')
            inp = tf.transpose(inp, [0, 3, 1, 2])
            return sess.run(inp)

        if not np.allclose(
                self.with_ngraph(run_test_ngraph),
                self.without_ngraph(run_test_tf)):
            raise AssertionError

    @pytest.mark.parametrize("padding", ("VALID", "SAME"))
    def test_nhwc(self, padding):
        out_backprop_in_sizes = self.OUT_BACKPROP_IN_SIZES[padding]
        x1, x2 = self.make_filter_and_backprop_args(out_backprop_in_sizes)
        t1 = constant_op.constant(self.INPUT_SIZES_NHWC)
        t2 = constant_op.constant(x2, shape=self.FILTER_IN_SIZES)
        t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
        t3 = tf.transpose(t3, [0, 2, 3, 1])
        inp = nn_ops.conv2d_backprop_input(
            t1,
            t2,
            t3,
            strides=[1, 2, 2, 1],
            padding=padding,
            data_format='NHWC')

        def run_test(sess):
            return sess.run(inp)

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)
               ).all():
            raise AssertionError
