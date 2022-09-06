# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow conv2D runtime tests for 'arguments check'

They replicate the tf python test 'testOpEdgeCases' defined in
tensorflow/tensorflow/python/kernel_tests/conv_ops_test.py in principle.
With nGraph,the tf python test fails in kernel computation of the 
placeholder as placholder values are missing and it does not reach the 
kernel computation for Conv2D (now encapsulated).
Tf tests these input parameters in kernel construction while we can only 
test them during kernel computation. Hence, TF test can get away with not
specifying the placeholder values

The 3rd and 4th test(under # Filter larger than input/ "Negative dimension
size") in testOpEdgeCases are construction time errors and do not reach
nGraph.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import errors_impl
from common import NgraphTest
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test

import numpy as np


class TestConv2DBackpropInput(NgraphTest, test.TestCase):
    INPUT_SIZES_NHWC = [1, 7, 6, 2]
    FILTER_IN_SIZES = [3, 3, 2, 2]

    def make_filter_and_backprop_args(self):
        total_size_1 = 1
        total_size_2 = 1

        for s in self.INPUT_SIZES_NHWC:
            total_size_1 *= s
        for s in self.FILTER_IN_SIZES:
            total_size_2 *= s

        x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
        x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]

        return x1, x2

    @pytest.mark.parametrize(("strides",), (
        ([2, 1, 1, 1],),
        ([1, 1, 1, 2],),
    ))
    def test_conv2d_stride_in_batch_not_supported(self):
        inp_values, filt_values = self.make_filter_and_backprop_args()

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            filt = array_ops.placeholder(dtypes.float32)
            return sess.run(
                nn_ops.conv2d(inp, filt, strides=[2, 1, 1, 1], padding="SAME"),
                {
                    inp: inp_values,
                    filt: filt_values
                })

        with pytest.raises(Exception) as excinfo:
            self.with_ngraph(run_test)
        if self.assertRaises(errors_impl.InvalidArgumentError):
            if not "Check 'input_order_shape.compatible" in excinfo.value.message:
                raise AssertionError

        # TF also fails
        with pytest.raises(Exception) as excinfo1:
            self.without_ngraph(run_test)
