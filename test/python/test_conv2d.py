# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow fusedConv2D tests.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

from common import NgraphTest


class TestConv2D(NgraphTest):
    INPUT_SIZES = [3, 1, 6, 2]
    FILTER_SIZES = [1, 1, 2, 2]
    BIAS_SIZES = [2]

    def test_conv2d_multiply(self):
        inp_values = np.random.rand(*self.INPUT_SIZES)
        filt_values = np.random.rand(*self.FILTER_SIZES)
        bias_values = np.random.rand(*self.BIAS_SIZES)

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            filt = array_ops.placeholder(dtypes.float32)
            bias = array_ops.placeholder(dtypes.float32)
            return sess.run(
                tf.math.multiply(
                    nn_ops.conv2d(
                        inp, filt, strides=[1, 1, 1, 1], padding="SAME"), bias),
                {
                    inp: inp_values,
                    filt: filt_values,
                    bias: bias_values,
                })

        assert np.allclose(
            self.without_ngraph(run_test), self.with_ngraph(run_test))
