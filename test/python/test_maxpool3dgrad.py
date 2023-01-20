# ==============================================================================
# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow Maxpool3DGrad tests.

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


class TestMaxpool3DGrad(NgraphTest):
    INPUT_SIZES = [1, 3, 5, 4, 1]
    OUTPUT_SIZES = [1, 3, 5, 4, 1]
    GRAD_SIZES = [1, 3, 5, 4, 1]

    def test_maxpool3d_grad(self):
        inp_values = np.random.rand(*self.INPUT_SIZES)
        out_values = np.random.rand(*self.OUTPUT_SIZES)
        grad_values = np.random.rand(*self.GRAD_SIZES)

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            out = array_ops.placeholder(dtypes.float32)
            grad = array_ops.placeholder(dtypes.float32)
            return sess.run(
                nn_ops.MaxPool3DGrad(
                    orig_input=inp,
                    orig_output=out,
                    grad=grad,
                    ksize=[1, 1, 1, 1, 1],
                    strides=[1, 1, 1, 1, 1],
                    padding="SAME"), {
                        inp: inp_values,
                        out: out_values,
                        grad: grad_values,
                    })

        if not np.allclose(
                self.without_ngraph(run_test), self.with_ngraph(run_test)):
            raise AssertionError
