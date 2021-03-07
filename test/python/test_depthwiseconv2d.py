# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow depthwise_conv2d operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
import numpy as np
from common import NgraphTest


class TestDepthwiseConv2dOperations(NgraphTest):

    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Only for Linux')
    @pytest.mark.parametrize("padding", ["VALID", "SAME"])
    @pytest.mark.parametrize(('tensor_in_sizes', 'filter_in_sizes'),
                             [([1, 2, 3, 2], [2, 2, 2, 2]),
                              ([1, 3, 2, 1], [2, 1, 1, 2]),
                              ([1, 3, 1, 2], [1, 1, 2, 2])])
    def test_depthwise_conv2d(self, padding, tensor_in_sizes, filter_in_sizes):
        tensor_in_sizes = tensor_in_sizes
        filter_in_sizes = filter_in_sizes
        total_size_1 = 1
        total_size_2 = 1

        for s in tensor_in_sizes:
            total_size_1 *= s
        for s in filter_in_sizes:
            total_size_2 *= s

        x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
        x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]

        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t1.set_shape(tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        conv = nn_ops.depthwise_conv2d_native(
            t1, t2, strides=[1, 1, 1, 1], padding=padding)
        sess_fn = lambda sess: sess.run(conv)

        assert np.isclose(
            self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)).all()
