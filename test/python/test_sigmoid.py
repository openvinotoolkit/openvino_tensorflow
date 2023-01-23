# ==============================================================================
# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow relu6 test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

from common import NgraphTest


class TestSigmoid(NgraphTest):

    def test_sigmoid(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))
        y = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))
        z = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))

        a = x + y + z
        b = tf.nn.sigmoid(a)

        # input value and expected value
        x_np = np.full((2, 3), 1.0)
        y_np = np.full((2, 3), 1.0)
        z_np = np.full((2, 3), 1.0)
        a_np = x_np + y_np + z_np
        b_np = 1. / (1. + np.exp(-a_np))
        expected = b_np

        sess_fn = lambda sess: sess.run((a, b),
                                        feed_dict={
                                            x: x_np,
                                            y: y_np,
                                            z: z_np
                                        })
        if not np.allclose(
                self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)):
            raise AssertionError
