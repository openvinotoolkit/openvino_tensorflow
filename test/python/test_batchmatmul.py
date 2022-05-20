# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow BatchMatMul operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestBatchMatmul(NgraphTest):

    def test_matmul(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(2, 2, 3))
        y = tf.compat.v1.placeholder(tf.float32, shape=(2, 3, 2))
        out = tf.raw_ops.BatchMatMul(x=x, y=y, adj_x=False, adj_y=False)

        feed_dict = {
            x: np.arange(1, 13).reshape([2, 2, 3]),
            y: np.arange(13, 25).reshape([2, 3, 2])
        }

        def run_test(sess):
            return sess.run(out, feed_dict=feed_dict)

        if not np.allclose(
                self.with_ngraph(run_test),
                self.without_ngraph(run_test),
                atol=1e-2):

            raise AssertionError
