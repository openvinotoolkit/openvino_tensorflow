# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow floor operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestTanhOp(NgraphTest):

    @pytest.mark.parametrize(("test_input", "expected"),
                             ((1.4, np.tanh(1.4)), (0.5, np.tanh(0.5)),
                              (-0.3, np.tanh(-0.3))))
    def test_tanh_1d(self, test_input, expected):
        val = tf.compat.v1.placeholder(tf.float32, shape=(1,))
        atol = 1e-5
        out = tf.tanh(val)

        sess_fn = lambda sess: sess.run((out,), feed_dict={val: (test_input,)})
        result = self.with_ngraph(sess_fn)
        if not np.amax(np.absolute(result[0] - expected)) < atol:
            raise AssertionError

    def test_tanh_2d(self):
        test_input = ((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
        expected = np.tanh(test_input)

        val = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))
        atol = 1e-5

        out = tf.tanh(val)
        sess_fn = lambda sess: sess.run((out,), feed_dict={val: test_input})
        (result,) = self.with_ngraph(sess_fn)
        if not np.amax(np.absolute(result == expected)) < atol:
            raise AssertionError
