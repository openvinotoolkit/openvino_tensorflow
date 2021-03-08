# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow sign operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from common import NgraphTest


class TestSignOperations(NgraphTest):

    @pytest.mark.parametrize(("test_input", "expected"), ((1.4, 1), (-0.5, -1),
                                                          (0.0, 0)))
    def test_sign_1d(self, test_input, expected):
        val = tf.compat.v1.placeholder(tf.float32, shape=(1,))
        out = tf.sign(val)
        sess_fn = lambda sess: sess.run((out,), feed_dict={val: (test_input,)})
        assert np.allclose(
            self.with_ngraph(sess_fn), self.without_ngraph(sess_fn))
        assert np.allclose(self.with_ngraph(sess_fn), expected)

    def test_sign_2d(self):
        test_input = ((1.5, -2.5, -3.5), (-4.5, 5.5, 0))
        expected = ((1, -1, -1), (-1, 1, 0))
        val = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))
        out = tf.sign(val)
        sess_fn = lambda sess: sess.run((out,), feed_dict={val: test_input})
        assert np.allclose(
            self.with_ngraph(sess_fn), self.without_ngraph(sess_fn))
        assert np.allclose(self.with_ngraph(sess_fn), expected)
