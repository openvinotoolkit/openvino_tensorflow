# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

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


class TestSoftmax(NgraphTest):

    def test_softmax_2d(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))

        # input value and expected value
        x_np = np.random.rand(2, 3)
        one_only_on_dim = list(x_np.shape)
        dim = len(x_np.shape) - 1
        one_only_on_dim[dim] = 1
        y_np = np.exp(x_np)
        a_np = y_np / np.reshape(np.sum(y_np, dim), one_only_on_dim)
        expected = a_np
        a = tf.nn.softmax(x)
        sess_fn = lambda sess: sess.run((a), feed_dict={x: x_np})
        if not np.allclose(
                self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)):
            raise AssertionError
        if not np.allclose(self.with_ngraph(sess_fn), expected):
            raise AssertionError

    def test_softmax_3d(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(2, 3, 2))

        # input value and expected value
        x_np = np.random.rand(2, 3, 2)
        one_only_on_dim = list(x_np.shape)
        dim = len(x_np.shape) - 1
        one_only_on_dim[dim] = 1
        y_np = np.exp(x_np)
        a_np = y_np / np.reshape(np.sum(y_np, dim), one_only_on_dim)
        expected = a_np
        a = tf.nn.softmax(x)
        sess_fn = lambda sess: sess.run((a), feed_dict={x: x_np})
        if not np.allclose(
                self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)):
            raise AssertionError
        if not np.allclose(self.with_ngraph(sess_fn), expected):
            raise AssertionError

    def test_softmax_4d(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(2, 3, 2, 4))

        # input value and expected value
        x_np = np.random.rand(2, 3, 2, 4)
        one_only_on_dim = list(x_np.shape)
        dim = len(x_np.shape) - 1
        one_only_on_dim[dim] = 1
        y_np = np.exp(x_np)
        a_np = y_np / np.reshape(np.sum(y_np, dim), one_only_on_dim)
        expected = a_np
        a = tf.nn.softmax(x)
        sess_fn = lambda sess: sess.run((a), feed_dict={x: x_np})
        if not np.allclose(
                self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)):
            raise AssertionError
        if not np.allclose(self.with_ngraph(sess_fn), expected):
            raise AssertionError
