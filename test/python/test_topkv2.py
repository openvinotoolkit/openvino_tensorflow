# ==============================================================================
# Copyright (C) 2023 Intel Corporation

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


class TestTopKV2(NgraphTest):

    def test_topkv2_1d(self):
        input = [1.0, 5.0, 6.0, 12.0]
        p = tf.compat.v1.placeholder(dtype=tf.float32)
        out = tf.nn.top_k(p, k=3, sorted=True)

        def run_test(sess):
            return sess.run(out, feed_dict={p: input}).values

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)
               ).all():
            raise AssertionError

    def test_topkv2_2d(self):
        input = [[40.0, 30.0, 20.0, 10.0], [10.0, 20.0, 15.0, 70.0]]
        p = tf.compat.v1.placeholder(dtype=tf.float32)
        out = tf.nn.top_k(p, k=3, sorted=True)

        def run_test(sess):
            return sess.run(out, feed_dict={p: input}).values

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)
               ).all():
            raise AssertionError

    def test_topkv2_3d(self):
        input = [[[40.0, 30.0, 20.0], [20.0, 15.0, 70.0]],
                 [[45.0, 25.0, 43.0], [24.0, 12.0, 7.0]]]
        p = tf.compat.v1.placeholder(dtype=tf.float32)
        out = tf.nn.top_k(p, k=3, sorted=True)

        def run_test(sess):
            return sess.run(out, feed_dict={p: input}).values

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)
               ).all():
            raise AssertionError

    def test_topkv2_nosort(self):
        input = [[40.0, 30.0, 20.0, 10.0], [10.0, 20.0, 15.0, 70.0]]
        p = tf.compat.v1.placeholder(dtype=tf.float32)
        out = tf.nn.top_k(p, k=3, sorted=False)

        def run_test(sess):
            return sess.run(out, feed_dict={p: input}).values

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)
               ).all():
            raise AssertionError
