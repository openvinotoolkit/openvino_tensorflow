# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow cast operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestCastOperations(NgraphTest):

    def test_cast_1d(self):
        val = tf.compat.v1.placeholder(tf.float32, shape=(2,))
        out = tf.cast(val, dtype=tf.int32)

        def run_test(sess):
            return sess.run(out, feed_dict={val: (5.5, 2.0)})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_cast_2d(self):
        test_input = ((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
        val = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))
        out = tf.cast(val, dtype=tf.int32)

        def run_test(sess):
            return sess.run(out, feed_dict={val: test_input})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
