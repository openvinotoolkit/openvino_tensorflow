# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow abs operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os

from common import NgraphTest


class TestAbsOperations(NgraphTest):

    @pytest.mark.parametrize("test_input", (1.4, -0.5, -1))
    def test_abs_1d(self, test_input):
        val = tf.compat.v1.placeholder(tf.float32, shape=(1,))
        out = tf.abs(val)

        def run_test(sess):
            return sess.run((out,), feed_dict={val: (test_input,)})

        if not self.with_ngraph(run_test) == self.without_ngraph(run_test):
            raise AssertionError

    def test_abs_2d(self):
        test_input = ((1.5, -2.5, 0.0, -3.5), (-4.5, -5.5, 6.5, 1.0))
        val = tf.compat.v1.placeholder(tf.float32, shape=(2, 4))
        out = tf.abs(val)

        def run_test(sess):
            return sess.run(out, feed_dict={val: test_input})

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)
               ).all():
            raise AssertionError
