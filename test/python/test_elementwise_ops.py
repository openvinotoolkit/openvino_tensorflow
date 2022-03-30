# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow elementwise operations test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestElementwiseOperations(NgraphTest):

    @pytest.mark.parametrize(("v1", "v2", "expected"),
                             ((1.0, -1.0, [1.0]), (100, 200, ([200],)),
                              ([0.0, 5.0, 10.0], [6.0],
                               (np.array([[6.0, 6.0, 10.0]]),))))
    def test_maximum(self, v1, v2, expected):
        val1 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        val2 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        out = tf.maximum(val1, val2)

        sess_fn = lambda sess: sess.run((out,),
                                        feed_dict={
                                            val1: (v1,),
                                            val2: (v2,)
                                        })[0]
        if not (self.with_ngraph(sess_fn) == self.without_ngraph(sess_fn)
               ).all():
            raise AssertionError
        if not (self.with_ngraph(sess_fn) == expected).all():
            raise AssertionError

    @pytest.mark.parametrize(
        ("v1", "v2", "expected"),
        ((1.4, 1.0, [False]), (-1.0, -1.0, ([True],)), (-1.0, 1000, [True]),
         (200, 200, ([True],)), ([-1.0, 1.0, -4], [0.1, 0.1, -4],
                                 (np.array([[True, False, True]]),)),
         ([-1.0, 1.0, -4], [-1.0], (np.array([[True, False, True]]),))))
    def test_less_equal(self, v1, v2, expected):
        val1 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        val2 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        out = tf.less_equal(val1, val2)

        sess_fn = lambda sess: sess.run((out,),
                                        feed_dict={
                                            val1: (v1,),
                                            val2: (v2,)
                                        })[0]
        if not (self.with_ngraph(sess_fn) == self.without_ngraph(sess_fn)
               ).all():
            raise AssertionError
        if not (self.with_ngraph(sess_fn) == expected).all():
            raise AssertionError

    @pytest.mark.parametrize(
        ("v1", "v2", "expected"),
        ((1.4, 1.0, [False]), (-1.0, -1.0, ([False],)), (-1.0, 1000, [True]),
         (200, 200, ([False],)), ([-1.0, 1.0, -4], [0.1, 0.1, -4],
                                  (np.array([[True, False, False]]),)),
         ([-1.0, 1.0, -4], [-1.0], (np.array([[False, False, True]]),))))
    def test_less(self, v1, v2, expected):
        val1 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        val2 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        out = tf.less(val1, val2)

        sess_fn = lambda sess: sess.run((out,),
                                        feed_dict={
                                            val1: (v1,),
                                            val2: (v2,)
                                        })[0]
        if not (self.with_ngraph(sess_fn) == self.without_ngraph(sess_fn)
               ).all():
            raise AssertionError
        if not (self.with_ngraph(sess_fn) == expected).all():
            raise AssertionError

    @pytest.mark.parametrize(
        ("v1", "v2", "expected"),
        ((1.4, 1.0, [True]), (-1.0, -1.0, ([True],)), (-1.0, 1000, [False]),
         (200, 200, ([True],)), ([-1.0, 1.0, -4], [0.1, 0.1, -4],
                                 (np.array([[False, True, True]]),)),
         ([-1.0, 1.0, -4], [-1.0], (np.array([[True, True, False]]),))))
    def test_greater_equal(self, v1, v2, expected):
        val1 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        val2 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        out = tf.greater_equal(val1, val2)

        sess_fn = lambda sess: sess.run((out,),
                                        feed_dict={
                                            val1: (v1,),
                                            val2: (v2,)
                                        })[0]
        if not (self.with_ngraph(sess_fn) == self.without_ngraph(sess_fn)
               ).all():
            raise AssertionError
        if not (self.with_ngraph(sess_fn) == expected).all():
            raise AssertionError

    @pytest.mark.parametrize(
        ("v1", "v2", "expected"),
        ((1.4, 1.0, [True]), (-1.0, -1.0, ([False],)), (-1.0, 1000, [False]),
         (200, 200, ([False],)), ([-1.0, 1.0, -4], [0.1, 0.1, -4],
                                  (np.array([[False, True, False]]),)),
         ([-1.0, 1.0, -4], [-1.0], (np.array([[False, True, False]]),))))
    def test_greater(self, v1, v2, expected):
        val1 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        val2 = tf.compat.v1.placeholder(tf.float32, shape=(None))
        out = tf.greater(val1, val2)

        sess_fn = lambda sess: sess.run((out,),
                                        feed_dict={
                                            val1: (v1,),
                                            val2: (v2,)
                                        })[0]
        if not (self.with_ngraph(sess_fn) == self.without_ngraph(sess_fn)
               ).all():
            raise AssertionError
        if not (self.with_ngraph(sess_fn) == expected).all():
            raise AssertionError

    @pytest.mark.parametrize(("v1", "v2", "expected"),
                             ((True, True, [True]), (True, False, ([False],)),
                              (1.0, -2.0, ([True],)), (False, 100, ([False],)),
                              ([False, True, False], [True],
                               (np.array([[False, True, False]]),))))
    def test_logical_and(self, v1, v2, expected):
        val1 = tf.compat.v1.placeholder(tf.bool, shape=(None))
        val2 = tf.compat.v1.placeholder(tf.bool, shape=(None))
        out = tf.logical_and(val1, val2)

        sess_fn = lambda sess: sess.run((out,),
                                        feed_dict={
                                            val1: (v1,),
                                            val2: (v2,)
                                        })[0]
        if not (self.with_ngraph(sess_fn) == self.without_ngraph(sess_fn)
               ).all():
            raise AssertionError
        if not (self.with_ngraph(sess_fn) == expected).all():
            raise AssertionError

    @pytest.mark.parametrize(("test_input", "expected"), ((False, True),
                                                          (True, False)))
    def test_logicalnot_1d(self, test_input, expected):
        val = tf.compat.v1.placeholder(tf.bool, shape=(1,))
        out = tf.logical_not(val)

        sess_fn = lambda sess: sess.run((out,), feed_dict={val: (test_input,)})[
            0]
        if not (self.with_ngraph(sess_fn) == self.without_ngraph(sess_fn)
               ).all():
            raise AssertionError
        if not (self.with_ngraph(sess_fn) == expected).all():
            raise AssertionError

    def test_logicalnot_2d(self):
        test_input = ((True, False, True), (False, True, False))
        expected = np.logical_not(test_input)
        val = tf.compat.v1.placeholder(tf.bool, shape=(2, 3))
        out = tf.logical_not(val)

        sess_fn = lambda sess: sess.run((out,), feed_dict={val: test_input})[0]
        if not (self.with_ngraph(sess_fn) == self.without_ngraph(sess_fn)
               ).all():
            raise AssertionError
        if not (self.with_ngraph(sess_fn) == expected).all():
            raise AssertionError
