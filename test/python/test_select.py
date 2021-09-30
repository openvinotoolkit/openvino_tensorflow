# ==============================================================================
# Copyright (C) 2021 Intel Corporation

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


class TestSelect(NgraphTest):

    def test_select_scalar(self):
        a = [1.5]
        p = tf.compat.v1.placeholder(dtype=tf.bool)
        out = tf.where(p, x=[1], y=[0])

        def run_test(sess):
            return sess.run(out, feed_dict={p: a})

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError

    def test_select_sameshape(self):
        a = [True, False, True, True]
        p = tf.compat.v1.placeholder(dtype=tf.bool)
        out = tf.where(p, x=[1] * 4, y=[0] * 4)

        def run_test(sess):
            return sess.run(out, feed_dict={p: a})

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError

    def test_select_diffrank(self):
        a = [1, 1]
        x = [[0, 0], [2, 2]]
        y = [[2, 2], [1, 1]]
        p = tf.compat.v1.placeholder(dtype=tf.bool)
        out = tf.where(p, x, y)

        def run_test(sess):
            return sess.run(out, feed_dict={p: a})

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError

    def test_select_complexshape1(self):
        a = np.random.random(size=[7]).astype(np.float32)
        x = np.random.random(size=[7, 3, 2, 1]).astype(np.float32)

        p = tf.compat.v1.placeholder(dtype=tf.bool)
        out = tf.where(p, x, x)

        def run_test(sess):
            return (sess.run(out, feed_dict={p: a}))

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError

    def test_select_complexshape2(self):
        a = np.random.random(size=[7]).astype(np.float32)
        x = np.random.random(size=[7, 3, 2, 7]).astype(np.float32)

        p = tf.compat.v1.placeholder(dtype=tf.bool)
        out = tf.where(p, x, x)

        def run_test(sess):
            return (sess.run(out, feed_dict={p: a}))

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError

    def test_select_complexshape3(self):
        a = np.random.random(size=[5]).astype(np.float32)
        x = np.random.random(size=[5, 3, 1]).astype(np.float32)

        p = tf.compat.v1.placeholder(dtype=tf.bool)
        out = tf.where(p, x, x)

        def run_test(sess):
            return (sess.run(out, feed_dict={p: a}))

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError


class TestWhere(NgraphTest):
    env_map = None

    def setup_method(self):
        self.env_map = self.store_env_variables(
            ['OPENVINO_TF_CONSTANT_FOLDING'])
        self.set_env_variable('OPENVINO_TF_CONSTANT_FOLDING', '1')

    def teardown_method(self):
        self.restore_env_variables(self.env_map)

    def test_where(self):
        a = np.array([1.1, 3.0], [2.2, 4.4]).astype(np.float32)
        p = tf.compat.v1.placeholder(dtype=tf.float32, shape=(2, 2))
        out = tf.where(tf.equal(p, 3.0))

        def run_test(sess):
            return sess.run(out, feed_dict={p: a})

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError

    def test_where_scalar(self):
        a = [1.5]
        p = tf.compat.v1.placeholder(dtype=tf.bool)
        out = tf.where(p)

        def run_test(sess):
            return sess.run(out, feed_dict={p: a})

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError

    def test_where_bool(self):
        a = [True, False, False, True, False]
        p = tf.compat.v1.placeholder(dtype=tf.bool)
        out = tf.where(p)

        def run_test(sess):
            return sess.run(out, feed_dict={p: a})

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError

    def test_where_complexshape1(self):
        a = np.random.random(size=[7]).astype(np.float32)
        p = tf.compat.v1.placeholder(dtype=tf.bool)
        out = tf.where(p)

        def run_test(sess):
            return (sess.run(out, feed_dict={p: a}))

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all():
            raise AssertionError
