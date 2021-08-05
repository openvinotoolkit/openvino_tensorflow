# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow gather operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import numpy as np

from common import NgraphTest


class TestGatherOperations(NgraphTest):

    # Scalar indices
    def test_gather_0(self):
        val = tf.compat.v1.placeholder(tf.float32, shape=(5,))
        out = tf.raw_ops.Gather(params=val, indices=1)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: (10.0, 20.0, 30.0, 40.0, 50.0)})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()


class TestGatherV2Operations(NgraphTest):

    # Scalar indices
    def test_gather_0(self):
        val = tf.compat.v1.placeholder(tf.float32, shape=(5,))
        out = tf.gather(val, 1)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: (10.0, 20.0, 30.0, 40.0, 50.0)})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    # Vector indices, no broadcast required
    def test_gather_1(self):
        val = tf.compat.v1.placeholder(tf.float32, shape=(5,))
        out = tf.gather(val, [2, 1])

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: (10.0, 20.0, 30.0, 40.0, 50.0)})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    # Vector indices, broadcast required
    def test_gather_2(self):
        val = tf.compat.v1.placeholder(tf.float32, shape=(2, 5))
        out = tf.gather(val, [2, 1], axis=1)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: np.arange(10).reshape([2, 5])})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
        print(self.with_ngraph(run_test))

    # Vector indices, broadcast required, negative axis
    def test_gather_3(self):
        val = tf.compat.v1.placeholder(tf.float32, shape=(2, 5))
        out = tf.gather(val, [2, 1], axis=-1)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: np.arange(10).reshape([2, 5])})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
        print(self.with_ngraph(run_test))

    # higher rank indices... not working right now
    def test_gather_4(self):
        val = tf.compat.v1.placeholder(tf.float32, shape=(2, 5))
        out = tf.gather(val, [[0, 1], [1, 0]], axis=1)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: np.arange(10).reshape([2, 5])})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
        print(self.with_ngraph(run_test))


class TestGatherNdOperations(NgraphTest):

    # Simple indexing
    def test_gather_0(self):
        val = tf.compat.v1.placeholder(tf.string, shape=(2, 2))
        out = tf.raw_ops.Gather(params=val, indices=[[0, 0], [1, 1]])

        def run_test(sess):
            return sess.run((out,), feed_dict={val: [['a', 'b'], ['c',
                                                                  'd']]})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    # Slice indexing into a matrix
    def test_gather_1(self):
        val = tf.compat.v1.placeholder(tf.string, shape=(2, 2))
        out = tf.raw_ops.Gather(params=val, indices=[[1], [0]])

        def run_test(sess):
            return sess.run((out,), feed_dict={val: [['a', 'b'], ['c',
                                                                  'd']]})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    # Indexing into a 3-tensor
    def test_gather_2(self):
        val = tf.compat.v1.placeholder(tf.string, shape=(2, 2, 2))
        out = tf.raw_ops.Gather(params=val, indices=[[0, 1], [1, 0]])

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={
                                val: [[['a0', 'b0'], ['c0', 'd0']],
                                      [['a1', 'b1'], ['c1', 'd1']]]
                            })[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
