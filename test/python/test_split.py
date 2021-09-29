# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow split operation test

"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pytest

from common import NgraphTest


class TestSplitOperations(NgraphTest):

    @pytest.mark.parametrize(("shape", "sizes", "split_dim"),
                             (((2, 7), [1, 1, 5], 1), ((2, 7), [1, 1], 0),
                              ((9, 9, 9), [2, 2, 5], 1)))
    def test_split_sizes(self, shape, sizes, split_dim):
        a = tf.compat.v1.placeholder(tf.float32, shape)

        tensors = tf.split(a, sizes, split_dim)
        input_val = np.random.random_sample(shape)

        # numpy split requires split positions instead of
        # sizes, for sizes vector [1, 1, 5] split positions
        # would be [1, 2], for [1, 1, 5, 2] : [1, 2, 7] etc..

        np_split_points = [sizes[0]]
        for i in range(1, len(sizes) - 1):
            np_split_points.append(np_split_points[i - 1] + sizes[i])

        slices = np.split(input_val, np_split_points, split_dim)

        sess_fn = lambda sess: sess.run([tensors], feed_dict={a: input_val})
        results = self.with_ngraph(sess_fn)
        if not len(results[0]) == len(slices):
            raise AssertionError
        for i in range(len(slices)):
            if not slices[i].shape == results[0][i].shape:
                raise AssertionError
            if not np.allclose(slices[i], results[0][i]):
                raise AssertionError

    @pytest.mark.parametrize(("shape", "number", "split_dim"),
                             (((2, 7), 7, 1), ((2, 7), 1, 0), ((9, 9), 3, 0)))
    def test_split_num(self, shape, number, split_dim):
        a = tf.compat.v1.placeholder(tf.float32, shape)

        tensors = tf.split(a, number, split_dim)
        input_val = np.random.random_sample(shape)
        slices = np.split(input_val, number, split_dim)

        sess_fn = lambda sess: sess.run([tensors], feed_dict={a: input_val})
        results = self.with_ngraph(sess_fn)
        if not len(results[0]) == len(slices):
            raise AssertionError
        for i in range(len(slices)):
            if not slices[i].shape == results[0][i].shape:
                raise AssertionError
            if not np.allclose(slices[i], results[0][i]):
                raise AssertionError

    def test_split_outputs_order(self):
        a = tf.compat.v1.placeholder(tf.float32, (5,))

        (t0, t1) = tf.split(a, [2, 3], 0)
        # Add operation is in the same ngraph block, following split.
        # Results should be fetched in correct order by Add op
        # implemented in ngraph_builder.cc
        t1plus = t1 + [0, 0, 0]
        t0plus = [0, -1] + t0

        sess_fn = lambda sess: sess.run([t1plus, t0plus],
                                        feed_dict={a: [0, 1, 2, 3, 4]})
        (r1, r0) = self.with_ngraph(sess_fn)
        if not len(r1) == 3:
            raise AssertionError
        if not len(r0) == 2:
            raise AssertionError
        if not np.allclose(r1, [2, 3, 4]):
            raise AssertionError
        if not np.allclose(r0, [0, 0]):
            raise AssertionError

    def test_split_cpu_one_output(self):
        a = tf.compat.v1.placeholder(tf.float32, (5,))

        (t0, t1) = tf.split(a, [2, 3], 0)
        t1plus = t1 + [0, 0, 0]
        sess_fn = lambda sess: sess.run([t1plus],
                                        feed_dict={a: [0, 1, 2, 3, 4]})
        r1 = self.with_ngraph(sess_fn)
        if not len(r1[0]) == 3:
            raise AssertionError
        if not np.allclose(r1[0], [2, 3, 4]):
            raise AssertionError
