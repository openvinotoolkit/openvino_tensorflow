"""Tests for tensorflow.ops.tf.scatter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from common import NgraphTest

from tensorflow.python.platform import test


def _AsType(v, vtype):
    return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _NumpyUpdate(ref, indices, updates):
    for i, indx in np.ndenumerate(indices):
        ref[indx] = updates[i]


class TestScatterNdUpdate(NgraphTest, test.TestCase):

    def test_scatter_basic_1(self):

        test_indices = np.array([[4], [3], [1], [7]])
        test_updates = np.array([9, 10, 11, 12])
        test_shape = np.array([8])
        expected = [0, 11, 0, 10, 9, 0, 0, 12]

        indices = tf.compat.v1.placeholder(
            test_indices.dtype, shape=test_indices.shape)
        updates = tf.compat.v1.placeholder(
            test_updates.dtype, shape=test_updates.shape)
        shape = tf.compat.v1.placeholder(
            test_shape.dtype, shape=test_shape.shape)

        out = tf.scatter_nd(indices, updates, shape)

        if not np.isclose(self.with_ngraph(lambda sess: sess.run(out, feed_dict={indices: test_indices, updates: test_updates, shape: test_shape})), np.array([expected])).all():
            raise AssertionError

    def test_scatter_basic_2(self):

        test_indices = np.array([[0], [2]])
        test_updates = np.array([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7],
                                  [8, 8, 8, 8]],
                                 [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7],
                                  [8, 8, 8, 8]]])
        test_shape = np.array([4, 4, 4])
        expected = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

        indices = tf.compat.v1.placeholder(
            test_indices.dtype, shape=test_indices.shape)
        updates = tf.compat.v1.placeholder(
            test_updates.dtype, shape=test_updates.shape)
        shape = tf.compat.v1.placeholder(
            test_shape.dtype, shape=test_shape.shape)

        out = tf.scatter_nd(indices, updates, shape)

        if not np.isclose(self.with_ngraph(lambda sess: sess.run(out, feed_dict={indices: test_indices, updates: test_updates, shape: test_shape})), np.array([expected])).all():
                    raise AssertionError
