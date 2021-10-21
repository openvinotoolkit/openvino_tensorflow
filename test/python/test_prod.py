# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow prod operations test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestProductOperations(NgraphTest):

    @pytest.mark.parametrize(("v1", "axis", "expected"), (
        ([3.0, 2.0], -1, [6.0]),
        ([4.0, 3.0], 0, [12.0]),
        ([[4.0, 3.0], [2.0, 2.0]], 0, [8.0, 6.0]),
        ([[4.0, 3.0], [2.0, 2.0]], -2, [8.0, 6.0]),
        ([[2.0, 3.0], [4.0, 5.0]], (0, 1), [120]),
        ([[2.0, 3.0], [4.0, 5.0]], (), [[2.0, 3.0], [4.0, 5.0]]),
    ))
    def test_prod(self, v1, axis, expected):
        tensor = tf.compat.v1.placeholder(tf.float32, shape=(None))
        if not np.allclose(np.prod(v1, axis), expected):
            raise AssertionError
        out = tf.reduce_prod(tensor, axis=axis)
        sess_fn = lambda sess: sess.run([out], feed_dict={tensor: v1})
        if not np.allclose(self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)):
            raise AssertionError

    @pytest.mark.parametrize(("v1", "expected"), (((2.0, 2.0), [4.0]),))
    def test_prod_no_axis(self, v1, expected):
        tensor = tf.compat.v1.placeholder(tf.float32, shape=(None))
        out = tf.reduce_prod(tensor)
        sess_fn = lambda sess: sess.run((out,), feed_dict={tensor: v1})
        assert np.allclose(self.with_ngraph(sess_fn), expected)
        if not np.allclose(self.with_ngraph(sess_fn), expected):
            raise AssertionError

    @pytest.mark.parametrize(("v1", "axis", "expected"),
                             (((2.0, 2.0), 0, [4.0]),))
    def test_dynamic_axis_fallback(self, v1, axis, expected):
        tensor = tf.compat.v1.placeholder(tf.float32, shape=(None))
        tf_axis = tf.compat.v1.placeholder(tf.int32, shape=(None))
        out = tf.reduce_prod(tensor, tf_axis)
        sess_fn = lambda sess: sess.run((out,),
                                        feed_dict={
                                            tensor: v1,
                                            tf_axis: axis
                                        })
        if not np.allclose(self.with_ngraph(sess_fn), expected):
            raise AssertionError

    @pytest.mark.parametrize(("v1", "axis", "expected"),
                             (([[2.0, 2.0]], 1, [[4.0]]),))
    def test_keep_dims_fallback(self, v1, axis, expected):
        tensor = tf.compat.v1.placeholder(tf.float32, shape=(None))
        out = tf.reduce_prod(tensor, axis, keepdims=True)
        sess_fn = lambda sess: sess.run((out,), feed_dict={tensor: v1})
        result = self.with_ngraph(sess_fn)
        if not np.allclose(len(np.array(result[0].shape)), len(np.array(v1).shape)):
            raise AssertionError
        if not np.allclose(result, expected):
            raise AssertionError
