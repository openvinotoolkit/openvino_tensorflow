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

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.platform import test

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from common import NgraphTest


class TestFloorOperations(NgraphTest):

    @pytest.mark.parametrize(("test_input", "expected"),
                             ((1.4, 1.0), (0.5, 0.0), (-0.3, -1.0)))
    def test_floor_1d(self, test_input, expected):
        val = tf.compat.v1.placeholder(tf.float32, shape=(1,))
        out = tf.floor(val)
        sess_fn = lambda sess: sess.run(out, feed_dict={val: (test_input,)})
        if not np.isclose(
                self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)).all():
            raise AssertionError

    def test_floor_2d(self):
        test_input = ((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
        expected = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        val = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))
        out = tf.floor(val)
        assert np.isclose(
            self.with_ngraph(lambda sess: sess.run(
                out, feed_dict={val: test_input})), np.array(expected)).all()


class TestFloorOperations2(test.TestCase, NgraphTest):

    def _compare(self, x, dtype):
        np_floor, np_ceil = np.floor(x), np.ceil(x)

        inx = ops.convert_to_tensor(x)
        inx = tf.compat.v1.placeholder(dtype, shape=inx.shape)
        tf_floor = lambda sess: sess.run(
            math_ops.floor(inx), feed_dict={inx: x})

        self.assertAllEqual(np_floor, self.with_ngraph(tf_floor))

    def _testDtype(self, dtype):
        data = (np.arange(-3, 3) / 4.).reshape(1, 3, 2).astype(dtype)
        self._compare(data, dtype)

    def testTypes(self):
        for dtype in [
                np.float16, np.float32, np.float64,
                dtypes_lib.bfloat16.as_numpy_dtype
        ]:
            with self.subTest(dtype=dtype):
                self._testDtype(dtype)
