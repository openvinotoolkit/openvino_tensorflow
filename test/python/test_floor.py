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
        assert np.isclose(
            self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)).all()

    def test_floor_2d(self):
        test_input = ((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
        expected = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        val = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))
        out = tf.floor(val)
        assert np.isclose(
            self.with_ngraph(lambda sess: sess.run(
                out, feed_dict={val: test_input})), np.array(expected)).all()
