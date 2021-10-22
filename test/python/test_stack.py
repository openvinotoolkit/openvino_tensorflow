# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow stack operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pytest

from common import NgraphTest


class TestStackOperations(NgraphTest):

    @pytest.mark.parametrize(
        ("shapes", "axis"),
        (
            ([(1, 2), (1, 2)], 2),
            ([(3, 2), (3, 2)], 0),
            ([(2, 3), (2, 3)], 1),
            ([(1, 2, 4, 5), (1, 2, 4, 5), (1, 2, 4, 5)], 3),
            ([(1, 2, 3, 5), (1, 2, 3, 5), (1, 2, 3, 5)], 4),
            ([(3, 3, 7, 3), (3, 3, 7, 3)], -1),
            # Disabling the following test cases for now as there are some
            # changes need to be done in the framework
            #([(0)], 0),
            #([(1)], 1)
        ))
    def test_stack(self, shapes, axis):
        values = [np.random.random_sample(s) for s in shapes]
        expected = np.stack(values, axis)
        placeholders = [tf.compat.v1.placeholder(tf.float32, s) for s in shapes]
        a = tf.stack(placeholders, axis)
        sess_fn = lambda sess: sess.run(
            [a], feed_dict={p: v for p, v in zip(placeholders, values)})
        if not np.allclose(
                self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)):
            raise AssertionError
