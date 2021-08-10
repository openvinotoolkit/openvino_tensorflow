# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow Reverse test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestReverse(NgraphTest):

    def test_reverse(self):

        t = tf.compat.v1.placeholder(tf.float32, shape=(1, 2, 3, 4))
        axis = tf.compat.v1.placeholder(tf.int32, shape=(1))

        t_np = [[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                 [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]]
        axis_np = [-1]

        reversed = tf.reverse(tensor=t, axis=axis)

        sess_fn = lambda sess: sess.run((reversed,),
                                        feed_dict={
                                            t: t_np,
                                            axis: axis_np
                                        })[0]

        np.allclose(self.with_ngraph(sess_fn), self.without_ngraph(sess_fn))
