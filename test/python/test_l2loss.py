# ==============================================================================
# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow L2loss test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pytest

from common import NgraphTest

np.random.seed(5)


class TestL2Loss(NgraphTest):

    @pytest.mark.parametrize(("xshape"), ((3, 4, 5), (1,)))
    def test_l2loss(self, xshape):
        x = tf.compat.v1.placeholder(tf.float32, shape=xshape)
        out = tf.nn.l2_loss(x)
        values = np.random.rand(*xshape)
        sess_fn = lambda sess: sess.run((out), feed_dict={x: values})
        if not np.allclose(
                self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)):
            raise AssertionError

    def test_l2loss_empty(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=())
        out = tf.nn.l2_loss(x)
        sess_fn = lambda sess: sess.run((out), feed_dict={x: None})

        # expect to be nan
        if not (self.with_ngraph(sess_fn) != self.without_ngraph(sess_fn)):
            raise AssertionError
