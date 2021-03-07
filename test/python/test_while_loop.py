# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow log operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestWhileLoop(NgraphTest):

    def test_while_loop(self):
        # Simple example taken from TF docs for tf.while
        i = tf.constant(0)
        c = lambda i: tf.less(i, 10)
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])

        # We'll need soft placement here
        cfg = tf.compat.v1.ConfigProto(allow_soft_placement=True)

        with tf.compat.v1.Session(config=cfg) as sess:
            sess_fn = lambda sess: sess.run((r,))
            result = self.with_ngraph(sess_fn)
            assert result[0] == [10]
