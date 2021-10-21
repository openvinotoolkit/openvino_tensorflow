# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow test for dynamic shapes

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import numpy as np
import random
from random import SystemRandom
cryptogen = SystemRandom()

from common import NgraphTest


class TestResizeToDynamicShape(NgraphTest):

    def test_resize_to_dynamic_shape(self):
        # Test input with some arbitrary shape.
        test_input = np.random.rand(128, 10, 10, 20, 5)
        val = tf.compat.v1.placeholder(tf.float32, shape=(128, 10, 10, 20, 5))

        # Reshape to a random permutation of the input shape. We use a fixed seed
        # so that we get same results on CPU and nGraph, and we have to do some
        # hackery to make sure the actual op survives constant folding.
        seed = cryptogen.randint(0, 999999)
        shuffled_shape = tf.compat.v1.random_shuffle(tf.shape(val), seed=seed)
        out = tf.reshape(val, shuffled_shape)

        def run_test(sess):
            return sess.run(out, feed_dict={val: test_input})

        # Disable as much optimization as we can.
        config = tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                optimizer_options=tf.compat.v1.OptimizerOptions(
                    opt_level=tf.compat.v1.OptimizerOptions.L0,
                    do_common_subexpression_elimination=False,
                    do_constant_folding=False,
                    do_function_inlining=False,
                )))

        if not (self.without_ngraph(run_test, config) == self.with_ngraph(run_test, config)).all():
            raise AssertionError
