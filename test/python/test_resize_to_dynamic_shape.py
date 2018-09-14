# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""nGraph TensorFlow bridge test for dynamic shapes

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import os
import numpy as np
import random

from common import NgraphTest


class TestResizeToDynamicShape(NgraphTest):

    def test_resize_to_dynamic_shape(self):
        # Test input with some arbitrary shape.
        test_input = np.random.rand(128, 10, 10, 20, 5)
        val = tf.placeholder(tf.float32, shape=(128, 10, 10, 20, 5))

        # Reshape to a random permutation of the input shape. We use a fixed seed
        # so that we get same results on CPU and nGraph, and we have to do some
        # hackery to make sure the actual op survives constant folding.
        seed = random.randint(0, 999999)
        shuffled_shape = tf.random_shuffle(tf.shape(val), seed=seed)
        out = tf.reshape(val, shuffled_shape)

        def run_test(sess):
            return sess.run(out, feed_dict={val: test_input})

        # Disable as much optimization as we can.
        config = tf.ConfigProto(
            graph_options=tf.GraphOptions(
                optimizer_options=tf.OptimizerOptions(
                    opt_level=tf.OptimizerOptions.L0,
                    do_common_subexpression_elimination=False,
                    do_constant_folding=False,
                    do_function_inlining=False,
                )))

        assert (self.without_ngraph(run_test, config) == self.with_ngraph(
            run_test, config)).all()
