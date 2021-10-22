# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow pad operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import openvino_tensorflow
from common import NgraphTest

np.random.seed(5)


class TestSharedConst(NgraphTest):

    def test_sharedconst1(self):
        # The const in this case is shared between Pad1 and Pad2
        # separated by Abs which is not encapsulated. The Pad op
        # is different in the sense that the shared const is a
        # static input to it and is not used by the op directly
        # but is used to create 2 new Consts.

        openvino_tensorflow.set_disabled_ops('Abs')
        input_data1 = tf.compat.v1.placeholder(tf.float32, shape=(2, 3))
        paddings = tf.compat.v1.constant([[2, 1], [2, 2]])

        pad1 = tf.pad(input_data1, paddings)
        abs2 = tf.abs(pad1)
        pad2 = tf.pad(abs2, paddings)

        inp = ((4, 2, 4), (4, 4, 1))
        pad = ((5, 3), (5, 5))

        def run_test(sess):
            return sess.run(pad2, feed_dict={input_data1: inp})

        if not (self.with_ngraph(run_test) == self.without_ngraph(run_test)
               ).all():
            raise AssertionError

        # Clean up
        openvino_tensorflow.set_disabled_ops('')
