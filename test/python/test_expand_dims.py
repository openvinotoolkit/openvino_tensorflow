# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow ExpandDims test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestExpandDims(NgraphTest):

    def test_expand_dims_float64(self):
        tensor = tf.compat.v1.placeholder(np.float64, (10, 10))
        outputs = tf.raw_ops.ExpandDims(input=tensor, axis=0)
        feed_dict = {tensor: np.random.randn(10, 10)}

        sess_fn = lambda sess: sess.run((outputs), feed_dict=feed_dict)

        if not np.allclose(
                self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)):
            raise AssertionError
