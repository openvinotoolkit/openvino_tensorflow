# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow split operation test

"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pytest

from common import NgraphTest


class TestLogSoftmaxOperations(NgraphTest):

    def test_logsoftmax(self):
        type = np.float32
        max = np.finfo(type).max
        features = np.array([[1., 1., 1., 1.], [max, 1., 2., 3.]]).astype(type)
        logsoftmax = tf.nn.log_softmax(features)
        sess_fn = lambda sess: sess.run([logsoftmax])
        out = self.with_ngraph(sess_fn)
        if not np.allclose(
                np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                          [0, -max, -max, -max]]),
                out,
                rtol=1.e-5,
                atol=1.e-5):
            raise AssertionError
