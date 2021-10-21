# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow NMSV2 operation test

"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pytest

from common import NgraphTest


class TestNMSOperations(NgraphTest):

    def test_NMSV2(self):

        boxes = tf.compat.v1.placeholder(tf.float32, shape=(6, 4))
        scores = tf.compat.v1.placeholder(tf.float32, shape=(6))
        max_output_size = tf.compat.v1.placeholder(tf.int32, shape=(None))

        boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                    [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
        scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
        max_output_size_np = 3

        nmsv2 = tf.raw_ops.NonMaxSuppressionV2(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=0.5)

        def run_test(sess):
            return sess.run(
                (nmsv2,),
                feed_dict={
                    boxes: boxes_np,
                    scores: scores_np,
                    max_output_size: max_output_size_np
                })

        if not np.allclose(self.without_ngraph(run_test), self.with_ngraph(run_test)):
            raise AssertionError

    def test_NMSV3(self):

        boxes = tf.compat.v1.placeholder(tf.float32, shape=(6, 4))
        scores = tf.compat.v1.placeholder(tf.float32, shape=(6))
        max_output_size = tf.compat.v1.placeholder(tf.int32, shape=(None))

        boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                    [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
        scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
        max_output_size_np = 3

        nmsv3 = tf.raw_ops.NonMaxSuppressionV3(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=0.5,
            score_threshold=0.59)

        def run_test(sess):
            return sess.run(
                (nmsv3,),
                feed_dict={
                    boxes: boxes_np,
                    scores: scores_np,
                    max_output_size: max_output_size_np
                })

        if not np.allclose(self.without_ngraph(run_test), self.with_ngraph(run_test)):
            raise AssertionError
