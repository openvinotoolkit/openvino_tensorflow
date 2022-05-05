# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow CTCGreedyDecoder operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops

tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestCTCGreedyDecoder(NgraphTest):

    def _decode(probs, seq_lens, blank_index=None):
        probs = tf.compat.v1.transpose(probs, perm=[1, 0, 2])
        log_probs = tf.math.log(probs + 1e-7)
        decoded_outputs, decoded_log_probs = tf.nn.ctc_greedy_decoder(
            inputs=log_probs, sequence_length=seq_lens, blank_index=blank_index)

        return (decoded_outputs[0].indices, decoded_outputs[0].values,
                decoded_outputs[0].dense_shape, decoded_log_probs)

    def test_all_invalid(self):
        inputs = np.array([
            [
                [0.1, 0.3, 0.6],  # t=0
                [0.1, 0.2, 0.7],  # t=1
                [0.3, 0.2, 0.5]  # t=2
            ],
            [
                [0.1, 0.3, 0.6],  # t=0
                [0.1, 0.2, 0.7],  # t=1
                [0.3, 0.2, 0.5]  # t=2
            ],
            [
                [0.1, 0.3, 0.6],  # t=0
                [0.1, 0.2, 0.7],  # t=1
                [0.3, 0.2, 0.5]  # t=2
            ]
        ])
        samples, max_time_steps, classes = inputs.shape

        probs = tf.compat.v1.placeholder(
            tf.float32, shape=(samples, max_time_steps, classes))
        seq_lens = tf.compat.v1.placeholder(tf.int32, shape=(samples,))
        out = TestCTCGreedyDecoder._decode(probs, seq_lens)

        feed_dict = {probs: inputs, seq_lens: [3, 3, 3]}

        def run_test(sess):
            return sess.run(out, feed_dict=feed_dict)

        *ng_decoded_outputs, ng_decoded_log_probs = self.with_ngraph(run_test)
        *tf_decoded_outputs, tf_decoded_log_probs = self.without_ngraph(
            run_test)

        for x, y in zip(ng_decoded_outputs, tf_decoded_outputs):
            if not (x == y).all():
                raise AssertionError

        if not np.allclose(
                ng_decoded_log_probs, tf_decoded_log_probs, atol=1e-5):
            raise AssertionError

    def test_one_invalid(self):
        inputs = np.array([
            [
                [0.1, 0.3, 0.6],  # t=0
                [0.1, 0.2, 0.7],  # t=1
                [0.3, 0.2, 0.5]  # t=2
            ],
            [  # valid sample
                [0.6, 0.1, 0.3],  # t=0
                [0.1, 0.2, 0.7],  # t=1
                [0.5, 0.3, 0.2]  # t=2
            ],
            [
                [0.1, 0.3, 0.6],  # t=0
                [0.1, 0.2, 0.7],  # t=1
                [0.3, 0.2, 0.5]  # t=2
            ]
        ])
        samples, max_time_steps, classes = inputs.shape

        probs = tf.compat.v1.placeholder(
            tf.float32, shape=(samples, max_time_steps, classes))
        seq_lens = tf.compat.v1.placeholder(tf.int32, shape=(samples,))
        out = TestCTCGreedyDecoder._decode(probs, seq_lens)

        feed_dict = {probs: inputs, seq_lens: [3, 3, 3]}

        def run_test(sess):
            return sess.run(out, feed_dict=feed_dict)

        *ng_decoded_outputs, ng_decoded_log_probs = self.with_ngraph(run_test)
        *tf_decoded_outputs, tf_decoded_log_probs = self.without_ngraph(
            run_test)

        for x, y in zip(ng_decoded_outputs, tf_decoded_outputs):
            if not (x == y).all():
                raise AssertionError

        if not np.allclose(
                ng_decoded_log_probs, tf_decoded_log_probs, atol=1e-5):
            raise AssertionError
