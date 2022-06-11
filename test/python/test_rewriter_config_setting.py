# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow test for checking backend setting using rewriter config for grappler

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import os
import numpy as np
import shutil
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.core.protobuf import rewriter_config_pb2
import openvino_tensorflow

from common import NgraphTest


class TestRewriterConfigBackendSetting(NgraphTest):

    def test_config_updater_api(self):
        dim1 = 3
        dim2 = 4
        a = tf.compat.v1.placeholder(tf.float32, shape=(dim1, dim2), name='a')
        x = tf.compat.v1.placeholder(tf.float32, shape=(dim1, dim2), name='x')
        b = tf.compat.v1.placeholder(tf.float32, shape=(dim1, dim2), name='y')
        axpy = (a * x) + b

        config = tf.compat.v1.ConfigProto()
        rewriter_options = rewriter_config_pb2.RewriterConfig()
        rewriter_options.meta_optimizer_iterations = (
            rewriter_config_pb2.RewriterConfig.ONE)
        rewriter_options.min_graph_nodes = -1
        ovtf_optimizer = rewriter_options.custom_optimizers.add()
        ovtf_optimizer.name = "ovtf-optimizer"
        config.MergeFrom(
            tf.compat.v1.ConfigProto(
                graph_options=tf.compat.v1.GraphOptions(
                    rewrite_options=rewriter_options)))

        with tf.compat.v1.Session(config=config) as sess:
            outval = sess.run(
                axpy,
                feed_dict={
                    a: 1.5 * np.ones((dim1, dim2)),
                    b: np.ones((dim1, dim2)),
                    x: np.ones((dim1, dim2))
                })
        if not (outval == 2.5 * (np.ones((dim1, dim2)))).all():
            raise AssertionError
