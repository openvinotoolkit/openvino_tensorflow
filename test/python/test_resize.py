# ==============================================================================
# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow resizebilinear, resizenearestneighbor test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestResize(NgraphTest):

    def test_resizebilinear(self):
        rng = np.random.default_rng(2021)
        x = rng.random((32, 416, 416, 3))
        images = tf.compat.v1.placeholder(np.float32, (32, 416, 416, 3))

        size = tf.compat.v1.constant([224, 224], dtype=tf.int32)

        resize_bl = tf.compat.v1.image.resize_bilinear(
            images,
            size,
            align_corners=False,
            name=None,
            half_pixel_centers=False)
        sess_fn = lambda sess: sess.run((resize_bl), feed_dict={images: x})
        np.allclose(self.with_ngraph(sess_fn), self.without_ngraph(sess_fn))

    def test_resizenearestneighbor(self):
        rng = np.random.default_rng(2021)
        x = rng.random((32, 224, 224, 3))
        images = tf.compat.v1.placeholder(np.float32, (32, 224, 224, 3))

        size = tf.compat.v1.constant([416, 416], dtype=tf.int32)

        resize_nn = tf.compat.v1.image.resize_nearest_neighbor(
            images,
            size,
            align_corners=False,
            name=None,
            half_pixel_centers=False)
        sess_fn = lambda sess: sess.run((resize_nn), feed_dict={images: x})
        np.allclose(self.with_ngraph(sess_fn), self.without_ngraph(sess_fn))
