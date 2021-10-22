# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow CropAndResize operation test

"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pytest

from common import NgraphTest


class TestCropAndResize(NgraphTest):

    def test_crop_and_resize_downscale_bilinear(self):

        BATCH_SIZE = 4
        NUM_BOXES = 15
        IMAGE_HEIGHT = 1024
        IMAGE_WIDTH = 768
        CHANNELS = 3
        CROP_SIZE = (64, 32)

        image = np.random.normal(
            size=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        boxes = np.random.uniform(size=(NUM_BOXES, 4))
        box_indices = np.random.uniform(
            size=(NUM_BOXES,), low=0, high=BATCH_SIZE)
        output = tf.image.crop_and_resize(
            image, boxes, box_indices, CROP_SIZE, method="bilinear")

        def run_test(sess):
            return sess.run((output,))

        if not np.allclose(
                self.without_ngraph(run_test), self.with_ngraph(run_test), 1e-5,
                1e-6):
            raise AssertionError

    def test_crop_and_resize_upscale_nearest(self):

        BATCH_SIZE = 7
        NUM_BOXES = 13
        IMAGE_HEIGHT = 24
        IMAGE_WIDTH = 24
        CHANNELS = 3
        CROP_SIZE = (128, 128)

        image = np.random.normal(
            size=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        boxes = np.random.uniform(size=(NUM_BOXES, 4))
        box_indices = np.random.uniform(
            size=(NUM_BOXES,), low=0, high=BATCH_SIZE)
        output = tf.image.crop_and_resize(
            image, boxes, box_indices, CROP_SIZE, method="nearest")

        def run_test(sess):
            return sess.run((output,))

        if not np.allclose(
                self.without_ngraph(run_test), self.with_ngraph(run_test), 1e-5,
                1e-6):
            raise AssertionError
