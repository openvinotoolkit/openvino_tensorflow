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
"""nGraph TensorFlow bridge MaxPoolBackprop operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import max_pool_grad

from common import NgraphTest

NHWC_TO_NCHW = (0, 3, 1, 2)
NCHW_TO_NHWC = (0, 2, 3, 1)


class TestMaxPoolBackpropInput(NgraphTest):
    input_nhwc = np.random.rand(128, 224, 224, 3)
    input_nchw = np.transpose(input_nhwc, NHWC_TO_NCHW)
    output_nhwc = np.random.rand(128, 224, 224, 3)
    output_nchw = np.transpose(output_nhwc, NHWC_TO_NCHW)
    strides_nhwc = ksize_nhwc = [1, 2, 3, 1]
    strides_nchw = ksize_nchw = [1, 1, 2, 3]
    grad_nhwc = {
        "VALID": np.random.rand(128, 112, 74, 3),
        "SAME": np.random.rand(128, 112, 75, 3)
    }
    grad_nchw = {
        "VALID": np.transpose(grad_nhwc["VALID"], NHWC_TO_NCHW),
        "SAME": np.transpose(grad_nhwc["SAME"], NHWC_TO_NCHW)
    }

    @pytest.mark.parametrize("padding", ("VALID", "SAME"))
    def test_nhwc(self, padding):
        strides = self.strides_nhwc
        ksize = self.ksize_nhwc
        output = self.output_nhwc
        np_nhwc = self.grad_nhwc[padding]
        if padding == "VALID":
            grad = tf.placeholder(tf.float32, shape=(128, 112, 74, 3))
        elif padding == "SAME":
            grad = tf.placeholder(tf.float32, shape=(128, 112, 75, 3))

        with self.device:
            a = max_pool_grad(
                self.input_nhwc,
                output,
                grad,
                ksize,
                strides,
                padding=padding,
                data_format="NHWC")
            with self.session as sess:
                result = sess.run(a, feed_dict={grad: np_nhwc})

        with tf.device('/cpu:0'):
            b = max_pool_grad(
                self.input_nhwc,
                output,
                grad,
                ksize,
                strides,
                padding=padding,
                data_format="NHWC")
            with self.session as sess:
                expected = sess.run(b, feed_dict={grad: np_nhwc})

        np.testing.assert_allclose(result, expected, rtol=5e-7)

    @pytest.mark.parametrize("padding", ("VALID", "SAME"))
    def test_nchw(self, padding):
        strides = self.strides_nchw
        ksize = self.ksize_nchw
        output = self.output_nchw
        np_nchw = self.grad_nchw[padding]
        if padding == "VALID":
            grad = tf.placeholder(tf.float32, shape=(128, 3, 112, 74))
        elif padding == "SAME":
            grad = tf.placeholder(tf.float32, shape=(128, 3, 112, 75))

        with self.device:
            a = max_pool_grad(
                self.input_nchw,
                output,
                grad,
                ksize,
                strides,
                padding=padding,
                data_format="NCHW")
            with self.session as sess:
                result = sess.run(a, feed_dict={grad: np_nchw})
        # To validate on the CPU side we will need to run in NHWC, because the CPU
        # implementation of avgpool backprop does not support NCHW. We will
        # transpose on the way in and on the way out
        with tf.device('/cpu:0'):
            grad = tf.transpose(grad, NCHW_TO_NHWC)
            np_nhwc = self.grad_nhwc[padding]
            output = self.output_nhwc
            ksize = self.ksize_nhwc
            strides = self.strides_nhwc
            b = max_pool_grad(
                self.input_nhwc,
                output,
                grad,
                ksize,
                strides,
                padding=padding,
                data_format="NHWC")
            b = tf.transpose(b, NHWC_TO_NCHW)
            with self.session as sess:
                expected = sess.run(b, feed_dict={grad: np_nhwc})

        np.testing.assert_allclose(result, expected, rtol=5e-7)
