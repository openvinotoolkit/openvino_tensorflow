# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
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

np.random.seed(5)

B = 4
C = 3
A = 112


class TestMaxPoolBackpropInput(NgraphTest):

    # NHWC
    input_nhwc = np.random.rand(B, A, A, C)
    strides_nhwc = ksize_nhwc = [1, 2, 3, 1]
    output_nhwc = {
        "VALID": np.random.rand(B, (A // 2), (A // 3), C),
        "SAME": np.random.rand(B, (A // 2), (A // 3) + 1, C)
    }
    grad_nhwc = {
        "VALID": np.random.rand(B, (A // 2), (A // 3), C),
        "SAME": np.random.rand(B, (A // 2), (A // 3) + 1, C)
    }

    # NCHW
    input_nchw = np.transpose(input_nhwc, NHWC_TO_NCHW)
    strides_nchw = ksize_nchw = [1, 1, 2, C]
    output_nchw = {
        "VALID": np.random.rand(B, C, (A // 2), (A // 3)),
        "SAME": np.random.rand(B, C, (A // 2), (A // 3) + 1)
    }
    grad_nchw = {
        "VALID": np.random.rand(B, C, (A // 2), (A // 3)),
        "SAME": np.random.rand(B, C, (A // 2), (A // 3) + 1)
    }

    @pytest.mark.parametrize("padding", ("VALID", "SAME"))
    def test_nhwc(self, padding):
        strides = self.strides_nhwc
        ksize = self.ksize_nhwc
        output = self.output_nhwc[padding]
        g_nhwc = self.grad_nhwc[padding]
        if padding == "VALID":
            grad = tf.placeholder(tf.float32, shape=(B, (A // 2), (A // 3), C))
        elif padding == "SAME":
            grad = tf.placeholder(
                tf.float32, shape=(B, (A // 2), (A // 3) + 1, C))
        out = max_pool_grad(
            self.input_nhwc,
            output,
            grad,
            ksize,
            strides,
            padding=padding,
            data_format="NHWC")
        sess_fn = lambda sess: sess.run(out, feed_dict={grad: g_nhwc})
        assert (np.allclose(
            self.with_ngraph(sess_fn), self.without_ngraph(sess_fn), rtol=5e-7))

    @pytest.mark.parametrize("padding", ("VALID", "SAME"))
    def test_nchw(self, padding):
        strides = self.strides_nchw
        ksize = self.ksize_nchw
        output = self.output_nchw[padding]
        g_nchw = self.grad_nchw[padding]
        if padding == "VALID":
            grad = tf.placeholder(tf.float32, shape=(B, C, (A // 2), (A // 3)))
        elif padding == "SAME":
            grad = tf.placeholder(
                tf.float32, shape=(B, C, (A // 2), (A // 3) + 1))

        def test_on_ng(sess):
            a = max_pool_grad(
                self.input_nchw,
                output,
                grad,
                ksize,
                strides,
                padding=padding,
                data_format="NCHW")
            return sess.run(a, feed_dict={grad: g_nchw})

        # To validate on the CPU side we will need to run in NHWC, because the CPU
        # implementation of maxpool backprop does not support NCHW. We will
        # transpose on the way in and on the way out
        def test_on_tf(sess):
            grad_t = tf.transpose(grad, NCHW_TO_NHWC)
            ksize = self.ksize_nhwc
            strides = self.strides_nhwc
            input_t = np.transpose(self.input_nchw, NCHW_TO_NHWC)
            output_t = np.transpose(output, NCHW_TO_NHWC)
            b = max_pool_grad(
                input_t,
                output_t,
                grad_t,
                ksize,
                strides,
                padding=padding,
                data_format="NHWC")
            b = tf.transpose(b, NHWC_TO_NCHW)
            return sess.run(b, feed_dict={grad: g_nchw})

        assert np.allclose(
            self.with_ngraph(test_on_ng), self.without_ngraph(test_on_tf))
