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
"""nGraph TensorFlow bridge AvgPoolBackprop operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.gen_nn_ops import avg_pool_grad
import numpy as np

from common import NgraphTest


class TestAvgPoolBackpropInput(NgraphTest):

    forward_arg_shape_NHWC = [128, 224, 224, 3]
    forward_arg_shape_NCHW = [128, 3, 224, 224]
    ksize = [1, 2, 3, 1]
    strides = [1, 2, 3, 1]
    grad_input_nhwc = {
        "VALID": np.random.rand(128, 112, 74, 3),
        "SAME": np.random.rand(128, 112, 75, 3)
    }
    grad_input_nchw = {
        "VALID": np.random.rand(128, 3, 74, 224),
        "SAME": np.random.rand(128, 3, 75, 224)
    }

    @pytest.mark.parametrize("padding", ("VALID", "SAME"))
    def test_nhwc(self, padding):
        np_nhwc = self.grad_input_nhwc[padding]
        if padding == "VALID":
            grad_input = tf.placeholder(tf.float32, shape=(128, 112, 74, 3))
        elif padding == "SAME":
            grad_input = tf.placeholder(tf.float32, shape=(128, 112, 75, 3))
        out = avg_pool_grad(
            self.forward_arg_shape_NHWC,
            grad_input,
            self.ksize,
            self.strides,
            padding=padding,
            data_format="NHWC")

        def run_test(sess):
            return sess.run(out, feed_dict={grad_input: np_nhwc})

        assert np.isclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test)).all()

    @pytest.mark.parametrize("padding", ("VALID", "SAME"))
    def test_nchw(self, padding):
        self.ksize = [1, 1, 3, 1]
        np_nchw = self.grad_input_nchw[padding]
        if padding == "VALID":
            grad_input = tf.placeholder(tf.float32, shape=(128, 3, 74, 224))
        elif padding == "SAME":
            grad_input = tf.placeholder(tf.float32, shape=(128, 3, 75, 224))

        out_ngtf = avg_pool_grad(
            self.forward_arg_shape_NCHW,
            grad_input,
            self.ksize,
            self.strides,
            padding=padding,
            data_format="NCHW")

        # To validate on the CPU side we will need to run in NHWC, because the CPU
        # implementation of avgpool backprop does not support NCHW. We will
        # transpose on the way in and on the way out
        grad_input_transposed = tf.transpose(grad_input, [0, 2, 3, 1])
        self.ksize = [1, 3, 1, 1]
        self.strides = [1, 3, 1, 2]
        b = avg_pool_grad(
            self.forward_arg_shape_NHWC,
            grad_input_transposed,
            self.ksize,
            self.strides,
            padding=padding,
            data_format="NHWC")
        out_tf = tf.transpose(b, [0, 3, 1, 2])
        assert np.isclose(
            self.with_ngraph(lambda sess: sess.run(
                out_ngtf, feed_dict={grad_input: self.grad_input_nchw[padding]})
                            ),
            self.without_ngraph(lambda sess: sess.run(
                out_tf, feed_dict={grad_input: self.grad_input_nchw[padding]}))
        ).all()
