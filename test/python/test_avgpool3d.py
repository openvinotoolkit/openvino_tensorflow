# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functional tests for 3d pooling operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.gen_array_ops import pack_eager_fallback
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from common import NgraphTest


def GetTestConfigs():
    """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
    test_configs = [("NDHWC", False), ("NDHWC", True)]
    if test.is_gpu_available(cuda_only=True):
        # "NCHW" format is currently supported exclusively on CUDA GPUs.
        test_configs += [("NCDHW", True)]
    return test_configs


# TODO(mjanusz): Add microbenchmarks for 3d pooling.
class TestMaxPool3D(NgraphTest, test.TestCase):

    def _VerifyOneTest(self, pool_func, input_sizes, window, strides, padding,
                       data_format, expected, use_gpu):
        """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called: co.MaxPool, co.AvgPool.
      input_sizes: Input tensor dimensions.
      window: Tuple of kernel dims: planes, rows, cols.
      strides: Tuple of strides for dims: planes, rows, cols.
      padding: Padding type.
      data_format: The data format we use to run the pooling operation.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether to run ops on GPU.
    """
        total_size = 1
        for s in input_sizes:
            total_size *= s
        # Initializes the input tensor with array containing incrementing
        # numbers from 1.
        x = [f * 1.0 for f in range(1, total_size + 1)]
        with self.cached_session(use_gpu=use_gpu) as sess:
            t = constant_op.constant(x, shape=input_sizes).eval()
            window = [1] + list(window) + [1]
            strides = [1] + list(strides) + [1]
            temp = tf.compat.v1.placeholder(np.float32, shape=t.shape)
            p = pool_func(
                temp,
                ksize=window,
                strides=strides,
                padding=padding,
                data_format=data_format)
            sess_fn = lambda sess: sess.run(p, feed_dict={temp: t})
            vals = self.with_ngraph(sess_fn)
        # Verifies values.
        actual = vals.flatten()
        self.assertAllClose(expected, actual)

    def _VerifyValues(self, pool_func, input_sizes, window, strides, padding,
                      expected):
        for data_format, use_gpu in GetTestConfigs():
            self._VerifyOneTest(pool_func, input_sizes, window, strides,
                                padding, data_format, expected, use_gpu)

    def testAvgPool3dValidPadding(self):
        expected_output = [20.5, 21.5, 22.5]
        self._VerifyValues(
            nn_ops.avg_pool3d,
            input_sizes=[1, 3, 3, 3, 3],
            window=(2, 2, 2),
            strides=(2, 2, 2),
            padding="VALID",
            expected=expected_output)

    def testAvgPool3dSamePadding(self):
        expected_output = [20.5, 21.5, 22.5, 26.5, 27.5, 28.5]
        self._VerifyValues(
            nn_ops.avg_pool3d,
            input_sizes=[1, 2, 2, 4, 3],
            window=(2, 2, 2),
            strides=(2, 2, 2),
            padding="SAME",
            expected=expected_output)

    def testAvgPool3dSamePaddingDifferentStrides(self):
        expected_output = [1.5, 4.5, 7.5, 17.5, 20.5, 23.5, 33.5, 36.5, 39.5]
        self._VerifyValues(
            nn_ops.avg_pool3d,
            input_sizes=[1, 5, 8, 1, 1],
            window=(1, 2, 3),
            strides=(2, 3, 1),
            padding="SAME",
            expected=expected_output)
