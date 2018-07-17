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
"""nGraph TensorFlow bridge depthwise_conv2d operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops

from common import NgraphTest


class TestConv2DBackpropInput(NgraphTest):
  INPUT_SIZES_NCHW = [1, 2, 7, 6]
  INPUT_SIZES_NHWC = [1, 7, 6, 2]
  FILTER_IN_SIZES = [3, 3, 2, 2]
  OUT_BACKPROP_IN_SIZES = {"VALID": [1, 2, 3, 2], "SAME": [1, 2, 4, 3]}

  def make_filter_and_backprop_args(self, out_backprop_in_sizes):
    total_size_1 = 1
    total_size_2 = 1

    for s in out_backprop_in_sizes:
      total_size_1 *= s
    for s in self.FILTER_IN_SIZES:
      total_size_2 *= s

    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]

    return x1, x2

  @pytest.mark.parametrize("padding", ("VALID", "SAME"))
  def test_nchw(self, padding):
    # The expected size of the backprop will depend on whether padding is VALID
    # or SAME.
    out_backprop_in_sizes = self.OUT_BACKPROP_IN_SIZES[padding]
    x1, x2 = self.make_filter_and_backprop_args(out_backprop_in_sizes)

    with self.device:
      with self.session as sess:
        t1 = constant_op.constant(self.INPUT_SIZES_NCHW)
        t2 = constant_op.constant(x2, shape=self.FILTER_IN_SIZES)
        t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
        inp = nn_ops.conv2d_backprop_input(
            t1, t2, t3,
            strides=[1, 1, 2, 2], padding=padding, data_format='NCHW')
        value = sess.run(inp)

    # To validate on the CPU side we will need to run in NHWC, because the CPU
    # implementation of conv/conv backprop does not support NCHW. We will
    # transpose on the way in and on the way out.
    with tf.device('/cpu:0'):
      with self.session as sess:
        t1 = constant_op.constant(self.INPUT_SIZES_NHWC)
        t2 = constant_op.constant(x2, shape=self.FILTER_IN_SIZES)
        t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
        t3 = tf.transpose(t3, [0, 2, 3, 1])
        inp = nn_ops.conv2d_backprop_input(
            t1, t2, t3,
            strides=[1, 2, 2, 1], padding=padding, data_format='NHWC')
        inp = tf.transpose(inp, [0, 3, 1, 2])
        expected = sess.run(inp)

    assert (value == expected).all()

  @pytest.mark.parametrize("padding", ("VALID", "SAME"))
  def test_nhwc(self, padding):
    out_backprop_in_sizes = self.OUT_BACKPROP_IN_SIZES[padding]
    x1, x2 = self.make_filter_and_backprop_args(out_backprop_in_sizes)

    with self.device:
      with self.session as sess:
        t1 = constant_op.constant(self.INPUT_SIZES_NHWC)
        t2 = constant_op.constant(x2, shape=self.FILTER_IN_SIZES)
        t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
        t3 = tf.transpose(t3, [0, 2, 3, 1])
        inp = nn_ops.conv2d_backprop_input(
            t1, t2, t3,
            strides=[1, 2, 2, 1], padding=padding, data_format='NHWC')
        value = sess.run(inp)

    with tf.device('/cpu:0'):
      with self.session as sess:
        t1 = constant_op.constant(self.INPUT_SIZES_NHWC)
        t2 = constant_op.constant(x2, shape=self.FILTER_IN_SIZES)
        t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
        t3 = tf.transpose(t3, [0, 2, 3, 1])
        inp = nn_ops.conv2d_backprop_input(
            t1, t2, t3,
            strides=[1, 2, 2, 1], padding=padding, data_format='NHWC')
        expected = sess.run(inp)

    assert (value == expected).all()
