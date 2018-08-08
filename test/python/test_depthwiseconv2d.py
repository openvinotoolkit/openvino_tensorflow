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


@pytest.mark.skip(reason="new deviceless mode WIP")
class TestDepthwiseConv2dOperations(NgraphTest):
  @pytest.mark.parametrize("padding", ("VALID", "SAME"))
  def test_depthwise_conv2d(self, padding):
    tensor_in_sizes = [1, 2, 3, 2]
    filter_in_sizes = [2, 2, 2, 2]
    total_size_1 = 1
    total_size_2 = 1

    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s

    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with self.device:
      with tf.Session(config=self.config) as sess:
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t1.set_shape(tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        conv = nn_ops.depthwise_conv2d_native(
            t1, t2, strides=[1, 1, 1, 1], padding=padding)
        value = sess.run(conv)

    with self.session as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t1.set_shape(tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      conv = nn_ops.depthwise_conv2d_native(
          t1, t2, strides=[1, 1, 1, 1], padding=padding)
      expected = sess.run(conv)

    assert (value == expected).all()
