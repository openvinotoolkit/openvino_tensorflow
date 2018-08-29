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
"""nGraph TensorFlow bridge slice operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

from common import NgraphTest


class TestSliceOperations(NgraphTest):
  def test_slice(self):
    inp = np.random.rand(4, 4).astype("f")
    slice_ts = []
    expected = []

    a = np.array([float(x) for x in inp.ravel(order="C")])
    a.shape = (4, 4)

    x = tf.placeholder(dtype=dtypes.float32)
    slice_ts.append(array_ops.slice(x, [0, 0], [2, 2]))
    slice_ts.append(array_ops.slice(x, [0, 0], [-1, -1]))
    slice_ts.append(array_ops.slice(x, [2, 2], [-1, -1]))

    def run_test(sess):
      return sess.run(slice_ts,feed_dict={ x: a })

    slice_vals = self.with_ngraph(run_test)

    expected.append(inp[:2, :2])
    expected.append(inp[:, :])
    expected.append(inp[2:, 2:])

    for v, e in zip(slice_vals, expected):
      np.testing.assert_array_equal(v, e)

  def test_strided_slice(self):
    inp = np.random.rand(4, 5).astype("f")
    slice_ts = []
    expected = []

    a = np.array([float(x) for x in inp.ravel(order="C")])
    a.shape = (4, 5)

    x = tf.placeholder(dtype=dtypes.float32)
    slice_ts.append(x[:])
    slice_ts.append(x[...])
    slice_ts.append(x[:, :])
    slice_ts.append(x[:, ...])
    slice_ts.append(x[1:, :-2])
    slice_ts.append(x[::2, :-2])
    slice_ts.append(x[1, :])
    #slice_ts.append(x[:, 1])
    slice_ts.append(x[1, 1])
    slice_ts.append(x[0])
    slice_ts.append(x[0][1])
    slice_ts.append(x[-1])

    def run_test(sess):
      return sess.run(slice_ts, feed_dict={ x: a })

    slice_vals = self.with_ngraph(run_test)

    expected.append(inp[:])
    expected.append(inp[...])
    expected.append(inp[:, :])
    expected.append(inp[:, ...])
    expected.append(inp[1:, :-2])
    expected.append(inp[::2, :-2])
    expected.append(inp[1, :])
    #expected.append(inp[:, 1])
    expected.append(inp[1, 1])
    expected.append(inp[0])
    expected.append(inp[0][1])
    expected.append(inp[-1])

    for v, e in zip(slice_vals, expected):
      np.testing.assert_array_equal(v, e)
