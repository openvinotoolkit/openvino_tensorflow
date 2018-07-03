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

import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

from common import NgraphTest


class TestSliceOperations(NgraphTest):
  def test_slice_and_strided_slice(self):
    with self.device:
      inp = np.random.rand(4, 4).astype("f")
      with self.session as sess:
        a = constant_op.constant(
            [float(x) for x in inp.ravel(order="C")],
            shape=[4, 4],
            dtype=dtypes.float32)
        # slice
        slice_t = array_ops.slice(a, [0, 0], [2, 2])
        # strided slice
        slice2_t = a[:2, :2]

        slice_val, slice2_val = sess.run([slice_t, slice2_t])

    np.testing.assert_array_equal(slice_val, inp[:2, :2])
    np.testing.assert_array_equal(slice2_val, inp[:2, :2])

  def test_slice_and_strided_slice_neg(self):
    with self.device:
      inp = np.random.rand(4, 4).astype("f")
      with self.session as sess:
        a = constant_op.constant(
            [float(x) for x in inp.ravel(order="C")],
            shape=[4, 4],
            dtype=dtypes.float32)
        # slice
        slice_t = array_ops.slice(a, [0, 0], [-1, -1])
        # strided slice
        slice2_t = a[:-1, :-1]

        slice_val, slice2_val = sess.run([slice_t, slice2_t])

    np.testing.assert_array_equal(slice_val, inp[:4, :4])
    np.testing.assert_array_equal(slice2_val, inp[:4, :4])
