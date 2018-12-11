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
import tensorflow as tf

import numpy as np
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
            return sess.run(slice_ts, feed_dict={x: a})

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
        slice_ts.append(x[:, :])
        slice_ts.append(x[1:, :-2])
        slice_ts.append(x[::2, :-2])
        slice_ts.append(x[1, :])
        slice_ts.append(x[:, 1])
        slice_ts.append(x[1, 1])
        slice_ts.append(x[0])
        slice_ts.append(x[0][1])
        slice_ts.append(x[-1])

        # Various ways of representing identity slice
        slice_ts.append(x[:, :])
        slice_ts.append(x[::, ::])
        slice_ts.append(x[::1, ::1])

        # Reverse in each dimension independently
        slice_ts.append(x[::-1, :])
        slice_ts.append(x[:, ::-1])

        ## negative index tests i.e. n-2 in first component
        slice_ts.append(x[-2::-1, ::1])

        # degenerate by offering a forward interval with a negative stride
        slice_ts.append(x[0:-1:-1, :])
        # degenerate with a reverse interval with a positive stride
        slice_ts.append(x[-1:0, :])
        # empty interval in every dimension
        slice_ts.append(x[-1:0, 2:3:-1])
        slice_ts.append(x[2:2, 2:3:-1])
        # stride greater than range
        slice_ts.append(x[1:3:7, :])

        # Unsupported on ngraph currently
        # slice_ts.append(x[:, tf.newaxis])
        # slice_ts.append(x[...])
        # slice_ts.append(x[:, ...])

        def run_test(sess):
            return sess.run(slice_ts, feed_dict={x: a})

        slice_vals = self.with_ngraph(run_test)

        expected.append(inp[:])
        expected.append(inp[:, :])
        expected.append(inp[1:, :-2])
        expected.append(inp[::2, :-2])
        expected.append(inp[1, :])
        expected.append(inp[:, 1])
        expected.append(inp[1, 1])
        expected.append(inp[0])
        expected.append(inp[0][1])
        expected.append(inp[-1])
        #TODO: support ellipses and new_axis correctly

        # Various ways of representing identity slice
        expected.append(inp[:, :])
        expected.append(inp[::, ::])
        expected.append(inp[::1, ::1])

        # Reverse in each dimension independently
        expected.append(inp[::-1, :])
        expected.append(inp[:, ::-1])

        ## negative index tests i.e. n-2 in first component
        expected.append(inp[-2::-1, ::1])

        # degenerate by offering a forward interval with a negative stride
        expected.append(inp[0:-1:-1, :])
        # degenerate with a reverse interval with a positive stride
        expected.append(inp[-1:0, :])
        # empty interval in every dimension
        expected.append(inp[-1:0, 2:3:-1])
        expected.append(inp[2:2, 2:3:-1])
        # stride greater than range
        expected.append(inp[1:3:7, :])

        # Unsupported on ngraph currently
        #expected.append(inp[...])
        #expected.append(inp[:, ...])
        # expected.append(inp[:, tf.newaxis])
        for v, e in zip(slice_vals, expected):
            np.testing.assert_array_equal(v, e)

    def test_strided_slice_zerodim(self):
        inp = np.random.rand(4, 0, 5).astype("f")
        slice_ts = []
        expected = []

        a = np.array([float(x) for x in inp.ravel(order="C")])
        a.shape = (4, 0, 5)

        x = tf.placeholder(dtype=dtypes.float32)

        #(slicing an empty dim by empty slice)
        slice_ts.append(x[1:2, 2:2, 1:2])
        #(slicing an empty dim by non empty slice)
        slice_ts.append(x[1:2, 1:2, 1:2])

        def run_test(sess):
            return sess.run(slice_ts, feed_dict={x: a})

        slice_vals = self.with_ngraph(run_test)

        expected.append(inp[1:2, 2:2, 1:2])
        expected.append(inp[1:2, 1:2, 1:2])

        for v, e in zip(slice_vals, expected):
            np.testing.assert_array_equal(v, e)

    def test_incorrect_strided_slice(self):
        inp = 0
        slice_ts = []

        x = tf.placeholder(dtype=dtypes.float32)

        #(slicing an empty dim by empty slice)
        slice_ts.append(x[1:1])

        def run_test(sess):
            return sess.run(slice_ts, feed_dict={x: inp})

        with pytest.raises(Exception) as excinfo:
            slice_vals = self.with_ngraph(run_test)
        assert "Index out of range using input dim 1; input has only 0 dims" in excinfo.value.message
