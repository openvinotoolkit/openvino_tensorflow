# ==============================================================================
#  Copyright 2019 Intel Corporation
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
"""nGraph TensorFlow bridge floor operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf

from common import NgraphTest


class TestTopKV2(NgraphTest):

    def test_topkv2_1d(self):
        input = [1.0, 5.0, 6.0, 12.0]
        p = tf.placeholder(dtype=tf.float32)
        out = tf.nn.top_k(p, k=3, sorted=True)

        def run_test(sess):
            return sess.run(out, feed_dict={p: input}).values

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_topkv2_2d(self):
        input = [[40.0, 30.0, 20.0, 10.0], [10.0, 20.0, 15.0, 70.0]]
        p = tf.placeholder(dtype=tf.float32)
        out = tf.nn.top_k(p, k=3, sorted=True)

        def run_test(sess):
            return sess.run(out, feed_dict={p: input}).values

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_topkv2_3d(self):
        input = [[[40.0, 30.0, 20.0], [20.0, 15.0, 70.0]],
                 [[45.0, 25.0, 43.0], [24.0, 12.0, 7.0]]]
        p = tf.placeholder(dtype=tf.float32)
        out = tf.nn.top_k(p, k=3, sorted=True)

        def run_test(sess):
            return sess.run(out, feed_dict={p: input}).values

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    @pytest.mark.skip(
        reason="Falls back to TF, sorted=False is not supported currently")
    def test_topkv2_nosort(self):
        input = [[40.0, 30.0, 20.0, 10.0], [10.0, 20.0, 15.0, 70.0]]
        p = tf.placeholder(dtype=tf.float32)
        out = tf.nn.top_k(p, k=3, sorted=False)

        def run_test(sess):
            return sess.run(out, feed_dict={p: input}).values

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
