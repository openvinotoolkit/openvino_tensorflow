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
"""nGraph TensorFlow bridge cast operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf

from common import NgraphTest


class TestCastOperations(NgraphTest):

    def test_cast_1d(self):
        val = tf.placeholder(tf.float32, shape=(2,))
        out = tf.cast(val, dtype=tf.int32)

        def run_test(sess):
            return sess.run(out, feed_dict={val: (5.5, 2.0)})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_cast_2d(self):
        test_input = ((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
        val = tf.placeholder(tf.float32, shape=(2, 3))
        out = tf.cast(val, dtype=tf.int32)

        def run_test(sess):
            return sess.run(out, feed_dict={val: test_input})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
