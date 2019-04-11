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


class TestSelect(NgraphTest):

    def test_select_scalar(self):
        a = [1.5]
        p = tf.placeholder(dtype=tf.bool)
        out = tf.where(p, x=[1], y=[0])

        def run_test(sess):
            return sess.run(out, feed_dict={p: a})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_select_sameshape(self):
        a = [True, False, True, True]
        p = tf.placeholder(dtype=tf.bool)
        out = tf.where(p, x=[1] * 4, y=[0] * 4)

        def run_test(sess):
            return sess.run(out, feed_dict={p: a})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_select_diffrank(self):
        a = [1, 1]
        x = [[0, 0], [2, 2]]
        y = [[2, 2], [1, 1]]
        p = tf.placeholder(dtype=tf.bool)
        out = tf.where(p, x, y)

        def run_test(sess):
            return sess.run(out, feed_dict={p: a})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_select_complexshape1(self):
        a = np.random.randint(2, size=[7])
        x = np.random.uniform(0, 11, [7, 3, 2, 1])

        p = tf.placeholder(dtype=tf.bool)
        out = tf.where(p, x, x)

        def run_test(sess):
            return (sess.run(out, feed_dict={p: a}))

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_select_complexshape2(self):
        a = np.random.randint(2, size=[7])
        x = np.random.uniform(0, 11, [7, 3, 2, 7])

        p = tf.placeholder(dtype=tf.bool)
        out = tf.where(p, x, x)

        def run_test(sess):
            return (sess.run(out, feed_dict={p: a}))

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_select_complexshape3(self):
        a = np.random.randint(2, size=[5])
        x = np.random.uniform(0, 11, [5, 3, 1])

        p = tf.placeholder(dtype=tf.bool)
        out = tf.where(p, x, x)

        def run_test(sess):
            return (sess.run(out, feed_dict={p: a}))

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
