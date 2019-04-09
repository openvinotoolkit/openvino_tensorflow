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
"""nGraph TensorFlow bridge gather operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import os
import numpy as np

from common import NgraphTest


class TestGatherOperations(NgraphTest):

    # Scalar indices
    @pytest.mark.skip(reason="Backend specific test")
    def test_gather_0(self):
        val = tf.placeholder(tf.float32, shape=(5,))
        out = tf.gather(val, 1)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: (10.0, 20.0, 30.0, 40.0, 50.0)})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    # Vector indices, no broadcast required
    @pytest.mark.skip(reason="Backend specific test")
    def test_gather_1(self):
        val = tf.placeholder(tf.float32, shape=(5,))
        out = tf.gather(val, [2, 1])

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: (10.0, 20.0, 30.0, 40.0, 50.0)})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    # Vector indices, broadcast required
    @pytest.mark.skip(reason="Backend specific test")
    def test_gather_2(self):
        val = tf.placeholder(tf.float32, shape=(2, 5))
        out = tf.gather(val, [2, 1], axis=1)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: np.arange(10).reshape([2, 5])})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
        print(self.with_ngraph(run_test))

    # Vector indices, broadcast required, negative axis
    @pytest.mark.skip(reason="Backend specific test")
    def test_gather_3(self):
        val = tf.placeholder(tf.float32, shape=(2, 5))
        out = tf.gather(val, [2, 1], axis=-1)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: np.arange(10).reshape([2, 5])})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
        print(self.with_ngraph(run_test))

    # higher rank indices... not working right now
    @pytest.mark.skip(reason="WIP: higher rank indices")
    def test_gather_4(self):
        val = tf.placeholder(tf.float32, shape=(2, 5))
        out = tf.gather(val, [[0, 1], [1, 0]], axis=1)

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: np.arange(10).reshape([2, 5])})[0]

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
        print(self.with_ngraph(run_test))
