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
"""nGraph TensorFlow bridge floor operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf

from common import NgraphTest


class TestFloorOperations(NgraphTest):
    @pytest.mark.parametrize(("test_input", "expected"),
                             ((1.4, 1.0), (0.5, 0.0), (-0.3, -1.0)))
    def test_floor_1d(self, test_input, expected):
        print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

        val = tf.placeholder(tf.float32, shape=(1,))

        with tf.device(self.test_device):
            out = tf.floor(val)

            with tf.Session(config=self.config) as sess:
                print("Python: Running with Session")
                result = sess.run((out,), feed_dict={val: (test_input,)})
                print("value: {}".format(result[0]))
                assert result[0] == expected

    def test_floor_2d(self):
        test_input = ((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
        expected = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

        print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

        val = tf.placeholder(tf.float32, shape=(2, 3))

        with tf.device(self.test_device):
            out = tf.floor(val)

            with tf.Session(config=self.config) as sess:
                print("Python: Running with Session")
                (result,) = sess.run((out,), feed_dict={val: test_input})
                assert (result == expected).all()
