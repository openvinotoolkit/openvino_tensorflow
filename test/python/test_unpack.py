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
"""nGraph TensorFlow bridge unpack operation test

"""

import tensorflow as tf
import numpy as np
import pytest

from common import NgraphTest


class TestUnpackOperations(NgraphTest):

    @pytest.mark.parametrize(("shape", "axis"), (((1, 2), -1), ((2, 3), 0),
                                                 ((4, 5, 6), 2)))
    def test_unpack_all(self, shape, axis):
        a = tf.placeholder(tf.float64, shape)
        input_val = np.random.random_sample(shape)
        with self.session as sess:
            with self.cpu_device:
                b = tf.placeholder(tf.float64, shape)
                unstacked = tf.unstack(b, axis=axis)
                expected = sess.run([unstacked], feed_dict={b: input_val})
            with self.device:
                tensors = tf.unstack(a, axis=axis)
                results = sess.run([tensors], feed_dict={a: input_val})
                assert len(results[0]) == len(expected[0])
                for i in range(len(expected[0])):
                    assert expected[0][i].shape == results[0][i].shape
                    assert np.allclose(expected[0][i], results[0][i])

    @pytest.mark.parametrize(("shape", "axis", "num"), (((1, 2), -1, 2),
                                                        ((4, 5, 6), 2, 6)))
    def test_unpack_num(self, shape, axis, num):
        print("Axis : ", axis)
        print(" Num : ", num)
        a = tf.placeholder(tf.float64, shape)
        input_val = np.random.random_sample(shape)
        with self.session as sess:
            with self.cpu_device:
                b = tf.placeholder(tf.float64, shape)
                unstacked = tf.unstack(b, num=num, axis=axis)
                expected = sess.run([unstacked], feed_dict={b: input_val})
            with self.device:
                tensors = tf.unstack(a, num=num, axis=axis)
                results = sess.run([tensors], feed_dict={a: input_val})
                assert len(results[0]) == len(expected[0])
                for i in range(len(expected[0])):
                    assert expected[0][i].shape == results[0][i].shape
                    assert np.allclose(expected[0][i], results[0][i])
