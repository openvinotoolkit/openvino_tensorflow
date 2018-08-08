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
from common import NgraphTest


@pytest.mark.skip(reason="new deviceless mode WIP")
class TestTileOp(NgraphTest):
  def test_tile_nonzero(self):
    print("begining")
    x = tf.placeholder(tf.float32, shape=(2, 3, 4)) 
    y = tf.constant([1, 2, 1])
    z = tf.constant([1, 0, 3])
    
    # input value and expected value
    x_np = np.random.rand(2, 3, 4)
    y_np = np.array([1, 2, 1])
    z_np = np.array([1, 0, 3])

    with tf.device("/device:NGRAPH:0"):
      a = tf.tile(x,y)
      b = tf.tile(x,z)
      with self.session as sess:
        (result_a, result_b) = sess.run((a, b),
             feed_dict={
                 x: x_np,
                 y: y_np,
                 z: z_np
             })

    with tf.device("/device:CPU:0"):
      a = tf.tile(x,y)
      b = tf.tile(x,z)
      with self.session as sess:
        (expected_a, expected_b) = sess.run((a, b),
             feed_dict={
                 x: x_np,
                 y: y_np,
                 z: z_np
             })

    print("result_a:", result_a)
    print("expected_a:", expected_a)
    print("result_a:", result_b)
    print("expected_a:", expected_b)
    atol = 1e-5 
    error_a = np.absolute(result_a-expected_a)
    assert np.amax(error_a) <= atol



