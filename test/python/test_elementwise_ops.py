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
"""nGraph TensorFlow bridge elementwise operations test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf


from common import NgraphTest


class TestElementwiseOperations(NgraphTest):

  @pytest.mark.parametrize(("v1", "v2", "expected"),
                           ((1.0, -1.0, [1.0]), (100, 200, ([200],)), 
                           ([ 0.0, 5.0, 10.0], [6.0], (np.array([[6.0,  6.0, 10.0]]),))
))                            
  def test_maximum(self, v1, v2, expected):
    print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

    val1 = tf.placeholder(tf.float32, shape=(None))
    val2 = tf.placeholder(tf.float32, shape=(None))

    with tf.device(self.test_device):
      out = tf.maximum(val1, val2)

      with tf.Session(config=self.config) as sess:
        result = sess.run((out,), feed_dict={val1: (v1,), val2: (v2,)})
        print ("V1 ", v1, " V2 ", v2)
        print ("Result :", result)
        print ("Expected :", expected)
        assert np.allclose(result, expected)


  @pytest.mark.parametrize(("v1", "v2", "expected"),
                           ((1.4, 1.0, [False]), (-1.0, -1.0, ([True],)), 
                           (-1.0, 1000, [True] ), (200, 200, ([True],)),
                           ([ -1.0, 1.0, -4], [0.1, 0.1, -4], (np.array([[True,  False, True]]),)),
                           ([ -1.0, 1.0, -4], [-1.0], (np.array([[True,  False, True]]),))
))                            
  def test_less_equal(self, v1, v2, expected):
    print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

    val1 = tf.placeholder(tf.float32, shape=(None))
    val2 = tf.placeholder(tf.float32, shape=(None))

    with tf.device(self.test_device):
      out = tf.less_equal(val1, val2)

      with tf.Session(config=self.config) as sess:
        result = sess.run((out,), feed_dict={val1: (v1,), val2: (v2,)})
        print ("V1 ", v1, " V2 ", v2)
        print ("Result :", result)
        print ("Expected :", expected)
        assert np.allclose(result, expected)


  @pytest.mark.parametrize(("v1", "v2", "expected"),
                           ((1.4, 1.0, [False]), (-1.0, -1.0, ([False],)), 
                           (-1.0, 1000, [True] ), (200, 200, ([False],)),
                           ([ -1.0, 1.0, -4], [0.1, 0.1, -4], (np.array([[True,  False, False]]),)),
                           ([ -1.0, 1.0, -4], [-1.0], (np.array([[False,  False, True]]),))
))                            
  def test_less(self, v1, v2, expected):
    print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

    val1 = tf.placeholder(tf.float32, shape=(None))
    val2 = tf.placeholder(tf.float32, shape=(None))

    with tf.device(self.test_device):
      out = tf.less(val1, val2)

      with tf.Session(config=self.config) as sess:
        result = sess.run((out,), feed_dict={val1: (v1,), val2: (v2,)})
        print ("V1 ", v1, " V2 ", v2)
        print ("Result :", result)
        print ("Expected :", expected)
        assert np.allclose(result, expected)


  @pytest.mark.parametrize(("v1", "v2", "expected"),
                           ((1.4, 1.0, [True]), (-1.0, -1.0, ([True],)), 
                           (-1.0, 1000, [False] ), (200, 200, ([True],)),
                           ([ -1.0, 1.0, -4], [0.1, 0.1, -4], (np.array([[False,  True, True]]),)),
                           ([ -1.0, 1.0, -4], [-1.0], (np.array([[True,  True, False]]),))
))                            
  def test_greater_equal(self, v1, v2, expected):
    print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

    val1 = tf.placeholder(tf.float32, shape=(None))
    val2 = tf.placeholder(tf.float32, shape=(None))

    with tf.device(self.test_device):
      out = tf.greater_equal(val1, val2)

      with tf.Session(config=self.config) as sess:
        result = sess.run((out,), feed_dict={val1: (v1,), val2: (v2,)})
        print ("V1 ", v1, " V2 ", v2)
        print ("Result :", result)
        print ("Expected :", expected)
        assert np.allclose(result, expected)


  @pytest.mark.parametrize(("v1", "v2", "expected"),
                           ((1.4, 1.0, [True]), (-1.0, -1.0, ([False],)), 
                           (-1.0, 1000, [False] ), (200, 200, ([False],)),
                           ([ -1.0, 1.0, -4], [0.1, 0.1, -4], (np.array([[False,  True, False]]),)),
                           ([ -1.0, 1.0, -4], [-1.0], (np.array([[False,  True, False]]),))
))                            
  def test_greater(self, v1, v2, expected):
    print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

    val1 = tf.placeholder(tf.float32, shape=(None))
    val2 = tf.placeholder(tf.float32, shape=(None))

    with tf.device(self.test_device):
      out = tf.greater(val1, val2)

      with tf.Session(config=self.config) as sess:
        result = sess.run((out,), feed_dict={val1: (v1,), val2: (v2,)})
        print ("V1 ", v1, " V2 ", v2)
        print ("Result :", result)
        print ("Expected :", expected)
        assert np.allclose(result, expected)


  @pytest.mark.parametrize(("v1", "v2", "expected"),
                           ((True, True, [True]), (True, False, ([False],)), 
                           (1.0, -2.0, ([True],)), (False, 100, ([False],)), 
                           ([ False, True, False], [True], (np.array([[False,  True, False]]),))
))                            
  def test_logical_and(self, v1, v2, expected):
    print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

    val1 = tf.placeholder(tf.bool, shape=(None))
    val2 = tf.placeholder(tf.bool, shape=(None))

    with tf.device(self.test_device):
      out = tf.logical_and(val1, val2)

      with tf.Session(config=self.config) as sess:
        result = sess.run((out,), feed_dict={val1: (v1,), val2: (v2,)})
        print ("V1 ", v1, " V2 ", v2)
        print ("Result :", result)
        print ("Expected :", expected)
        assert np.allclose(result, expected)

