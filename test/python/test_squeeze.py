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
"""nGraph TensorFlow bridge squeeze operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import getpass

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python import pywrap_tensorflow as py_tf
from tensorflow.python.framework import errors_impl

import pytest

import unittest

from ctypes import cdll
cdll.LoadLibrary(
    'libngraph_device.so'
)

class TestSqueezeOperations(unittest.TestCase):

    test_device = "/device:NGRAPH:0"
    soft_placement = False
    log_placement = True

    def test_squeeze(self):
        print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

        # Get the list of devices
        tf_devices = device_lib.list_local_devices()

        shape1 = (1, 2, 3, 1)
        shape2 = (2, 1, 3)
        shape3 = (2, 1, 3, 1, 1)
        shape4 = (1 ,1)
        shape5 = (1)

        a = tf.placeholder(tf.float32, shape=shape1)
        b = tf.placeholder(tf.float32, shape=shape2)
        c = tf.placeholder(tf.float32, shape=shape3)
        d = tf.placeholder(tf.float32, shape=shape4)
        e = tf.placeholder(tf.float32, shape=shape5)

        with tf.device(self.test_device):

            a1 = tf.squeeze(a)
            b1 = tf.squeeze(b)
            c1 = tf.squeeze(c, [1, 4])
            d1 = tf.squeeze(d)
            e1 = tf.squeeze(e)

            # input value and expected value

            a_val = np.random.random_sample(shape1)
            b_val = np.random.random_sample(shape2)
            c_val = np.random.random_sample(shape3)
            d_val = np.random.random_sample(shape4)
            e_val = np.random.random_sample(shape5)

            a_sq = np.squeeze(a_val)
            b_sq = np.squeeze(b_val)
            c_sq = np.squeeze(c_val, axis=(1,4))
            d_sq = np.squeeze(d_val)
            e_sq = np.squeeze(e_val)

            config = tf.ConfigProto(
            allow_soft_placement=self.soft_placement,
            log_device_placement=self.log_placement,
            inter_op_parallelism_threads=1)

            with tf.Session(config=config) as sess:
                print("Python: Running with Session")
                (result_a, result_b, result_c, result_d, result_e) = sess.run(
                    (a1, b1, c1, d1, e1),
                    feed_dict={
                 a: a_val,
                 b: b_val,
                 c: c_val,
                 d: d_val,
                 e: e_val
                    })
                print("shape a:", result_a.shape)
                print("shape b:", result_b.shape)
                print("shape c:", result_c.shape)
                print("shape d:", result_d.shape)
                print("shape e:", result_e.shape)

                self.assertEqual(result_a.shape, a_sq.shape)
                self.assertEqual(result_b.shape, b_sq.shape)
                self.assertEqual(result_c.shape, c_sq.shape)
                self.assertEqual(result_d.shape, d_sq.shape)
                self.assertEqual(result_e.shape, e_sq.shape)

                print("Result A:", result_a)
                print("Expected A:", a_sq)
                self.assertTrue(np.allclose(result_a, a_sq))
                self.assertTrue(np.allclose(result_b, b_sq))
                self.assertTrue(np.allclose(result_c, c_sq))
                self.assertTrue(np.allclose(result_d, d_sq))
                self.assertTrue(np.allclose(result_e, e_sq))

    def test_incorrect_squeeze(self):
        shape1 = (1, 2, 3, 1)
        a = tf.placeholder(tf.float32, shape=shape1)
        with tf.device(self.test_device):
            with pytest.raises(ValueError):
                a1 = tf.squeeze(a, [0, 1])


if __name__ == '__main__':
    unittest.main()
