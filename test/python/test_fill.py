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
"""nGraph TensorFlow bridge fill operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pytest

from common import NgraphTest


@pytest.mark.skip(reason="new deviceless mode WIP")
class TestFillOperations(NgraphTest):
  @pytest.mark.parametrize(
      ("shape", "value"),
      ((
       ([1, 2, 3, 1], 10),
       ([2, 1, 3], 2.0),
       ([2, 1, 3, 1, 1], -1.0),
       ([1, 2], False),
       ([1], 3),
       ([], 1),
       ([0],1),
       )))
  def test_fill(self, shape, value):

    tf_value = tf.placeholder(tf.float32, shape=(None))

    with tf.device(self.test_device):
      out = tf.fill(shape, tf_value)
      expected = np.full(shape, value)

      with tf.Session(config=self.config) as sess:
        (result,) = sess.run((out,), feed_dict={tf_value: value})
        assert result.shape == expected.shape
        assert np.allclose(expected, result)

  @pytest.mark.parametrize(("shape", "value"),
                           (
                           ([2, 2], 3),
                           ))
  def test_dynamic_dims_fallback(self, shape, value):

    tf_dims = tf.placeholder(tf.int32, shape=(None))
    tf_value = tf.placeholder(tf.float32, shape=(None))

    with tf.device(self.test_device):
      out = tf.fill(tf_dims, tf_value)
      expected = np.full(shape, value)

      cfg = self.config
      # expecting fallback to CPU
      cfg.allow_soft_placement = True
      with tf.Session(config=cfg) as sess:
        (result,) = sess.run((out,), feed_dict={tf_value: value,
                                                tf_dims: shape})
        assert result.shape == expected.shape
        assert np.allclose(expected, result)

    tf.reset_default_graph()
