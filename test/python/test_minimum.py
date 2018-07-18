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
"""nGraph TensorFlow bridge minimum operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf

from common import NgraphTest


class TestMinimumOperations(NgraphTest):
  @pytest.mark.parametrize(("x", "y", "expected"),
                           ((1.5, -3.5, -3.5), (-4.5, -5.5, -5.5)))
  def test_minimum(self, x, y, expected):

    tf_x = tf.placeholder(tf.float32, shape=None)
    tf_y = tf.placeholder(tf.float32, shape=None)

    with self.device:
      out = tf.minimum(tf_x, tf_y)

      with self.session as sess:
        (result,) = sess.run((out,), feed_dict={tf_x: x, tf_y: y})
        assert result == expected
