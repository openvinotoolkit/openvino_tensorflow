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
"""nGraph TensorFlow bridge stack operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pytest

from common import NgraphTest

class TestStackOperations(NgraphTest):
  @pytest.mark.parametrize(
      ("shapes", "axis"),
      (
       ([(1, 2), (1, 2)], 2),
       ([(3, 2), (3, 2)], 0), 
       ([(2, 3), (2, 3)], 1),
       ([(1, 2, 4, 5), (1, 2, 4, 5), (1, 2, 4, 5)], 3),
       ([(1, 2, 3, 5), (1, 2, 3, 5), (1, 2, 3, 5)], 4),
       ([(3, 3, 7, 3), (3, 3, 7, 3)], -1),
       ([(0)], 0),
       ([(1)], 1)
      )
      )
  def test_stack(self, shapes, axis):
    values = [ np.random.random_sample(s) for s in shapes ]
    expected = np.stack(values, axis)
    placeholders = [ tf.placeholder(tf.float64, s) for s in shapes ]
    with self.device:
      a = tf.stack(placeholders, axis)
      with self.session as sess:
        (result,) = sess.run([a], feed_dict={p: v for p, v in zip(placeholders, values)})
        assert result.shape == expected.shape
        assert np.allclose(result, expected)
