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
"""nGraph TensorFlow bridge log operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf

from common import NgraphTest


@pytest.mark.skip(reason="new deviceless mode WIP")
class TestWhileLoop(NgraphTest):
  def test_while_loop(self):
    with self.device:
      # Simple example taken from TF docs for tf.while
      i = tf.constant(0)
      c = lambda i: tf.less(i, 10)
      b = lambda i: tf.add(i, 1)
      r = tf.while_loop(c, b, [i])

      # We'll need soft placement here
      cfg = self.config
      cfg.allow_soft_placement = True

      with tf.Session(config=cfg) as sess:
        result = sess.run((r,))
        assert result[0] == 10
