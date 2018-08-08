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
"""nGraph TensorFlow bridge abs operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import os

from common import NgraphTest


class TestAbsOperations(NgraphTest):
  @pytest.mark.parametrize("test_input", (1.4, -0.5, -1))
  def test_abs_1d(self, test_input):
    val = tf.placeholder(tf.float32, shape=(1,))
    out = tf.abs(val)

    def run_test(sess):
      return sess.run((out,), feed_dict={val: (test_input,)})

    assert self.with_ngraph(run_test) == self.without_ngraph(run_test)

  def test_abs_2d(self):
    test_input = ((1.5, -2.5, 0.0, -3.5), (-4.5, -5.5, 6.5, 1.0))
    val = tf.placeholder(tf.float32, shape=(2, 4))
    out = tf.abs(val)

    def run_test(sess):
      return sess.run(out, feed_dict={val: test_input})

    assert (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
