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
"""nGraph TensorFlow bridge cast operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf

from common import NgraphTest

class TestExpandDims(NgraphTest):
    @pytest.mark.parametrize("axis", ([0, 2, 3]))
    def test_expand_dims(self, axis):
        s = [2, 3, 5]
        val = tf.ones(s, tf.float32)

        with tf.device(self.test_device):
            out = tf.expand_dims(val, axis) 
            with tf.Session(config=self.config) as sess:
                result = sess.run(out)

        s.insert(axis, 1)
        expected = tuple(s)
        assert result.shape == expected
