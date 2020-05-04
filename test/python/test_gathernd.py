# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
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
"""nGraph TensorFlow bridge gather_nd operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow.compat.v1 as tf
import os
import numpy as np

from common import NgraphTest


class TestGatherNDOperations(NgraphTest):

    def test_gather_nd(self):
        val = tf.placeholder(tf.float32, shape=(5, 10))
        indices = np.zeros([1, 3, 3, 1], dtype=np.int32)
        out = tf.gather_nd(val, indices, batch_dims=0, name='output')

        def run_test(sess):
            return sess.run((out,),
                            feed_dict={val: np.arange(50).reshape([5, 10])})[0]

        self.with_ngraph(run_test)

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
