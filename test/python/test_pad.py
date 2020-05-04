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
"""nGraph TensorFlow bridge pad operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest

np.random.seed(5)


class TestCastOperations(NgraphTest):

    def test_pad(self):
        input_data = tf.compat.v1.placeholder(tf.int32, shape=(2, 3))
        paddings = tf.compat.v1.placeholder(tf.int32, shape=(2, 2))

        out = tf.pad(input_data, paddings)

        inp = ((4, 2, 4), (4, 4, 1))
        pad = ((5, 3), (5, 5))

        def run_test(sess):
            return sess.run(out, feed_dict={input_data: inp, paddings: pad})

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
