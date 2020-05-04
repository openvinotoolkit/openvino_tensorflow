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
"""nGraph TensorFlow bridge abs operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os

from common import NgraphTest


class TestLog1pOperations(NgraphTest):

    def test_log1p(self):
        test_input = (-3.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1, 10)
        val = tf.compat.v1.placeholder(tf.float32, shape=(8,))
        out = tf.math.log1p(val)

        def run_test(sess):
            return sess.run(out, feed_dict={val: test_input})

        ng_result = self.with_ngraph(run_test)
        tf_result = self.without_ngraph(run_test)

        assert (len(ng_result) == len(tf_result))

        for i, j in zip(ng_result, tf_result):
            assert (i == j) or (np.isnan(i) and np.isnan(j))
