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
"""nGraph TensorFlow bridge random uniform operations test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common import NgraphTest


class TestRandomUniformOperations(NgraphTest):

    def test_rand_uniform(self):
        samples = 10000
        shape = [samples]
        data = tf.random.uniform(shape, seed=0)
        out = tf.math.reduce_sum(data, axis=0, keepdims=True)
        rel_out = tf.math.divide(out, samples)

        sess_fn = lambda sess: sess.run([rel_out], feed_dict={})
        assert np.allclose(
            self.with_ngraph(sess_fn),
            self.without_ngraph(sess_fn),
            rtol=1.e-2,
            atol=1.e-2)
