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
"""nGraph TensorFlow bridge squeeze operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pytest

from common import NgraphTest


class TestSqueezeOperations(NgraphTest):

    @pytest.mark.parametrize(("shape", "axis"),
                             (((1, 2, 3, 1), None), ((2, 1, 3), None),
                              ((2, 1, 3, 1, 1),
                               (1, 4)), ((1, 1), None), ((1,), None)))
    def test_squeeze(self, shape, axis):
        a = tf.compat.v1.placeholder(tf.float32, shape=shape)

        a1 = tf.squeeze(a, axis)
        a_val = np.random.random_sample(shape)
        a_sq = np.squeeze(a_val, axis=axis)

        sess_fn = lambda sess: sess.run((a1,), feed_dict={a: a_val})
        (result_a,) = self.with_ngraph(sess_fn)
        assert result_a.shape == a_sq.shape
        assert np.allclose(result_a, a_sq)

    def test_incorrect_squeeze(self):
        shape1 = (1, 2, 3, 1)
        a = tf.compat.v1.placeholder(tf.float32, shape=shape1)
        with pytest.raises(ValueError):
            a1 = tf.squeeze(a, [0, 1])
            sess_fn = lambda sess: sess.run((a1,))
            (result_a,) = self.with_ngraph(sess_fn)
