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
"""nGraph TensorFlow bridge ReluGrad operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.ops.gen_math_ops import rsqrt_grad

from common import NgraphTest
import numpy as np


class TestRsqrtGrad(NgraphTest):

    @pytest.mark.parametrize(("shape",), (
        ([2, 3],),
        ([100],),
        ([3, 2],),
        ([3, 2, 3],),
        ([4, 2, 1, 3],),
    ))
    def test_rsqrtgrad(self, shape):
        a = tf.compat.v1.placeholder(tf.float32, shape)
        b = tf.compat.v1.placeholder(tf.float32, shape)

        y = np.random.rand(*shape)
        dy = np.random.rand(*shape)

        out = rsqrt_grad(a, b)

        def run_test(sess):
            return sess.run(out, feed_dict={a: y, b: dy})

        assert np.isclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test)).all()
