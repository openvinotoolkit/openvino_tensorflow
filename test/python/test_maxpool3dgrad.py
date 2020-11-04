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
"""nGraph TensorFlow bridge Maxpool3DGrad tests.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

from common import NgraphTest


class TestMaxpool3DGrad(NgraphTest):
    INPUT_SIZES = [1, 3, 5, 4, 1]
    OUTPUT_SIZES = [1, 3, 5, 4, 1]
    GRAD_SIZES = [1, 3, 5, 4, 1]

    def test_maxpool3d_grad(self):
        inp_values = np.random.rand(*self.INPUT_SIZES)
        out_values = np.random.rand(*self.OUTPUT_SIZES)
        grad_values = np.random.rand(*self.GRAD_SIZES)

        def run_test(sess):
            inp = array_ops.placeholder(dtypes.float32)
            out = array_ops.placeholder(dtypes.float32)
            grad = array_ops.placeholder(dtypes.float32)
            return sess.run(
                nn_ops.MaxPool3DGrad(
                    orig_input=inp,
                    orig_output=out,
                    grad=grad,
                    ksize=[1, 1, 1, 1, 1],
                    strides=[1, 1, 1, 1, 1],
                    padding="SAME"), {
                        inp: inp_values,
                        out: out_values,
                        grad: grad_values,
                    })

        assert np.allclose(
            self.without_ngraph(run_test), self.with_ngraph(run_test))
