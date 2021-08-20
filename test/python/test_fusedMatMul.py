# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow fusedMatMul tests.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from common import NgraphTest
from tensorflow.python.framework import dtypes

import numpy as np


class TestFusedMatMul(NgraphTest):

    @pytest.mark.parametrize(
        ("filename",),
        (
            ('fusedmatmul_0.pbtxt',),  # Relu
            ('fusedmatmul_1.pbtxt',),  # Relu6
            ('fusedmatmul_2.pbtxt',),  # No activation
        ))
    @pytest.mark.parametrize(("dim1", "dim2", "dim3"), ((3, 2, 2), (3, 4, 5)))
    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Only for Linux')
    def test_fusedmatmul_bias_pbtxt(self, filename, dim1, dim2, dim3):
        graph = self.import_pbtxt(filename)
        with graph.as_default() as g:
            x = self.get_tensor(g, "Placeholder_3:0", True)
            y = self.get_tensor(g, "Placeholder_4:0", True)
            z = self.get_tensor(g, "Placeholder_5:0", True)
            a = self.get_tensor(g, "Relu_1:0", True)

            inp1_values = 10 * np.random.rand(dim1, dim2) - 5
            inp2_values = 10 * np.random.rand(dim2, dim3) - 5
            bias_values = 10 * np.random.rand(dim3) - 5

            def run_test(sess):
                return sess.run(a, {
                    x: inp1_values,
                    y: inp2_values,
                    z: bias_values,
                })

            assert np.allclose(
                self.without_ngraph(run_test), self.with_ngraph(run_test), 1e-5,
                1e-6)
