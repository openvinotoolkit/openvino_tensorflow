# ==============================================================================
#  Copyright 2019-2020 Intel Corporation
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
"""nGraph TensorFlow bridge MaxPoolBackprop operation test

"""

# Currently, this test fails with a segmentation fault
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.ops.gen_nn_ops import max_pool_grad

import ngraph_bridge

# Test Ngraph Op MaxPoolBackprop with data format NHWC
# TF Op:MaxPoolGrad

np.random.seed(5)

#Inputs
N = 4
H = 8
W = 8
C = 3

valid_shape = [4, 3, 3, 3]
same_shape = [4, 4, 4, 3]

output_nhwc = {
    "VALID": np.random.rand(*valid_shape).astype('f'),
    "SAME": np.random.rand(*same_shape).astype('f')
}
grad_nhwc = {
    "VALID": np.random.rand(*valid_shape).astype('f'),
    "SAME": np.random.rand(*same_shape).astype('f')
}
stride_nhwc = [1, 2, 2, 1]
ksize_nhwc = [1, 3, 3, 1]


# TF graph
def tf_model(padding):
    orig_in = tf.compat.v1.placeholder(tf.float32, shape=[N, H, W, C])
    if padding == "VALID":
        grad = tf.compat.v1.placeholder(tf.float32, shape=valid_shape)
        orig_out = tf.compat.v1.placeholder(tf.float32, shape=valid_shape)
    elif padding == "SAME":
        grad = tf.compat.v1.placeholder(tf.float32, shape=same_shape)
        orig_out = tf.compat.v1.placeholder(tf.float32, shape=same_shape)

    # cast the input dtype to bfloat16 for TF
    orig_in_c = tf.cast(orig_in, tf.bfloat16)
    orig_out_c = tf.cast(orig_out, tf.bfloat16)
    grad_c = tf.cast(grad, tf.bfloat16)

    out = max_pool_grad(
        orig_in_c,
        orig_out_c,
        grad_c,
        ksize_nhwc,
        stride_nhwc,
        padding=padding,
        data_format="NHWC")

    # cast the output dtype back to float32
    output = tf.cast(out, tf.float32)
    return output, orig_in, orig_out, grad


# Ngraph graph
def ng_model(padding):
    orig_in = tf.compat.v1.placeholder(tf.float32, shape=[N, H, W, C])
    if padding == "VALID":
        grad = tf.compat.v1.placeholder(tf.float32, shape=valid_shape)
        orig_out = tf.compat.v1.placeholder(tf.float32, shape=valid_shape)
    elif padding == "SAME":
        grad = tf.compat.v1.placeholder(tf.float32, shape=same_shape)
        orig_out = tf.compat.v1.placeholder(tf.float32, shape=same_shape)

    out = max_pool_grad(
        orig_in,
        orig_out,
        grad,
        ksize_nhwc,
        stride_nhwc,
        padding=padding,
        data_format="NHWC")
    return out, orig_in, orig_out, grad


config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

i_np = np.random.rand(N, H, W, C).astype('f')  # NHWC


@pytest.mark.parametrize("padding", ("VALID", "SAME"))
def test_maxpoolbackprop_nhwc(padding):
    g_np = grad_nhwc[padding]
    o_np = output_nhwc[padding]

    #Test 1: tf_model TF-native
    with tf.compat.v1.Session(config=config) as sess_tf:
        ngraph_bridge.disable()
        tf_out, orig_in, orig_out, grad = tf_model(padding)
        feed_dict = {orig_in: i_np, orig_out: o_np, grad: g_np}
        tf_outval = sess_tf.run(tf_out, feed_dict=feed_dict)

    #Test 2: model2 with ngraph, NNP backend
    with tf.compat.v1.Session(config=config) as sess_ng:
        ngraph_bridge.enable()
        ngraph_bridge.update_config(config)
        os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
        ng_out, orig_in, orig_out, grad = ng_model(padding)
        feed_dict = {orig_in: i_np, orig_out: o_np, grad: g_np}
        ng_outval = sess_ng.run(ng_out, feed_dict=feed_dict)

    assert (np.allclose(tf_outval, ng_outval, rtol=0, atol=1e-02))
