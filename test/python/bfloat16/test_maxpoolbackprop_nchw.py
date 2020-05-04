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

# Test Ngraph Op MaxPoolBackprop with data format NCHW
# TF Op:MaxPoolGrad

np.random.seed(5)

#Inputs
N = 4
C = 3
H = 8
W = 8

valid_shape = [4, 3, 3, 3]
same_shape = [4, 3, 4, 4]

output_nchw = {
    "VALID": np.random.rand(*valid_shape).astype('f'),
    "SAME": np.random.rand(*same_shape).astype('f')
}
grad_nchw = {
    "VALID": np.random.rand(*valid_shape).astype('f'),
    "SAME": np.random.rand(*same_shape).astype('f')
}

stride_nhwc = [1, 2, 2, 1]
ksize_nhwc = [1, 3, 3, 1]

stride_nchw = [1, 1, 2, 2]
ksize_nchw = [1, 1, 3, 3]


# TF graph
def tf_model(padding):
    orig_in = tf.compat.v1.placeholder(tf.float32, shape=[N, C, H, W])
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

    # transpose to NHWC
    orig_in_t = tf.transpose(orig_in_c, (0, 2, 3, 1))
    orig_out_t = tf.transpose(orig_out_c, (0, 2, 3, 1))
    grad_t = tf.transpose(grad_c, (0, 2, 3, 1))

    out = max_pool_grad(
        orig_in_t,
        orig_out_t,
        grad_t,
        ksize_nhwc,
        stride_nhwc,
        padding=padding,
        data_format="NHWC")

    # cast the output dtype back to float32
    output = tf.cast(out, tf.float32)

    # transpose to NCHW
    output_nchw = tf.transpose(output, (0, 3, 1, 2))
    return output_nchw, orig_in, orig_out, grad


# Ngraph graph
def ng_model(padding):
    orig_in = tf.compat.v1.placeholder(tf.float32, shape=[N, C, H, W])
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
        ksize_nchw,
        stride_nchw,
        padding=padding,
        data_format="NCHW")
    return out, orig_in, orig_out, grad


config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

i_np = np.random.rand(N, C, H, W).astype('f')  # NHWC


@pytest.mark.parametrize("padding", ("VALID", "SAME"))
def test_maxpoolbackprop_nchw(padding):
    g_np = grad_nchw[padding]
    o_np = output_nchw[padding]

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
