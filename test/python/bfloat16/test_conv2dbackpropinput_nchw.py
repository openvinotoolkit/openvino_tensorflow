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
"""nGraph TensorFlow bridge Conv2d operation test

"""
import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import os
from tensorflow.python.ops import nn_ops
import ngraph_bridge

# Tests Ngraph Op: ConvolutionBackpropData with data format NCHW
#TF Op: conv2d_backprop_input

np.random.seed(5)
#Inputs
N = 1
C = 2
H = 7
W = 6

I = C
O = 2
filt_width = 3
filt_height = 3

input_sizes_nchw = [N, C, H, W]
input_sizes_nhwc = [N, H, W, C]
filter_size_hwio = [filt_height, filt_width, I, O]
out_backprop_valid = [1, 2, 3, 2]
out_backprop_same = [1, 2, 4, 3]
out_backprop_in_sizes = {"VALID": out_backprop_valid, "SAME": out_backprop_same}
stride_nhwc = [1, 2, 2, 1]
stride_nchw = [1, 1, 2, 2]


#TF graph
def tf_model(padding):
    t1 = tf.constant(input_sizes_nhwc, dtype=tf.int32, name='t1')
    t2 = tf.compat.v1.placeholder(
        dtype=tf.float32, shape=filter_size_hwio, name='t2')
    t3 = tf.compat.v1.placeholder(
        dtype=tf.float32, shape=out_backprop_in_sizes[padding], name='t3')
    #reshaping the out_backprop to NHWC since TF does not support NCHW
    t3 = tf.transpose(t3, [0, 2, 3, 1])

    #Cast dtype to bfloat16 for TF because NNP casts ng_model inputs
    t2 = tf.cast(t2, dtype=tf.bfloat16)
    t3 = tf.cast(t3, dtype=tf.bfloat16)

    inp = nn_ops.conv2d_backprop_input(
        t1, t2, t3, stride_nhwc, padding=padding, data_format='NHWC')

    #Reshaping back to NCHW to compare outputs
    inp = tf.transpose(inp, [0, 3, 1, 2])
    #Cast dtype back to float32 similar to NNP
    inp = tf.cast(inp, dtype=tf.float32)
    return inp, t2, t3


#Ngraph Graph
def ng_model(padding):
    t1 = tf.constant(input_sizes_nchw, dtype=tf.int32, name='t1')
    t2 = tf.comapt.v1.placeholder(
        dtype=tf.float32, shape=filter_size_hwio, name='t2')
    t3 = tf.compat.v1.placeholder(
        dtype=tf.float32, shape=out_backprop_in_sizes[padding], name='t3')

    inp = nn_ops.conv2d_backprop_input(
        t1, t2, t3, stride_nchw, padding=padding, data_format='NCHW')
    return inp, t2, t3


config = tf.comapt.v1.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)


@pytest.mark.parametrize("padding", ("VALID", "SAME"))
def test_conv2dbackpropinput_nchw(padding):
    np_filter = np.random.rand(*filter_size_hwio).astype('f')
    n_np_out = np.random.rand(*out_backprop_in_sizes[padding]).astype('f')
    #Reshape to NHWC for TF
    t_np_out = np.transpose(n_np_out, (0, 2, 3, 1))

    with tf.comapt.v1.Session(config=config) as sess_tf:
        ngraph_bridge.disable()
        tf_out, filter_size, out_backprop = tf_model(padding)
        feed_dict = {filter_size: np_filter, out_backprop: t_np_out}
        tf_outval = sess_tf.run(tf_out, feed_dict=feed_dict)

    #Test 2: model2 with ngraph, NNP backend
    with tf.comapt.v1.Session(config=config) as sess_ng:
        ngraph_bridge.enable()
        ngraph_bridge.update_config(config)
        os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
        ng_out, filter_size, out_backprop = ng_model(padding)
        feed_dict = {filter_size: np_filter, out_backprop: n_np_out}
        ng_outval = sess_ng.run(ng_out, feed_dict=feed_dict)

    assert np.allclose(tf_outval, ng_outval, rtol=0, atol=1e-02)
