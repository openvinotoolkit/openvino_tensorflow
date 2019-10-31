# ==============================================================================
#  Copyright 2019 Intel Corporation
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
"""nGraph TensorFlow FusedBatchNorm test

"""
import numpy as np
import tensorflow as tf
import os
import ngraph_bridge
import pytest

np.random.seed(5)

# Inputs
channels = 32
scale = np.random.rand(channels).astype('f')
offset = np.random.rand(channels).astype('f')
input_shape_nhwc = [4, 1, 2, channels]


def tf_model():
    x = tf.placeholder(tf.float32, shape=input_shape_nhwc)

    # cast the input dtype to bfloat16 for TF
    x_c = tf.cast(x, dtype=tf.bfloat16)

    out_list = tf.nn.fused_batch_norm(x, scale, offset, data_format='NHWC')

    # cast the output dtype back to float32
    norm = [tf.cast(i, dtype=tf.float32) for i in out_list]
    return norm, x


def ng_model():
    x = tf.placeholder(tf.float32, shape=input_shape_nhwc)
    norm = tf.nn.fused_batch_norm(x, scale, offset, data_format='NHWC')
    return norm, x


config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

k_np = np.random.rand(*input_shape_nhwc).astype('f')  # NHWC


def test_fusedbatchnorm_nhwc():
    #Test 1: tf_model TF-native
    with tf.Session(config=config) as sess_tf:
        ngraph_bridge.disable()
        tf_out, in_0 = tf_model()
        feed_dict = {in_0: k_np}
        tf_outval = sess_tf.run(tf_out, feed_dict=feed_dict)

    #Test 2: model2 with ngraph, NNP backend
    with tf.Session(config=config) as sess_ng:
        ngraph_bridge.enable()
        ngraph_bridge.update_config(config)
        os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
        ng_out, in_0 = ng_model()
        feed_dict = {in_0: k_np}
        ng_outval = sess_ng.run(ng_out, feed_dict=feed_dict)

    result1_bool = np.allclose(tf_outval[0], ng_outval[0], rtol=0, atol=1e-02)
    result2_bool = np.allclose(tf_outval[1], ng_outval[1], rtol=0, atol=1e-02)
    result3_bool = np.allclose(tf_outval[2], ng_outval[2], rtol=0, atol=1e-02)

    assert (result1_bool and result2_bool and result3_bool)
