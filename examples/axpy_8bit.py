# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow axpy int8

"""
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.client import timeline

import openvino_tensorflow

print("TensorFlow version: ", tf.version.GIT_VERSION, tf.version.VERSION)

# Define the data
a = tf.constant(np.full((2, 2), 5, dtype=np.int8), name='alpha')
x = tf.compat.v1.placeholder(tf.int8, [None, 2], name='x')
y = tf.compat.v1.placeholder(tf.int8, shape=(2, 2), name='y')

c = a * x
axpy = c + y

# Configure the session
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
config_ovtf_enabled = openvino_tensorflow.update_config(config, backend_name='CPU')

# Create session and run
with tf.compat.v1.Session(config=config_ovtf_enabled) as sess:
    print("Python: Running with Session")
    options = tf.compat.v1.RunOptions(
        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    for i in range(1):
        (result_axpy, result_c) = sess.run((axpy, c),
                                           feed_dict={
                                               x: np.ones((2, 2)),
                                               y: np.ones((2, 2)),
                                           },
                                           options=options,
                                           run_metadata=run_metadata)
        print(result_axpy)
