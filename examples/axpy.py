# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow axpy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import ctypes

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.client import timeline
import json

import openvino_tensorflow

print("TensorFlow version: ", tf.version.GIT_VERSION, tf.version.VERSION)

# Setup TensorBoard
graph_location = "/tmp/" + getpass.getuser() + "/tensorboard-logs/test"
print('Saving graph to: %s' % graph_location)
train_writer = tf.compat.v1.summary.FileWriter(graph_location)

# Define the data
a = tf.constant(np.full((2048, 2048), 0.05, dtype=np.float32), name='alpha')
x = tf.compat.v1.placeholder(tf.float32, [None, 2048], name='x')
y = tf.compat.v1.placeholder(tf.float32, shape=(2048, 2048), name='y')

c = a * x
axpy = c + y

# Configure the session
config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)
config_ngraph_enabled = openvino_tensorflow.update_config(config)

# Create session and run
with tf.compat.v1.Session(config=config_ngraph_enabled) as sess:
    print("Python: Running with Session")
    options = tf.compat.v1.RunOptions(
        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    event_times = []
    for i in range(10):
        (result_axpy, result_c) = sess.run((axpy, c),
                                           feed_dict={
                                               x: np.ones((2048, 2048)),
                                               y: np.ones((2048, 2048)),
                                           },
                                           options=options,
                                           run_metadata=run_metadata)
        print(i)
        event_times.append(timeline.Timeline(run_metadata.step_stats))

    print("Writing event trace")
    with open('tf_event_trace.json', 'w') as f:
        f.write("[\n")
        for event in event_times:
            chrome_trace = event.generate_chrome_trace_format(
                show_dataflow=False)
            parsed_trace = json.loads(chrome_trace)
            for tr in parsed_trace['traceEvents']:
                f.write(json.dumps(tr) + ',\n')

train_writer.add_graph(tf.compat.v1.get_default_graph())
