# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
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
"""nGraph TensorFlow axpy

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import ctypes

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import json

import ngraph_bridge

print("TensorFlow version: ", tf.version.GIT_VERSION, tf.version.VERSION)

# Setup TensorBoard
graph_location = "/tmp/" + getpass.getuser() + "/tensorboard-logs/test"
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)

# Define the data
a = tf.constant(np.full((2, 2), 5, dtype=np.int8), name='alpha')
x = tf.placeholder(tf.int8, [None, 2], name='x')
y = tf.placeholder(tf.int8, shape=(2, 2), name='y')

c = a * x
axpy = c + y

# Configure the session
config = tf.ConfigProto(inter_op_parallelism_threads=1)
config_ngraph_enabled = ngraph_bridge.update_config(config, backend_name='CPU')

# Create session and run
with tf.Session(config=config_ngraph_enabled) as sess:
    print("Python: Running with Session")
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    event_times = []
    for i in range(1):
        (result_axpy, result_c) = sess.run((axpy, c),
                                           feed_dict={
                                               x: np.ones((2, 2)),
                                               y: np.ones((2, 2)),
                                           },
                                           options=options,
                                           run_metadata=run_metadata)
        print(result_axpy)
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

train_writer.add_graph(tf.get_default_graph())
