# ==============================================================================
#  Copyright 2018 Intel Corporation
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
import tfgraphviz as tfg

print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

# Setup TensorBoard
graph_location = "/tmp/" + getpass.getuser() + "/tensorboard-logs/test"
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)

# Define LD_LIBRARY_PATH indicating where nGraph library is located for now.
# Eventually this won't be needed as the library will be available in either
# the Python site-packages or some other means
lib = ctypes.cdll.LoadLibrary('libngraph_device.so')

# Define the data
a = tf.constant(np.full((2, 3), 5.0, dtype=np.float32), name='alpha')
x = tf.placeholder(tf.float32, [None, 3], name='x')
y = tf.placeholder(tf.float32, shape=(2, 3), name='y')

# PLace this computation to NGRAPH
with tf.device("/device:NGRAPH:0"):
    c = a * x

axpy = c + y

# Configure the session
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

# Create session and run
with tf.Session(config=config) as sess:
    print("Python: Running with Session")
    for i in range(10):
        (result_axpy, result_c) = sess.run(
            (axpy, c),
            feed_dict={
                x: np.ones((2, 3)),
                y: np.ones((2, 3)),
            })
        print("[", i, "] ", i)
        print("Result: \n", result_axpy, " C: \n", result_c)

train_writer.add_graph(tf.get_default_graph())
tf.train.write_graph(
    tf.get_default_graph(), '.', 'axpy.pbtxt', as_text=True)
g = tfg.board(tf.get_default_graph())
g.render(filename="./axpy")
