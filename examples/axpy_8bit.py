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
"""nGraph TensorFlow axpy int8

"""
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.client import timeline

import ngraph_bridge

print("TensorFlow version: ", tf.version.GIT_VERSION, tf.version.VERSION)

# Define the data
a = tf.constant(np.full((2, 2), 5, dtype=np.int8), name='alpha')
x = tf.compat.v1.placeholder(tf.int8, [None, 2], name='x')
y = tf.compat.v1.placeholder(tf.int8, shape=(2, 2), name='y')

c = a * x
axpy = c + y

# Configure the session
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
config_ngraph_enabled = ngraph_bridge.update_config(config, backend_name='CPU')

# Create session and run
with tf.compat.v1.Session(config=config_ngraph_enabled) as sess:
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
