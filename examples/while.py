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

#
# Simple while-loop example.
#
# !!!NOTE!!!: For the time being it is necessary to run this with the
# environment variable NGRAPH_TF_SKIP_CLUSTERING=1. Otherwise the nGraph
# clustering step will choke, because the graph (of course) contains a cycle.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Define LD_LIBRARY_PATH indicating where nGraph library is located fornow.
# Eventually this won't be needed as the library will be available in either
# the Python site-packages or some other means
import ctypes
lib = ctypes.cdll.LoadLibrary('libngraph_device.so')

#with tf.device("/device:NGRAPH:0"):
if True:
  i = tf.constant(0)
  j = tf.constant(10)
  c = lambda i,j: tf.logical_and(tf.less(i, 10),tf.greater(j,4))
  b = lambda i,j: (tf.add(i, j),tf.subtract(j, i))
  r = tf.while_loop(c, b, (i,j))

  config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=True,
      inter_op_parallelism_threads=1)

  with tf.Session(config=config) as sess:
    r_out = sess.run(r)
    print("result:", r_out)
