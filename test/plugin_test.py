# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple MNIST classifier example with JIT XLA and timelines.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

FLAGS = None


def main(_):

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        inter_op_parallelism_threads=1)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    #with tf.device('/device:XLA_TEST_PLUGIN:0'):
    with tf.device('/device:NGRAPH:0'):
        y = tf.matmul(x, w) + b

    sess = tf.Session(config=config)

    tf.global_variables_initializer().run(session=sess)

    train_loops = FLAGS.train_loop_count
    start = time.time()
    for i in range(train_loops):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run([y], feed_dict={x: batch_xs, y_: batch_ys})
        print("step = %d" % (i))
    end = time.time()
    elapsed = (end - start)
    print("Time elapsed: ", elapsed, " seconds")

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    parser.add_argument(
        '--xla', type=bool, default=True, help='Turn xla via JIT on')
    parser.add_argument(
        '--train_loop_count',
        type=int,
        default=1000,
        help='Number of training iterations')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
