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
"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

import tensorflow as tf
import ngraph_bridge
tf.compat.v1.disable_eager_execution()
import numpy as np

FLAGS = None


def main(_):
    run_mnist(_)


def run_mnist(_):

    # Create the model
    x = tf.compat.v1.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    '''
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    '''
    # Enable soft placement and tracing as needed
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        inter_op_parallelism_threads=1)
    config_ngraph_enabled = ngraph_bridge.update_config(config)

    sess = tf.compat.v1.Session(config=config_ngraph_enabled)
    tf.compat.v1.global_variables_initializer().run(session=sess)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (60000, 784))
    x_train = x_train.astype(np.float32) / 255
    y_train = to_categorical(y_train, num_classes=10)
    # Train
    train_loops = FLAGS.train_loop_count
    for i in range(train_loops):
        index = np.random.choice(60000, 100)
        batch_xs = x_train[index]
        batch_ys = y_train[index]
        if (i == 1):
            start = time.time()
        sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})
        print("Step: ", i)

    end = time.time()

    # Save the TF graph as pdf
    tf.train.write_graph(
        tf.compat.v1.get_default_graph(),
        '.',
        'mnist_fprop_py.pbtxt',
        as_text=True)

    print("Inference time: %f seconds" % (end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    parser.add_argument(
        '--train_loop_count',
        type=int,
        default=10,
        help='Number of training iterations')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
