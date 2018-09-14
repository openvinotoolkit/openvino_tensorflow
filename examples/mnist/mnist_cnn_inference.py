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
"""A simplified deep MNIST classifier using convolutional layers.
This script has the following changes when compared to mnist_deep.py:
1. no dropout layer (which disables the rng op)
2. no truncated normal initialzation(which disables the while op)

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import getpass
import time
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import ngraph

FLAGS = None


def deepnn_inference(x):
    """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, a scalar placeholder). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). The scalar placeholder is meant for the probability of dropout. Since we don't
    use a dropout layer in this script, this placeholder is of no relavance and acts as a dummy.
  """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10], "W_fc2")
        b_fc2 = bias_variable([10])

        # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv, tf.placeholder(tf.float32)


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.get_variable(name, shape)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def train_mnist_cnn(FLAGS):
    # Config
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        inter_op_parallelism_threads=4)

    # Note: Additional configuration option to boost performance is to set the
    # following environment for the run:
    # OMP_NUM_THREADS=44 KMP_AFFINITY=granularity=fine,scatter
    # The OMP_NUM_THREADS number should correspond to the number of
    # cores in the system

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn_inference(x)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('test accuracy', accuracy)

    graph_location = "/tmp/" + getpass.getuser(
    ) + "/tensorboard-logs/mnist-convnet"
    print('Saving graph to: %s' % graph_location)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.model_dir)
        #sess.run(tf.global_variables_initializer())
        test_accuracy_final = 0
        num_eval_cycles = FLAGS.num_eval_cycles

        for i in range(num_eval_cycles):
            batch = mnist.test.next_batch(FLAGS.batch_size)
            t = time.time()
            summary, test_accuracy = sess.run([merged, accuracy],
                                              feed_dict={
                                                  x: batch[0],
                                                  y_: batch[1],
                                                  keep_prob: 0.5
                                              })
            test_accuracy_final = test_accuracy_final + test_accuracy
            reference_accucary = 0.9612
            print('step %d, test_accuracy %g, %g sec for infernce step' %
                  (i, test_accuracy, time.time() - t))
            train_writer.add_summary(summary, i)
        test_accuracy_final = test_accuracy_final / num_eval_cycles
        print("Inference  finished")
        print(test_accuracy_final)
        print('Reference accuracy %g' % (reference_accucary))
        assert (reference_accucary -
                test_accuracy_final) / reference_accucary < 0.05
        print(
            "The validation accuracy is within 5% of the trained reference data!"
        )
        return test_accuracy_final


def main(_):
    train_mnist_cnn(FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory where input data is stored')
    parser.add_argument(
        '--num_eval_cycles',
        type=int,
        default=78,
        help='Number of training iterations')

    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch Size')

    parser.add_argument(
        '--model_dir',
        type=str,
        default='./mnist_trained/',
        help='enter model dir')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
