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
#
# This file is derived from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py
# with changed by Intel using Horovod.
#
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************
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

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

import tensorflow as tf
import openvino_tensorflow
tf.compat.v1.disable_eager_execution()
import numpy as np
import horovod.tensorflow as hvd

FLAGS = None


def deepnn(x):
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
    return y_conv, tf.compat.v1.placeholder(tf.float32)


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    weight_var = tf.compat.v1.get_variable(name, shape)
    return weight_var


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def train_mnist_cnn(FLAGS):
    # Config
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=1)
    config_ovtf_enabled = openvino_tensorflow.update_config(config)

    # Note: Additional configuration option to boost performance is to set the
    # following environment for the run:
    # OMP_NUM_THREADS=44 KMP_AFFINITY=granularity=fine,scatter
    # The OMP_NUM_THREADS number should correspond to the number of
    # cores in the system

    # Create the model
    x = tf.compat.v1.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    # add distributed wrapper to "adam_optimizer"
    opt = hvd.DistributedOptimizer(tf.compat.v1.train.AdamOptimizer(1e-4))
    global_step = tf.contrib.framework.get_or_create_global_step()
    with tf.name_scope('distributed_optimizer'):
        train_step = opt.minimize(cross_entropy, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.compat.v1.summary.scalar('Training accuracy', accuracy)
    tf.compat.v1.summary.scalar('Loss function', cross_entropy)

    graph_location = "/tmp/" + getpass.getuser(
    ) + "/tensorboard-logs/mnist-convnet"
    print('Saving graph to: %s' % graph_location)

    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.compat.v1.get_default_graph())

    saver = tf.compat.v1.train.Saver()
    train_loops = FLAGS.train_loop_count
    num_test_images = FLAGS.test_image_count
    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),
        # Horovod: adjust number of steps based on number of ranks.
        #tf.train.StopAtStepHook(train_loops // hvd.size())
        tf.train.StopAtStepHook(train_loops)
    ]

    with tf.train.MonitoredTrainingSession(
            hooks=hooks, config=config_ovtf_enabled) as sess:

        step = 0
        start = time.time()

        loss_values = []
        test_accuracy = []
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, (60000, 784))
        x_train = x_train.astype(np.float32) / 255
        y_train = to_categorical(y_train, num_classes=10)
        while not sess.should_stop():
            index = np.random.choice(60000, FLAGS.batch_size)
            x_random = x_train[index]
            y_random = y_train[index]
            sess.run(train_step, feed_dict={x: x_random, y_: y_random})
            step += 1
            if step % 10 == 0:
                t = time.time()
                if hvd.rank() == 0:
                    print('step %d training accuracy %g %g sec to evaluate' %
                          (step,
                           sess.run(
                               accuracy, feed_dict={
                                   x: x_random,
                                   y_: y_random
                               }), time.time() - t))
            t = time.time()
            _, summary, loss = sess.run([train_step, merged, cross_entropy],
                                        feed_dict={
                                            x: x_random,
                                            y_: y_random,
                                            keep_prob: 0.5
                                        })
            loss_values.append(loss)
            if hvd.rank() == 0:
                print('step %d, loss %g, %g sec for training step' %
                      (step, loss, time.time() - t))
            train_writer.add_summary(summary, step)

            if step == (train_loops // hvd.size() - 1) and hvd.rank() == 0:
                x_test = np.reshape(x_test, (10000, 784))
                x_test = x_test.astype(np.float32) / 255
                y_test = to_categorical(y_test, num_classes=10)
                x_test = mnist.test.images[:num_test_images]
                y_test = mnist.test.labels[:num_test_images]
                print('test accuracy: ',
                      sess.run(accuracy, feed_dict={
                          x: x_test,
                          y_: y_test
                      }))
                test_accuracy.append(accuracy)

        print("Training finished. Running test")
        saver.save(sess, FLAGS.model_dir)
        return loss_values, test_accuracy


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
        '--train_loop_count',
        type=int,
        default=1000,
        help='Number of training iterations')

    parser.add_argument('--batch_size', type=int, default=50, help='Batch Size')

    parser.add_argument(
        '--test_image_count',
        type=int,
        default=None,
        help="Number of test images to evaluate on")

    parser.add_argument(
        '--model_dir',
        type=str,
        default='./mnist_trained/',
        help='enter model dir')

    FLAGS, unparsed = parser.parse_known_args()
    hvd.init()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
