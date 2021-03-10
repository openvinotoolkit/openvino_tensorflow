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
Reference to the original source code:
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_softmax.py
Add distributed fetaure with horovod
1. hvd.init()
2. Add distributed wrapper from hvd.DistributedOptimizer
3. Broadcast the variables from root rank to the rest processors: hvd.BroadcastGlobalVariablesHook(0)
4. Print the output for root rank only
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
import openvino_tensorflow
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import horovod.tensorflow as hvd

FLAGS = None

hvd.init()


def main(_):
    run_mnist(_)


def run_mnist(_):
    # Create the model
    with tf.name_scope("mnist_placholder"):
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
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #global_step = tf.train.get_or_create_global_step()
    global_step = tf.contrib.framework.get_or_create_global_step()
    opt = tf.train.GradientDescentOptimizer(0.5)
    # Add MPI Distributed Optimizer
    with tf.name_scope("horovod_opt"):
        opt = hvd.DistributedOptimizer(opt)
    train_step = opt.minimize(cross_entropy, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [
        hvd.BroadcastGlobalVariablesHook(0),
        tf.train.StopAtStepHook(last_step=10)
    ]

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Enable soft placement and tracing as needed
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        inter_op_parallelism_threads=1)
    config_ovtf_enabled = openvino_tensorflow.update_config(config)

    #config.graph_options.optimizer_options.global_jit_level = jit_level
    run_metadata = tf.compat.v1.RunMetadata()

    #init_op = tf.global_variables_initializer()
    print("Variables initialized ...")

    # The MonitoredTrainingSession takes care of session initialization
    with tf.train.MonitoredTrainingSession(
            hooks=hooks, config=config_ovtf_enabled) as mon_sess:
        start = time.time()
        train_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir,
                                                       mon_sess.graph)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, (60000, 784))
        x_train = x_train.astype(np.float32) / 255
        y_train = to_categorical(y_train, num_classes=10)
        while not mon_sess.should_stop():
            # Train
            index = np.random.choice(60000, 100)
            batch_xs = x_train[index]
            batch_ys = y_train[index]
            mon_sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            # Test trained model
            x_test = np.reshape(x_test, (10000, 784))
            x_test = x_test.astype(np.float32) / 255
            y_test = to_categorical(y_test, num_classes=10)
            if not mon_sess.should_stop():
                print("Accuracy: ",
                      mon_sess.run(accuracy, feed_dict={
                          x: x_test,
                          y_: y_test
                      }))

        end = time.time()

    if hvd.rank() == 0:
        print("Training time: %f seconds" % (end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
# run command for this distributed script
# mpirun -np 2 python mnist_softmax_distributed.py --data_dir=/mnt/data/mnist
