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
"""A single block ResNet network - used for testing.
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
import random
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import ctypes
import ngraph

FLAGS = None
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2

tf.set_random_seed(0)
np.random.seed(0)
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  #import ipdb; ipdb.set_trace()
  return tf.layers.batch_normalization(
      inputs=inputs,
      momentum=_BATCH_NORM_DECAY,axis=3, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def building_block_v1(inputs, filters, training, strides,
                       data_format):
  """
  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs

  #import ipdb;ipdb.set_trace()

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)

  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)


  inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=2,
        strides=2, padding='VALID',
        data_format=data_format)
  inputs = tf.identity(inputs, 'final_avg_pool')

  inputs = tf.reshape(inputs, [-1,112*112*3])
  inputs = tf.layers.dense(inputs=inputs, units=10)
  inputs = tf.identity(inputs, 'final_dense')

  return inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      #kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def resnet_inference(FLAGS):

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=1)

    exec_device = '/device:'+FLAGS.select_device+':0'

    with tf.device(exec_device):
        # Create the model
        x = tf.placeholder(tf.float32, [None,224,224,3])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net

        y_conv= building_block_v1(x, filters=3, training=False, strides=1, data_format='channels_last')

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(
            tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

        accuracy = tf.reduce_mean(correct_prediction)

    graph_location = "/tmp/" + getpass.getuser() + "/tensorboard-logs/resnet-infer-oneblock"
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        num_eval_cycles = FLAGS.num_eval_cycles

        for i in range(num_eval_cycles):
            y_req_list=np.array(random.sample(range(1,10),FLAGS.batch_size))#np.zeros([FLAGS.batch_size,10])
            y_req=np.zeros([FLAGS.batch_size,10])
            y_req[np.arange(FLAGS.batch_size),y_req_list]=1
            batch= [np.random.rand(FLAGS.batch_size,224,224,3),y_req]
            start = time.time()
            test_accuracy = sess.run(
                [accuracy],
                feed_dict={
                    x: batch[0],
                    y_: batch[1],
                })
            end = time.time()

            print ("step %d Time: %f accuracy: %f" %(i, end - start, test_accuracy[0]))

    train_writer.add_graph(tf.get_default_graph())            

def main(_):
    resnet_inference(FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--select_device',
        type=str,
        default='NGRAPH',
        help='select device to execute on')
    parser.add_argument(
        '--num_eval_cycles',
        type=int,
        default=10,
        help='Number of training iterations')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch Size')

    parser.add_argument(
        '--inference_only',
        type=int,
        default=1,
        help='inference_only')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
