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
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, w) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

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

  config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True)
  jit_level = 0
  if FLAGS.xla:
    # Turns on XLA JIT compilation.
    jit_level = tf.OptimizerOptions.ON_1

  config.graph_options.optimizer_options.global_jit_level = jit_level
  run_metadata = tf.RunMetadata()
  sess = tf.Session(config=config)

  saver = tf.train.Saver()

  if FLAGS.load:
    saver.restore(sess, "model.ckpt")
  else:
    # Train
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)
    #train_step = tf.train.AdagradOptimizer(0.5).minimize(cross_entropy)
    #train_step = tf.train.RMSPropOptimizer(0.5).minimize(cross_entropy)

    tf.global_variables_initializer().run(session=sess)

    train_loops = FLAGS.train_loop_count
    start = time.time()
    for i in range(train_loops):
      batch_xs, batch_ys = mnist.train.next_batch(100)

      # Create a timeline for the last loop and export to json to view with
      # chrome://tracing/.
      if i == train_loops - 1:
        (_, loss) = sess.run(
            [train_step, cross_entropy],
            feed_dict={x: batch_xs,
                       y_: batch_ys},
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open('timeline.ctf.json', 'w') as trace_file:
          trace_file.write(trace.generate_chrome_trace_format())
        print("step = %d, loss = %f" % (i, loss))
      else:
        (_, loss) = sess.run(
            [train_step, cross_entropy], feed_dict={
                x: batch_xs,
                y_: batch_ys
            })
        print("step = %d, loss = %f" % (i, loss))
    end = time.time()
    elapsed = (end - start)
    print("Time elapsed: ", elapsed, " seconds" )

    saver.save(sess, "model.ckpt")

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  print("Accuracy: ", sess.run(
      accuracy, feed_dict={
          x: mnist.test.images,
          y_: mnist.test.labels
      }))
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
      '--load', help='Load model from model.ckpt', action='store_true')
  parser.add_argument(
      '--train_loop_count',
      type=int,
      default=1000,
      help='Number of training iterations')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
