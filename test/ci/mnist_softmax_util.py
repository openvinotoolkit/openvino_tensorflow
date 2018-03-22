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

# NOTE: This script is based on the MNIST MLP implementation provided by the
# script 'tensorflow/examples/tutorials/mnist/mnist_softmax.py'.
"""
Provides functionality similar to the MNIST MLP script provided by Tensorflow 1.3's
'tensorflow/examples/tutorials/mnist/mnist_softmax.py' script.

The primary differences are:
    - This module provides the functionality as usable Python objects, rather than
      via a commend-line interface.
    - Added logic related to the Intel Nervana 'XLA_NGRAPH' device.
    - Additional features to support performance and correctness testing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf

# Set the random seed so that it produces same results every time. It has to be
# done pretty much at the the beginning - so we are doing it here.
tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

#===================================================================================================


def maybe_download_MNIST_data(data_download_script_path, data_dir):
    """
    Invoke a script (typically 'download-mnist-data.sh') to ensure that a local copy of MNIST's
    data files can be found in directory 'data_dir'.

    Useful for environments in which MNIST's own data-downloading logic doesn't work due to
    firewall / proxy issues.
    """

    cmd = [data_download_script_path, data_dir]
    p = subprocess.Popen(
        cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        raise Exception("ERROR: Subprocess command failed: {}".format(cmd))


#===================================================================================================


def compute_accuracy(
        mnist_data_set,
        tf_session,
        op_x,
        op_y,
        op_y_,
):
    """
  Compute the accuracy of the specified MNIST model, using the entire MNIST
  data set.  Return the accuracy as a single floating-point number.
  """
    correct_prediction = tf.equal(tf.argmax(op_y, 1), tf.argmax(op_y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_value = tf_session.run(
        accuracy,
        feed_dict={
            op_x: mnist_data_set.test.images,
            op_y_: mnist_data_set.test.labels
        })
    return accuracy_value


#===================================================================================================


def create_inference_model():
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b

    return {
        'x': x,
        'y': y,
    }


#===================================================================================================


def add_training_ops(inference_model_dict, ):
    y = inference_model_dict['y']

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

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    d = inference_model_dict.copy()
    d['y_'] = y_
    d['train_step'] = train_step
    d['batch_cost'] = cross_entropy

    return d


#===================================================================================================


def train_model(tf_config, training_model_dict, mnist_data_set,
                num_training_iterations, batch_size, graph_summary_dir_or_None,
                final_iteration_chrome_trace_filename_or_None,
                return_per_iteration_cost_value,
                return_per_iteration_accuracy_value, log_stream):
    """
  Train the model for 'num_training_iterations' iterations with batch size of
  'batch_size'.  (The final batch of each epoch may contain fewer examples.)

  Assume that the caller has already established any relevant contexts, such as those related
  to device placement.

  Parameters:
  - tf_config - TF config object to use.
  - training_model_dict - A dictionary like the one returned by 'create_inference_model'.
  - num_training_iterations - The total number of training iterations to run.
  - batch_size - Maximum batch size.
  - graph_summary_dir_or_None - If not None, a string that names a directory into which a graph
       summary will be dumped.
  - final_iteration_chrome_trace_filename_or_None - If not None, then the final training iteration
       will create a Chrome trace file with the specified pathname.
  - return_per_iteration_cost_value - If true, the returned dictionary will contain a 'batch_costs'
       entry.  It's value is a list containing the value of the cost function after each training
       iteration.
  - return_per_iteration_accuracy_value - If true, the returned dictionary will contain an
       'batch_accuracies' entry.  Its value is a list containing the value of the cost function after
       each training iteration.  The final entry in this list will be the same value as the
       returned dictionary's 'final_accuracy' entry.

  Return a dict with the following entries:
  - 'batch_costs' - See above.
  - 'batch_accuracies' - See above.
  - 'final_accuracy' - The accuracy of the model after the final training iteration, as evaluated
       against the entire MNIST data set.
  """
    batch_accuracies = []
    batch_costs = []
    x = training_model_dict['x']
    y = training_model_dict['y']
    y_ = training_model_dict['y_']
    train_step = training_model_dict['train_step']
    batch_cost = training_model_dict['batch_cost']

    run_metadata = tf.RunMetadata()
    sess = tf.Session(config=tf_config)
    tf.global_variables_initializer().run(session=sess)

    if graph_summary_dir_or_None is not None:
        writer = tf.summary.FileWriter(graph_summary_dir_or_None, sess.graph)

    # Train
    for i in range(num_training_iterations):
        if (i == 1):
            start = time.time()
        batch_xs, batch_ys = mnist_data_set.train.next_batch(batch_size)

        fetches = {
            'train_step': train_step,
            'batch_cost': batch_cost,
        }

        if return_per_iteration_cost_value:
            fetches['batch_cost'] = batch_cost

        feed_dict = {
            x: batch_xs,
            y_: batch_ys,
        }

        if (i == num_training_iterations - 1) and (
                final_iteration_chrome_trace_filename_or_None is not None):
            # Create a timeline for the last loop and export to json to view with
            # chrome://tracing/.
            run_result = sess.run(
                fetches,
                feed_dict,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata)

            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open('timeline.ctf.json', 'w') as trace_file:
                trace_file.write(trace.generate_chrome_trace_format())
        else:
            run_result = sess.run(
                fetches,
                feed_dict,
            )

        if return_per_iteration_cost_value:
            batch_costs.append(run_result['batch_value'])

        if return_per_iteration_accuracy_value:
            accuracy = compute_accuracy(mnist_data_set, sess, x, y, y_)
            batch_accuracies.append(accuracy)

        log_stream.write('Training step:: {} '.format(i))
        log_stream.write('cost:: {} \n'.format(run_result['batch_cost']))

    end = time.time()

    # Test trained model
    final_accuracy = compute_accuracy(mnist_data_set, sess, x, y, y_)

    return_dict = {
        'final_accuracy': final_accuracy,
        'training_time': end - start
    }

    if return_per_iteration_cost_value:
        return_dict['batch_costs'] = batch_costs

    if return_per_iteration_cost_value:
        return_dict['batch_accuracies'] = batch_accuracies

    sess.close()

    return return_dict


#===================================================================================================


def train_and_evaluate_model(
        mnist_data_dir,
        tf_device_str_or_None,
        enable_global_jit,
        allow_soft_placement,
        log_device_placement,
        num_training_iterations,
        batch_size,
        graph_summary_dir_or_None,
        final_iteration_chrome_trace_filename_or_None,
        return_per_iteration_cost_value,
        return_per_iteration_accuracy_value,
        log_stream,
):
    """
  Train and evaluate an MNIST MLP model.  Return a dictionary describing the training process
  and the final form of the trained model.  If specified, also create various log / output files.

  Typical values for tf_device_str_or_None:
    '/job:localhost/replica:0/task:0/cpu:0'
    '/job:localhost/replica:0/task:0/device:XLA_CPU:0'
    '/job:localhost/replica:0/task:0/device:NGRAPH:0'
    None - Don't create a TF device context at all.
  """
    for env_var in [
            'XLA_NGRAPH_CPU_PRIORITY',
            'XLA_NGRAPH_NGRAPH_PRIORITY',
    ]:
        if env_var in os.environ:
            log_stream.write(
                'WARNING: Environment var will influence device placement: {}={}\n'.
                format(env_var, os.environ[env_var]))

    # Import data
    mnist_data_set = input_data.read_data_sets(mnist_data_dir, one_hot=True)

    # Set up the TF config object...
    config = tf.ConfigProto(
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement,
        inter_op_parallelism_threads=1)

    if enable_global_jit:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    else:
        config.graph_options.optimizer_options.global_jit_level = 0

    if tf_device_str_or_None:
        with tf.device(tf_device_str_or_None):
            # Create the models...
            inference_model_dict = create_inference_model()
            training_model_dict = add_training_ops(inference_model_dict)

            # Do the training...
            training_results_dict = train_model(
                config, training_model_dict, mnist_data_set,
                num_training_iterations, batch_size, graph_summary_dir_or_None,
                final_iteration_chrome_trace_filename_or_None,
                return_per_iteration_cost_value,
                return_per_iteration_accuracy_value, log_stream)

    else:
        # Create the models...
        inference_model_dict = create_inference_model()
        training_model_dict = add_training_ops(inference_model_dict)

        # Do the training...
        training_results_dict = train_model(
            config, training_model_dict, mnist_data_set,
            num_training_iterations, batch_size, graph_summary_dir_or_None,
            final_iteration_chrome_trace_filename_or_None,
            return_per_iteration_cost_value,
            return_per_iteration_accuracy_value, log_stream)

    log_stream.write('\nTest parameters\n')
    log_stream.write('Device: {}\n'.format(tf_device_str_or_None))
    log_stream.write('XLA_NGRAPH_BACKEND: {}\n'.format(
        os.environ['XLA_NGRAPH_BACKEND']))
    log_stream.write('Training batch size: {}\n'.format(batch_size))
    log_stream.write(
        'Training iteration count: {}\n'.format(num_training_iterations))
    log_stream.write('Training time: {} seconds\n'.format(
        training_results_dict['training_time']))

    return training_results_dict


#===================================================================================================
