# ==============================================================================
#  Copyright 2019-2020 Intel Corporation
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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
import ngraph_bridge

import sys


def build_simple_model(input_array, tensor_var, var_modifier, array_multiplier):
    # Convert the numpy array to TF Tensor
    input = tf.cast(input_array, tf.float32)

    # Define the Ops
    mul = tf.compat.v1.math.multiply(input_array, array_multiplier)
    add = tf.compat.v1.math.add(mul, tensor_var)
    train_step = tensor_var.assign(add + var_modifier)

    with tf.control_dependencies([train_step]):
        train_op = tf.no_op('train_op')
    return add, train_op


def build_data_pipeline(input_array, map_function, batch_size):
    dataset = (tf.compat.v1.data.Dataset.from_tensor_slices(
        (tf.constant(input_array)
        )).map(map_function).batch(batch_size).prefetch(1))

    iterator = dataset.make_initializable_iterator()
    data_to_be_prefetched_and_used = iterator.get_next()

    return data_to_be_prefetched_and_used, iterator


def run_axpy_pipeline():
    input_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    multiplier = 10
    map_function = lambda x: x * multiplier
    batch_size = 1
    pipeline, iterator = build_data_pipeline(input_array, map_function,
                                             batch_size)
    var_init = 10
    init = tf.constant([var_init])
    var = tf.compat.v1.get_variable('x', initializer=init)

    var_modifier = 1
    array_multiplier = 5
    model = build_simple_model(pipeline, var, var_modifier, array_multiplier)

    expected_output_array = []
    output_array = []
    var_val = var_init
    with tf.compat.v1.Session() as sess:
        # Initialize the globals and the dataset
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(iterator.initializer)

        for i in range(1, 10):
            # Expected value is:
            expected_output = (
                (input_array[i - 1] * multiplier) * array_multiplier) + var_val
            expected_output_array.append(expected_output)
            var_val = expected_output + var_modifier

            # Run one iteration
            output, train_op = sess.run(model)
            output_array.append(output[0])
    return input_array, output_array, expected_output_array


def main(_):
    input_array, output_array, expected_output_array = run_axpy_pipeline()
    for i in range(1, 10):
        print("Iteration:", i, " Input: ", input_array[i - 1], " Output: ",
              output_array[i - 1], " Expected: ", expected_output_array[i - 1])
        sys.stdout.flush()


if __name__ == '__main__':
    os.environ['NGRAPH_TF_USE_PREFETCH'] = "1"
    tf.app.run(main=main)
