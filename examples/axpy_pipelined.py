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


def build_simple_model(input_array):
    # Convert the numpy array to TF Tensor
    input = tf.cast(input_array, tf.float32)

    # Define the Ops
    mul = tf.compat.v1.math.multiply(input_array, 5)
    add = tf.compat.v1.math.add(mul, 10)
    output = add
    return output


def build_data_pipeline(input_array, map_function, batch_size):
    dataset = (tf.compat.v1.data.Dataset.from_tensor_slices(
        (tf.constant(input_array)
        )).map(map_function).batch(batch_size).prefetch(1))

    iterator = dataset.make_initializable_iterator()
    data_to_be_prefetched_and_used = iterator.get_next()

    return data_to_be_prefetched_and_used, iterator


def run_axpy_pipeline():
    input_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_output_array = [-1, -1, 1, -1, -1, -1, -1, -1, -1]
    output_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    multiplier = 10

    for i in range(1, 10):
        input_array[i - 1] = input_array[i - 1] * i * multiplier
    map_function = lambda x: x * multiplier
    batch_size = 1
    pipeline, iterator = build_data_pipeline(input_array, map_function,
                                             batch_size)
    model = build_simple_model(pipeline)
    with tf.compat.v1.Session() as sess:
        # Initialize the globals and the dataset
        sess.run(iterator.initializer)

        for i in range(1, 10):
            # Expected value is:
            expected_output_array[i - 1] = (
                (input_array[i - 1] * multiplier) * 5) + 10

            # Run one iteration
            output = sess.run(model)
            output_array[i - 1] = output[0]
    return input_array, output_array, expected_output_array


def main(_):
    input_array, output_array, expected_output_array = run_axpy_pipeline()
    for i in range(1, 10):
        print("Iteration:", i, " Input: ", input_array[i - 1], " Output: ",
              output_array[i - 1], " Expected: ", expected_output_array[i - 1])
        sys.stdout.flush()


if __name__ == '__main__':
    #os.environ['NGRAPH_TF_USE_PREFETCH'] = "1"
    tf.app.run(main=main)
