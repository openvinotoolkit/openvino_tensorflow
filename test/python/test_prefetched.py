# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
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
"""nGraph TensorFlow bridge prefetch test

"""
import sys
import pytest
import getpass
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import dtypes
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import ngraph_bridge

import numpy as np
from common import NgraphTest

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class TestPrefetched(NgraphTest):

    def build_data_pipeline(self, input_array, map_function, batch_size):
        dataset = (tf.compat.v1.data.Dataset.from_tensor_slices(
            (tf.constant(input_array)
            )).map(map_function).batch(batch_size).prefetch(1))

        iterator = dataset.make_initializable_iterator()
        data_to_be_prefetched_and_used = iterator.get_next()
        return data_to_be_prefetched_and_used, iterator

    def build_model1(self, input_array, c1, c2):
        # Convert the numpy array to TF Tensor
        input_f = tf.cast(input_array, tf.float32)

        # Define the Ops
        pl1 = tf.compat.v1.placeholder(dtype=dtypes.int32)
        pl1_f = tf.cast(pl1, tf.float32)
        pl2 = tf.compat.v1.placeholder(dtype=dtypes.int32)
        pl2_f = tf.cast(pl2, tf.float32)

        mul = tf.compat.v1.math.multiply(input_f, c1)
        add = tf.compat.v1.math.add(mul, pl2_f)
        add2 = add + pl1_f
        output = add2 - c2
        return output, pl1, pl2

    def __run_test(self, pipeline_creator, model):
        # build model
        input_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        map_multiplier = 10
        map_function = lambda x: x * map_multiplier
        batch_size = 1
        pipeline, iterator = pipeline_creator(input_array, map_function,
                                              batch_size)

        # some constants
        c1 = 5.0
        c2 = 10.0
        model, pl1, pl2 = model(pipeline, c1, c2)

        outputs = []

        sess = tf.compat.v1.Session()

        # Initialize the globals and the dataset
        sess.run(iterator.initializer)

        for i in range(1, 10):
            output = sess.run(model, feed_dict={pl1: i, pl2: i + 3})
            outputs.append(output)

        return outputs

    def test_prefetch1(self):
        # set flags
        prefetch_env = "NGRAPH_TF_USE_PREFETCH"
        env_var_map = self.store_env_variables([prefetch_env])
        self.set_env_variable(prefetch_env, "1")

        # Run on nGraph
        ng_outputs = self.__run_test(self.build_data_pipeline,
                                     self.build_model1)

        # Reset Graph
        tf.compat.v1.reset_default_graph()

        # Run on TF
        disable_tf = "NGRAPH_TF_DISABLE"
        self.set_env_variable(disable_tf, "1")
        tf_outputs = self.__run_test(self.build_data_pipeline,
                                     self.build_model1)

        # Compare Values
        assert np.allclose(ng_outputs, tf_outputs)

        # unset env variable
        self.unset_env_variable(prefetch_env)
        self.unset_env_variable(disable_tf)
        self.restore_env_variables(env_var_map)
