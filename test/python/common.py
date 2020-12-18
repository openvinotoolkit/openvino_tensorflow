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

import os
import platform
import random

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.core.protobuf import rewriter_config_pb2

from google.protobuf import text_format

import ngraph_bridge

__all__ = ['LIBNGRAPH_BRIDGE', 'NgraphTest']

_ext = 'dylib' if platform.system() == 'Darwin' else 'so'

LIBNGRAPH_BRIDGE = 'libngraph_bridge.' + _ext


class NgraphTest(object):

    def get_tensor(self, graph, tname, loading_from_protobuf):
        return graph.get_tensor_by_name(("", "import/")[loading_from_protobuf] +
                                        tname)

    def import_pbtxt(self, pb_filename):
        graph_def = tf.compat.v1.GraphDef()
        with open(pb_filename, "r") as f:
            text_format.Merge(f.read(), graph_def)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)
        return graph

    def with_ngraph(self, l, config=None):
        # Passing config as None and then initializing it inside
        # because mutable objects should not be used as defaults in python
        if config is None:
            config = tf.compat.v1.ConfigProto()
        # TODO: Stop grappler on failure (Add fail_on_optimizer_errors=True)
        config = ngraph_bridge.update_config(config)

        ngraph_tf_disable_deassign_clusters = os.environ.pop(
            'NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

        os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
        ngraph_bridge.enable()
        with tf.compat.v1.Session(config=config) as sess:
            retval = l(sess)

        os.environ.pop('NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

        if ngraph_tf_disable_deassign_clusters is not None:
            os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = \
                ngraph_tf_disable_deassign_clusters

        return retval

    def without_ngraph(self, l, config=None):
        if config is None:
            config = tf.compat.v1.ConfigProto()
        ngraph_tf_disable_deassign_clusters = os.environ.pop(
            'NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

        ngraph_bridge.disable()
        with tf.compat.v1.Session(config=config) as sess:
            retval = l(sess)

        if ngraph_tf_disable_deassign_clusters is not None:
            os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = \
                ngraph_tf_disable_deassign_clusters

        return retval

    # returns a vector of length 'vector_length' with random
    # float numbers in range [start,end]
    def generate_random_numbers(self,
                                vector_length,
                                start,
                                end,
                                datatype="DTYPE_FLOAT"):
        if datatype == "DTYPE_INT":
            return [random.randint(start, end) for i in range(vector_length)]
        return [random.uniform(start, end) for i in range(vector_length)]

    # returns true if the the env variable is set
    def is_env_variable_set(self, env_var):
        return env_var in os.environ

    # sets the env variable
    def set_env_variable(self, env_var, env_var_val):
        os.environ[env_var] = env_var_val
        print("Setting env variable ", env_var, " to ", env_var_val)

    # unset the env variable
    def unset_env_variable(self, env_var):
        if self.is_env_variable_set(env_var):
            os.environ.pop(env_var)
            print("Unset env variable ", env_var)

    # get the env variable
    def get_env_variable(self, env_var):
        env_var_val = os.getenv(env_var)
        print("Got env variable ", env_var, " set to ", env_var_val)
        return env_var_val

    # store env variables
    def store_env_variables(self, list_of_env_names):
        # store the env variables in map
        env_var_map = {}
        for env_var in list_of_env_names:
            if self.is_env_variable_set(env_var):
                env_backend_val = self.get_env_variable(env_var)
                env_var_map[env_var] = env_backend_val
                print("Got env backend", env_backend_val)
                os.environ.pop(env_var)
        return env_var_map

    # restore env variables
    def restore_env_variables(self, env_var_map):
        print("Restoring env varaibles")
        for k, v in env_var_map.items():
            self.set_env_variable(k, v)
