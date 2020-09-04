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
"""nGraph TensorFlow bridge test for checking backend setting using rewriter config for grappler

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import os
import numpy as np
import shutil
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.core.protobuf import rewriter_config_pb2
import ngraph_bridge

from common import NgraphTest


class TestRewriterConfigBackendSetting(NgraphTest):

    @pytest.mark.skipif(
        not ngraph_bridge.is_grappler_enabled(),
        reason='Rewriter config only works for grappler path')
    def test_config_updater_api(self):
        dim1 = 3
        dim2 = 4
        a = tf.compat.v1.placeholder(tf.float32, shape=(dim1, dim2), name='a')
        x = tf.compat.v1.placeholder(tf.float32, shape=(dim1, dim2), name='x')
        b = tf.compat.v1.placeholder(tf.float32, shape=(dim1, dim2), name='y')
        axpy = (a * x) + b

        config = tf.compat.v1.ConfigProto()
        rewriter_options = rewriter_config_pb2.RewriterConfig()
        rewriter_options.meta_optimizer_iterations = (
            rewriter_config_pb2.RewriterConfig.ONE)
        rewriter_options.min_graph_nodes = -1
        ngraph_optimizer = rewriter_options.custom_optimizers.add()
        ngraph_optimizer.name = "ngraph-optimizer"
        config.MergeFrom(
            tf.compat.v1.ConfigProto(
                graph_options=tf.compat.v1.GraphOptions(
                    rewrite_options=rewriter_options)))

        with tf.compat.v1.Session(config=config) as sess:
            outval = sess.run(
                axpy,
                feed_dict={
                    a: 1.5 * np.ones((dim1, dim2)),
                    b: np.ones((dim1, dim2)),
                    x: np.ones((dim1, dim2))
                })
        assert (outval == 2.5 * (np.ones((dim1, dim2)))).all()
