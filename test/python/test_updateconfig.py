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
"""nGraph TensorFlow bridge update_config api test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.core.protobuf import rewriter_config_pb2

from common import NgraphTest
import ngraph_bridge


class TestUpdateConfig(NgraphTest):

    @pytest.mark.skipif(
        not ngraph_bridge.is_grappler_enabled(), reason='Only for Grappler')
    def test_update_config_adds_optimizer_only_once(self):

        # Helper function to count the number of occurances in a config
        def count_ng_optimizers(config):
            custom_opts = config.graph_options.rewrite_options.custom_optimizers
            count = 0
            for i in range(len(custom_opts)):
                if custom_opts[i].name == 'ngraph-optimizer':
                    count += 1
            return count

        # allow_soft_placement is set just to simulate
        # a real world non-empty initial ConfigProto
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        assert count_ng_optimizers(config) == 0
        config_new_1 = ngraph_bridge.update_config(config)
        config_new_2 = ngraph_bridge.update_config(config_new_1)
        assert count_ng_optimizers(config) == count_ng_optimizers(
            config_new_1) == count_ng_optimizers(config_new_2) == 1
