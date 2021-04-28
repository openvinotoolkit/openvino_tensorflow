# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow update_config api test

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
import openvino_tensorflow


class TestUpdateConfig(NgraphTest):

    @pytest.mark.skipif(
        not openvino_tensorflow.is_grappler_enabled(),
        reason='Only for Grappler')
    def test_update_config_adds_optimizer_only_once(self):

        # Helper function to count the number of occurances in a config
        def count_ng_optimizers(config):
            custom_opts = config.graph_options.rewrite_options.custom_optimizers
            count = 0
            for i in range(len(custom_opts)):
                if custom_opts[i].name == 'ovtf-optimizer':
                    count += 1
            return count

        # allow_soft_placement is set just to simulate
        # a real world non-empty initial ConfigProto
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        assert count_ng_optimizers(config) == 0
        config_new_1 = openvino_tensorflow.update_config(config)
        config_new_2 = openvino_tensorflow.update_config(config_new_1)
        assert count_ng_optimizers(config) == count_ng_optimizers(
            config_new_1) == count_ng_optimizers(config_new_2) == 1
