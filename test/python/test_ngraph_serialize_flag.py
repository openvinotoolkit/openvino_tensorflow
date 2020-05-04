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
"""Pytest for a simple run on model testing framework

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform
import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import re

from common import NgraphTest
import ngraph_bridge


class TestNgraphSerialize(NgraphTest):

    def test_ng_serialize_to_json(self):
        initial_contents = set(os.listdir())
        xshape = (3, 4, 5)
        x = tf.compat.v1.placeholder(tf.float32, shape=xshape)
        out = tf.nn.l2_loss(tf.abs(x))
        values = np.random.rand(*xshape)

        config = ngraph_bridge.update_config(tf.compat.v1.ConfigProto())
        ngraph_enable_serialize = os.environ.pop('NGRAPH_ENABLE_SERIALIZE',
                                                 None)
        os.environ['NGRAPH_ENABLE_SERIALIZE'] = '1'
        ngraph_bridge.enable()
        with tf.compat.v1.Session(config=config) as sess:
            out = sess.run((out), feed_dict={x: values})
        os.environ.pop('NGRAPH_ENABLE_SERIALIZE', None)
        if ngraph_enable_serialize is not None:
            os.environ['NGRAPH_ENABLE_SERIALIZE'] = \
                ngraph_enable_serialize

        final_contents = set(os.listdir())
        assert (len(final_contents) - len(initial_contents) == 1)
        new_files = final_contents.difference(initial_contents)
        flname = new_files.pop()
        assert (flname.startswith('tf_function_') and flname.endswith('json'))
        os.remove(flname)
