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
"""nGraph TensorFlow bridge test for tf2ngraph script for precompilation with shape hints

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
import ngraph_bridge
import json

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tools.build_utils import command_executor
from tools.tf2ngraph import convert, get_gdef, Tf2ngraphJson

from common import NgraphTest


def get_pbtxt_name(tag, p0_shape, p1_shape):
    return tag + ','.join(map(lambda x: str(x), p0_shape)) + '__' + ','.join(
        map(lambda x: str(x), p1_shape)) + '.pbtxt'


def create_graph(p0_shape, p1_shape):
    temp_pbtxt_name = get_pbtxt_name('temp_graph_in_', p0_shape, p1_shape)
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(tf.float32, shape=p0_shape, name='x')
        y = tf.compat.v1.placeholder(tf.float32, shape=p1_shape, name='y')
        z = tf.add(tf.abs(x), tf.abs(y), name="z")
        tf.io.write_graph(sess.graph, '.', temp_pbtxt_name, as_text=True)
    return x, y, z, temp_pbtxt_name


def get_inputs(p_shape):
    return np.random.rand(*p_shape)


def run_pbtxt(pbtxt_filename, inp0, inp1):
    tf.compat.v1.reset_default_graph()
    gdef = graph_pb2.GraphDef()
    with open(pbtxt_filename, 'r') as f:
        raw_contents = f.read()
    text_format.Parse(raw_contents, gdef)
    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(gdef, name='')
        x = tf.compat.v1.get_default_graph().get_tensor_by_name("x:0")
        y = tf.compat.v1.get_default_graph().get_tensor_by_name("y:0")
        z = tf.compat.v1.get_default_graph().get_tensor_by_name("z:0")
        return sess.run(z, feed_dict={x: inp0, y: inp1})


def check_pbtxt_has_exec(pbtxt_filename, num_expected_execs):
    with open(pbtxt_filename, 'r') as f:
        contents = '\n'.join(f.readlines())
        assert contents.count('_ngraph_aot_requested') == 1
        assert contents.count('_ngraph_aot_ngexec_') == num_expected_execs
        assert contents.count('_ngraph_aot_ngfunction_') == num_expected_execs


def helper(p0_shape, p1_shape, p0_actual_shape, p1_actual_shape, shapehints):
    inp0 = get_inputs(p0_actual_shape)
    inp1 = get_inputs(p1_actual_shape)
    x, y, z, temp_in_pbtxt_name = create_graph(p0_shape, p1_shape)
    temp_out_pbtxt_name = get_pbtxt_name('temp_graph_out_', p0_shape, p1_shape)
    json_name = 'temp_config_file.json'
    # shapehints is a list of dictionaries (keys are node names, vals are lists (of shapes))
    Tf2ngraphJson.dump_json(json_name, None, shapehints)

    command_executor('python ../../tools/tf2ngraph.py --input_pbtxt ' +
                     temp_in_pbtxt_name + ' --output_nodes z --output_pbtxt ' +
                     temp_out_pbtxt_name + ' --ng_backend INTERPRETER ' +
                     ' --config_file ' + json_name + ' --precompile')

    num_expected_execs = (len(shapehints), 1)[len(shapehints) == 0]
    check_pbtxt_has_exec(temp_out_pbtxt_name, num_expected_execs)

    tf_out_val = run_pbtxt(temp_in_pbtxt_name, inp0, inp1)
    ng_out_vals = run_pbtxt(temp_out_pbtxt_name, inp0, inp1)
    assert ((tf_out_val == ng_out_vals).all())

    os.remove(temp_in_pbtxt_name)
    os.remove(temp_out_pbtxt_name)
    os.remove(json_name)


# TODO: Add more test cases
class Testtf2ngraphShapehints(NgraphTest):

    @pytest.mark.parametrize(
        ('p0_shape', 'p1_shape', 'p0_actual_shape', 'p1_actual_shape',
         'shapehints'),
        (
            ([2, 2], [2, 2], [2, 2], [2, 2], [{}
                                             ]),  # np input needs shape hints
            ([2, 2], [None, 2], [2, 2], [2, 2], [{
                'y': [2, -1]
            }]),  # only 1 input needs shape hints
            (
                [2, None],
                [None, 3],
                [2, 3],
                [2, 3],
                [{
                    'y': [2, -1],
                    'x': [2, 3]  # both inputs need shape hints
                }]),
            ([None, None], [None, None], [5, 1], [5, 1], [{
                'y': [2, 3],
                'x': [2, 3]
            }, {
                'y': [5, 1],
                'x': [5, 1]
            }]),  # 2 executables are compiled
        ))
    @pytest.mark.skipif(
        not ngraph_bridge.is_grappler_enabled(),
        reason="Requires grappler build for tf2ngraph and AOT")
    def test_tf2ngraph_with_shape_hints_0(self, p0_shape, p1_shape,
                                          p0_actual_shape, p1_actual_shape,
                                          shapehints):
        helper(p0_shape, p1_shape, p0_actual_shape, p1_actual_shape, shapehints)

    @pytest.mark.parametrize(
        ('p0_shape', 'p1_shape', 'p0_actual_shape', 'p1_actual_shape',
         'shapehints'),
        (
            ([2, 2], [None, 2], [2, 2], [2, 2], [{
                'y': [2, 3]
            }]),  # conflicting shape hint
            ([2, 2], [None, 2], [2, 2], [2, 2], [{
                'y': [2]
            }]),  # shape hint is of conflicting rank
            ([2, 2], [None, 2], [2, 5], [2, 5], [{
                'y': [2, 2]
            }]),  # During run time bad shapes are passed
            ([2, 2], [None, 2], [2, 2], [2, 2], [{
                'x': [2, -1]
            }]),  # Input y does not have enough hints to concretize it
            ([2, 2], [None, 2], [2, 2], [2, 2], [{
                'y': [2, -1],
                'bogus': [1, 2]
            }]),  # passing a bogus node name
        ))
    @pytest.mark.skipif(
        not ngraph_bridge.is_grappler_enabled(),
        reason="Requires grappler build for tf2ngraph and AOT")
    def test_tf2ngraph_with_shape_hints_1(self, p0_shape, p1_shape,
                                          p0_actual_shape, p1_actual_shape,
                                          shapehints):
        with pytest.raises(Exception):
            helper(p0_shape, p1_shape, p0_actual_shape, p1_actual_shape,
                   shapehints)
