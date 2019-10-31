# ==============================================================================
#  Copyright 2019 Intel Corporation
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
"""nGraph TensorFlow bridge floor operation test
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform

import tensorflow as tf
from google.protobuf import text_format

import numpy as np
from common import NgraphTest
import os


class TestDumpingGraphs(NgraphTest):

    def get_tensor(self, graph, tname, tag):
        return graph.get_tensor_by_name(tag + tname)

    # Parameterized to have 0, 1 or more slashes
    @pytest.mark.parametrize(
        ('import_name_tag',),
        ((None,), ("",), ("import/scope1/scope2",), ("hello",), ("hello/",),
         ("hello//",), ("hello//a",), (".",), ("./",), ("a/.",), ("a/./b/",)))
    def test(self, import_name_tag):
        os.environ['NGRAPH_ENABLE_SERIALIZE'] = '1'
        # In this test we dump the serialized graph
        # This checks NgraphSerialize function
        # Specifically we want NgraphSerialize to work
        # even when there are '/' in the file name
        pb_filename = 'flib_graph_1.pbtxt'
        graph_def = tf.GraphDef()
        with open(pb_filename, "r") as f:
            text_format.Merge(f.read(), graph_def)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=import_name_tag)

        tag = (import_name_tag, "import/")[import_name_tag is None]
        if tag != "":
            if tag[-1] == '/':
                if tag[-2] == '/':
                    tag = tag[:-1]
            else:
                tag += '/'
        with graph.as_default() as g:
            x = self.get_tensor(g, "Placeholder:0", tag)
            y = self.get_tensor(g, "Placeholder_1:0", tag)
            z = self.get_tensor(g, "Placeholder_2:0", tag)

            a = self.get_tensor(g, "add_1:0", tag)
            b = self.get_tensor(g, "Sigmoid:0", tag)

            sess_fn = lambda sess: sess.run(
                b, feed_dict={i: np.full((2, 3), 1.0) for i in [x, y, z]})

            assert np.isclose(
                self.with_ngraph(sess_fn), 0.95257413 * np.ones([2, 3])).all()
            # This graph has a node named ngraph_cluster_0. If tag is "import"
            # we attempt to dump tf_function_import/ngraph_cluster_0.json
            # but we replace / with -- in NgraphSerialize
            # and dump tf_function_import--ngraph_cluster_0.json
            # When import_name_tag == "", we dump tf_function_ngraph_cluster_0.json

            expected_file = 'tf_function_' + tag.replace(
                "/", "--") + 'ngraph_cluster_0.json'
            assert expected_file in os.listdir('.')
            os.remove(expected_file)
