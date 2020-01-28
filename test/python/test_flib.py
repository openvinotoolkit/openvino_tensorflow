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
"""nGraph TensorFlow bridge floor operation test
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform

import tensorflow as tf
import numpy as np
from common import NgraphTest


class TestFlibOperations(NgraphTest):

    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Only for Linux')
    def test_flib_1(self):
        graph = self.import_pbtxt('flib_graph_1.pbtxt')
        with graph.as_default() as g:

            x = self.get_tensor(g, "Placeholder:0", True)
            y = self.get_tensor(g, "Placeholder_1:0", True)
            z = self.get_tensor(g, "Placeholder_2:0", True)

            a = self.get_tensor(g, "add_1:0", True)
            b = self.get_tensor(g, "Sigmoid:0", True)

            sess_fn = lambda sess: sess.run(
                [a, b], feed_dict={i: np.full((2, 3), 1.0) for i in [x, y, z]})

            res1 = self.with_ngraph(sess_fn)
            res2 = self.without_ngraph(sess_fn)

            exp = [np.full((2, 3), 3.0), np.full((2, 3), 0.95257413)]
            # Note both run on Host (because NgraphEncapsulate can only run on host)
            assert np.isclose(res1, res2).all()
            # Comparing with expected value
            assert np.isclose(res1, exp).all()

    @pytest.mark.skip(reason="Not passing through grappler")
    def test_flib_2(self):
        graph = self.import_pbtxt('flib_graph_2.pbtxt')
        with graph.as_default() as g:

            x = self.get_tensor(g, "Variable_2/peek/_2:0", True)
            y = self.get_tensor(g, "Variable_1/peek/_3:0", True)
            z = self.get_tensor(g, "Variable/peek/_4:0", True)

            a = self.get_tensor(g, "add_1:0", True)
            b = self.get_tensor(g, "Sigmoid:0", True)

            def sess_fn(sess):
                #sess.run(tf.global_variables_initializer())
                return sess.run(
                    [a, b],
                    feed_dict={i: np.full((2, 3), 1.0) for i in [x, y, z]})

            res1 = self.with_ngraph(sess_fn)
            res2 = self.without_ngraph(sess_fn)
            exp = [np.full((2, 3), 3.0), np.full((2, 3), 0.95257413)]
            # Note both run on Host (because NgraphEncapsulate can only run on host)
            assert np.isclose(res1, res2).all()
            # Comparing with expected value
            assert np.isclose(res1, exp).all()  #fails
