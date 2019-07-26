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
"""nGraph TensorFlow bridge test for tf2ngraph script for precompilation

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import os
import numpy as np
import shutil
import tensorflow as tf
import ngraph_bridge

from tools.build_utils import command_executor
from tools.tf2ngraph import convert, get_gdef

from common import NgraphTest


class Testtf2ngraph(NgraphTest):

    # utility function to make sure input format and location match
    @staticmethod
    def format_and_loc_match(format, loc):
        assert format in ['pb', 'pbtxt', 'savedmodel']
        implies = lambda p, q: (not p) or (p and q)
        #if its pbtxt, file name has pbtxt AND if its savedmodel, file name does not have pbtxt
        return implies(
            format == 'pbtxt', 'pbtxt' == loc.split('.')[-1]) and implies(
                format == 'savedmodel', 'pbtxt' != loc.split('.')[-1])

    @pytest.mark.parametrize(('commandline'),
                             (True,))  #TODO: add False for functional api test
    @pytest.mark.parametrize(('inp_format', 'inp_loc'), (
        ('pbtxt', 'sample_graph.pbtxt'),
        ('savedmodel', 'sample_graph'),
        ('pb', 'sample_graph.pb'),
        ('pbtxt', 'sample_graph_nodevice.pbtxt'),
    ))
    @pytest.mark.parametrize(('out_format',), (
        ('pbtxt',),
        ('pb',),
        ('savedmodel',),
    ))
    @pytest.mark.parametrize(('ng_device',), (('CPU',), ('INTERPRETER',)))
    def test_command_line_api(self, inp_format, inp_loc, out_format,
                              commandline, ng_device):
        # Only run this test when grappler is enabled
        if not ngraph_bridge.is_grappler_enabled():
            return
        assert Testtf2ngraph.format_and_loc_match(inp_format, inp_loc)
        out_loc = inp_loc.split('.')[0] + '_modified' + (
            '' if out_format == 'savedmodel' else ('.' + out_format))
        try:
            (shutil.rmtree, os.remove)[os.path.isfile(out_loc)](out_loc)
        except:
            pass
        conversion_successful = False
        try:
            extra_params = {
                'CPU': '{device_config:0}',
                'INTERPRETER': '{test_echo:1}'
            }[ng_device]
            if commandline:
                # In CI this test is expected to be run out of artifacts/test/python
                command_executor('python ../../tools/tf2ngraph.py --input_' +
                                 inp_format + ' ' + inp_loc +
                                 ' --output_nodes out_node --output_' +
                                 out_format + ' ' + out_loc + ' --ng_backend ' +
                                 ng_device + ' --extra_params ' + extra_params)
            else:
                convert(inp_format, inp_loc, out_format, out_loc, ['out_node'],
                        ng_device, extra_params)
            conversion_successful = True
        finally:
            if not conversion_successful:
                try:
                    (shutil.rmtree, os.remove)[os.path.isfile(out_loc)](out_loc)
                except:
                    pass
        assert conversion_successful

        gdef = get_gdef(out_format, out_loc)
        (shutil.rmtree, os.remove)[os.path.isfile(out_loc)](out_loc)

        with tf.Graph().as_default() as g:
            tf.import_graph_def(gdef, name='')
            # The graph should have exactly one encapsulate
            assert len([
                0 for i in g.get_operations() if i.type == 'NGraphEncapsulate'
            ]) == 1
            # TODO: check that the encapsulate op has correct backend and extra params attached to it
            x = self.get_tensor(g, "x:0", False)
            y = self.get_tensor(g, "y:0", False)
            out = self.get_tensor(g, "out_node:0", False)

            sess_fn = lambda sess: sess.run(
                [out], feed_dict={i: np.zeros((10,)) for i in [x, y]})

            res1 = self.with_ngraph(sess_fn)
            res2 = self.without_ngraph(sess_fn)

            exp = [0.5 * np.ones((10,))]
            # Note both run on Host (because NgraphEncapsulate can only run on host)
            assert np.isclose(res1, res2).all()
            # Comparing with expected value
            assert np.isclose(res1, exp).all()
