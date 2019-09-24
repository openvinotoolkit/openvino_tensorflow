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
"""Pytest for a simple run on model testing framework

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import platform
import os

from common import NgraphTest
from tools.build_utils import command_executor
import ngraph_bridge


class TestModelTester(NgraphTest):

    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Only for Linux')
    def test_MLP(self):
        cwd = os.getcwd()
        os.chdir('../model_level_tests/')
        grappler = ngraph_bridge.is_grappler_enabled()
        varopts = ngraph_bridge.are_variables_enabled()
        if grappler:
            if varopts:
                assert False, "Varopts and grappler does not build together right now"
            else:
                config = "grappler"
        else:
            config = "varopts" if varopts else "default"
        try:
            command_executor(
                "python test_main.py --run_basic_tests --models MLP --ignore_test time --configuration "
                + config)
        finally:
            os.chdir(cwd)
