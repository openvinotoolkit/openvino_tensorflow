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
"""nGraph TensorFlow bridge floor operation test

"""
from __future__ import absolute_import

import ctypes
import pytest

from common import NgraphTest
import ngraph_bridge


class TestNgraphAPI(NgraphTest):

    def test_disable(self):
        ngraph_bridge.disable()
        assert ngraph_bridge.is_enabled() == 0

    def test_enable(self):
        ngraph_bridge.enable()
        assert ngraph_bridge.is_enabled() == 1

    def test_set_backend_invalid(self):
        env_var_map = self.store_env_variables(["NGRAPH_TF_BACKEND"])
        self.unset_env_variable("NGRAPH_TF_BACKEND")
        current_backend = ngraph_bridge.get_backend()
        error_thrown = False
        try:
            ngraph_bridge.set_backend('POTATO')
        except:
            error_thrown = True
        ngraph_bridge.set_backend(current_backend)
        self.restore_env_variables(env_var_map)
        assert error_thrown

    def test_list_backends(self):
        assert len(ngraph_bridge.list_backends())

    def test_start_logging_placement(self):
        ngraph_bridge.start_logging_placement()
        assert ngraph_bridge.is_logging_placement() == 1

    def test_stop_logging_placement(self):
        ngraph_bridge.stop_logging_placement()
        assert ngraph_bridge.is_logging_placement() == 0
