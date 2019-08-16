# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
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

    def test_backends_len(self):
        assert ngraph_bridge.backends_len()

    def test_set_backend(self):
        ngraph_bridge.set_backend('CPU')
        assert ngraph_bridge.get_currently_set_backend_name() == "CPU"

    def test_set_backend_invalid(self):
        try:
            ngraph_bridge.set_backend('POTATO')
            error_thrown = False
        except:
            error_thrown = True
        assert error_thrown

    def test_list_backends(self):
        backends_count = ngraph_bridge.backends_len()
        assert len(ngraph_bridge.list_backends()) == backends_count

    def test_start_logging_placement(self):
        ngraph_bridge.start_logging_placement()
        assert ngraph_bridge.is_logging_placement() == 1

    def test_stop_logging_placement(self):
        ngraph_bridge.stop_logging_placement()
        assert ngraph_bridge.is_logging_placement() == 0
