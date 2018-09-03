# ==============================================================================
#  Copyright 2018 Intel Corporation
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

from common import NgraphTest


class TestNgraphAPI(NgraphTest):
  def test_is_enabled(self, ngraph_device):
    assert ngraph_device.ngraph_is_enabled()

  def test_disable(self, ngraph_device):
    ngraph_device.ngraph_disable()
    assert ngraph_device.ngraph_is_enabled() == 0

  def test_enable(self, ngraph_device):
    ngraph_device.ngraph_enable()
    assert ngraph_device.ngraph_is_enabled() == 1

  def test_backends_len(self, ngraph_device):
    assert ngraph_device.ngraph_backends_len()

  def test_set_backend(self, ngraph_device):
    assert ngraph_device.ngraph_set_backend(b"CPU")

  def test_set_backend_invalid(self, ngraph_device):
    assert ngraph_device.ngraph_set_backend(b"POTATO") == 0

  def test_list_backends(self, ngraph_device):
    backends_count = ngraph_device.ngraph_backends_len()
    assert ngraph_device.ngraph_list_backends(
        (ctypes.c_char_p * backends_count)(None), backends_count) == 1

  def test_start_logging_placement(self, ngraph_device):
    ngraph_device.ngraph_start_logging_placement()
    assert ngraph_device.ngraph_is_logging_placement() == 1

  def test_stop_logging_placement(self, ngraph_device):
    ngraph_device.ngraph_stop_logging_placement()
    assert ngraph_device.ngraph_is_logging_placement() == 0
