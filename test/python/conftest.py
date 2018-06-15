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
from ctypes import cdll
import glob
import os
import shutil

import pytest

from common import LIBNGRAPH_DEVICE


@pytest.fixture(scope='session', autouse=True)
def load_ngraph_device():
    cdll.LoadLibrary(LIBNGRAPH_DEVICE)


@pytest.fixture(scope='session', autouse=True)
def cleanup():
    yield
    for f in glob.glob('*.dot'):
        os.remove(f)
    for f in glob.glob('*.pbtxt'):
        os.remove(f)
    try:
        shutil.rmtree('cpu_codegen')
    except FileNotFoundError:
        pass
