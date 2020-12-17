#!/usr/bin/env python3
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
import argparse
import errno
import os
from subprocess import check_output, call
import sys
import shutil
import glob
import platform
from distutils.sysconfig import get_python_lib

from tools.test_utils import *


def main():
    '''
    Tests nGraph-TensorFlow Python 3. This script needs to be run after
    running build_ngtf.py which builds the ngraph-tensorflow-bridge
    and installs it to a virtual environment that would be used by this script.
    '''
    root_pwd = os.getcwd()

    build_dir = 'build_cmake'
    venv_dir = 'build_cmake/venv-tf-py3'
    artifacts_dir = os.path.join(build_dir, 'artifacts')

    load_venv(venv_dir)

    # First run the C++ gtests
    run_ngtf_cpp_gtests(artifacts_dir, './', None)

    # Next run Python unit tests
    run_ngtf_pytests_from_artifacts(artifacts_dir)

    # Finally run Resnet50
    run_resnet50_infer_from_artifacts(artifacts_dir, 1, 1)
    os.chdir(root_pwd)


if __name__ == '__main__':
    main()
