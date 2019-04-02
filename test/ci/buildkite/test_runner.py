#!/usr/bin/env python3
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
import argparse
import errno
import os
from subprocess import check_output, call
import sys
import shutil
import glob
import platform
from distutils.sysconfig import get_python_lib

#from tools.build_utils import load_venv, command_executor
from tools.test_utils import *


def main():
    '''
    Tests nGraph-TensorFlow Python 3. This script needs to be run after 
    running build_ngtf.py which builds the ngraph-tensorflow-bridge
    and installs it to a virtual environment that would be used by this script.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_cpp',
        help="Runs C++ tests (GTest based).\n",
        action="store_true")

    parser.add_argument(
        '--test_python',
        help="Runs Python tests (Pytest based).\n",
        action="store_true")

    parser.add_argument(
        '--test_tf_python',
        help="Runs TensorFlow Python tests (Pytest based).\n",
        action="store_true")

    parser.add_argument(
        '--test_resnet',
        help="Runs TensorFlow Python tests (Pytest based).\n",
        action="store_true")

    parser.add_argument(
        '--artifacts_dir',
        type=str,
        help=
        "Location of the artifacts that would be used for running the tests\n",
        action="store")

    arguments = parser.parse_args()

    #-------------------------------
    # Recipe
    #-------------------------------

    root_pwd = os.getcwd()

    # Check for mandetary parameters
    if not arguments.artifacts_dir:
        raise Exception("Need to specify --artifacts_dir")

    # Decide which tests to run
    if (arguments.test_cpp):
        run_ngtf_cpp_gtests(arguments.artifacts_dir, './', None)
    elif (arguments.test_python):
        run_ngtf_pytests_from_artifacts(arguments.artifacts_dir)
    elif (arguments.test_tf_python):
        raise Exception("TensorFlow Python tests are not yet supported")
    elif (arguments.test_resnet):
        run_resnet50_from_artifacts(arguments.artifacts_dir)
    else:
        raise Exception("No tests specified")

    os.chdir(root_pwd)


if __name__ == '__main__':
    main()
