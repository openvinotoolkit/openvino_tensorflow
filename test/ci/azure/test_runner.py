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
    Runs ov_tf tests. This script needs to be run after 
    running build_ngtf.py which builds the ov_tf
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
        '--test_resnet50_infer',
        help="Runs ResNet50 inference from IntelAI models.\n",
        action="store_true")

    parser.add_argument(
        '--artifacts_dir',
        type=str,
        help=
        "Location of the artifacts that would be used for running the tests\n",
        action="store")

    arguments = parser.parse_args()
    root_pwd = os.getcwd()

    # Check for mandetary parameters
    if not arguments.artifacts_dir:
        raise Exception("Need to specify --artifacts_dir")

    # Set the backend if specified
    backend = TestEnv.BACKEND()
    print("ov_tf Backend set to:", backend)

    # Decide which tests to run
    if (arguments.test_cpp):
        test_filter = None
        os.environ['OPENVINO_TF_LOG_0_DISABLED'] = '1'
        run_ngtf_cpp_gtests(arguments.artifacts_dir, './', test_filter)
    elif (arguments.test_python):
        run_ngtf_pytests_from_artifacts(arguments.artifacts_dir)
    elif (arguments.test_tf_python):
        os.environ['OPENVINO_TF_LOG_0_DISABLED'] = '1'
        run_tensorflow_pytests_from_artifacts(
            './', arguments.artifacts_dir + '/tensorflow/python', False)
    elif (arguments.test_resnet):
        if TestEnv.is_osx():
            run_resnet50_forward_pass_from_artifacts(
                './', arguments.artifacts_dir, 1, 32)
        else:
            batch_size = 128
            iterations = 10
            run_resnet50_from_artifacts('./', arguments.artifacts_dir,
                                        batch_size, iterations)
    elif (arguments.test_resnet50_infer):
        if TestEnv.is_osx():
            raise Exception("RN50 inference test not supported on Darwin/OSX")
        else:
            batch_size = 128
            iterations = 10
            if backend != 'CPU':
                batch_size = 1
                iterations = 1
            run_resnet50_infer_from_artifacts(arguments.artifacts_dir,
                                              batch_size, iterations)
    else:
        raise Exception("No tests specified")

    os.chdir(root_pwd)


if __name__ == '__main__':
    main()
