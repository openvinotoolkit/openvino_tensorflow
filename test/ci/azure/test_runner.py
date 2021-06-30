#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
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
    Runs openvino_tensorflow tests. This script needs to be run after 
    running build_ovtf.py which builds the openvino-tensorflow and
    installs it to a virtual environment that would be used by this script.
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
    print("Openvino Tensorflow Backend set to:", backend)

    # Decide which tests to run
    if (arguments.test_cpp):
        test_filter = None
        os.environ['OPENVINO_TF_LOG_0_DISABLED'] = '1'
        run_ovtf_cpp_gtests(arguments.artifacts_dir, './', test_filter)
    elif (arguments.test_python):
        run_ovtf_pytests_from_artifacts(arguments.artifacts_dir)
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

    assert os.path.exists(root_pwd), "Path doesn't exist {0}".format(root_pwd)
    os.chdir(root_pwd)


if __name__ == '__main__':
    main()
