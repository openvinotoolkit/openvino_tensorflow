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
        '--test_examples',
        help="Builds and tests the examples.\n",
        action="store_true")

    parser.add_argument(
        '--gpu_unit_tests_enable',
        help="Builds and tests the examples.\n",
        action="store_true")

    arguments = parser.parse_args()

    #-------------------------------
    # Recipe
    #-------------------------------

    root_pwd = os.getcwd()

    # Constants
    build_dir = 'build_cmake'
    venv_dir = 'build_cmake/venv-tf-py3'
    tf_src_dir = 'build_cmake/tensorflow'

    if (platform.system() != 'Darwin'):
        # Run the bazel based buil
        run_bazel_build_test(venv_dir, build_dir)

    # First run the C++ gtests
    run_ngtf_gtests(build_dir, None)

    # If the GPU tests are requested, then run them as well
    if (arguments.gpu_unit_tests_enable):
        os.environ['NGRAPH_TF_BACKEND'] = 'GPU'
        run_ngtf_gtests(
            build_dir,
            str("-ArrayOps.Quanti*:ArrayOps.Dequant*:BackendManager.BackendAssignment:"
                "MathOps.AnyKeepDims:MathOps.AnyNegativeAxis:MathOps.AnyPositiveAxis:"
                "MathOps.AllKeepDims:MathOps.AllNegativeAxis:MathOps.AllPositiveAxis:"
                "NNOps.Qu*:NNOps.SoftmaxZeroDimTest*:"
                "NNOps.SparseSoftmaxCrossEntropyWithLogits"))

    os.environ['NGRAPH_TF_BACKEND'] = 'CPU'

    # Next run Python unit tests
    load_venv(venv_dir)
    run_ngtf_pytests(venv_dir, build_dir)

    if (arguments.test_examples):
        # Run the C++ example build/run test
        run_cpp_example_test('build')

    # Next run the TensorFlow python tests
    run_tensorflow_pytests(venv_dir, build_dir, './', tf_src_dir)

    # Finally run Resnet50 based training and inferences
    run_resnet50(build_dir)

    os.chdir(root_pwd)


if __name__ == '__main__':
    main()
