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
    Tests openvino tensorflow Python 3. This script needs to be run after
    running build_ovtf.py which builds the openvino_tensorflow
    and installs it to a virtual environment that would be used by this script.
    '''
    root_pwd = os.getcwd()

    build_dir = 'build_cmake'
    venv_dir = 'build_cmake/venv-tf-py3'
    artifacts_dir = os.path.join(build_dir, 'artifacts')

    load_venv(venv_dir)

    # First run the C++ gtests
    run_ovtf_cpp_gtests(artifacts_dir, './', None)

    # Next run Python unit tests
    run_ovtf_pytests_from_artifacts(artifacts_dir)

    # Finally run Resnet50
    run_resnet50_infer_from_artifacts(artifacts_dir, 1, 1)
    os.chdir(root_pwd)


if __name__ == '__main__':
    main()
