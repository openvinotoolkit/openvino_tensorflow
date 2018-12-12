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

from build_ngtf import load_venv


def main():
    '''
    Tests nGraph-TensorFlow Python 3. This script needs to be run after 
    running build_ngtf.py which builds the ngraph-tensorflow-bridge
    and installs it to a virtual environment that would be used by this script.
    '''
    parser = argparse.ArgumentParser()
    arguments = parser.parse_args()

    #-------------------------------
    # Recipe
    #-------------------------------

    # Go to the build directory
    # mkdir build directory
    build_dir = 'build'

    root_pwd = os.getcwd()
    os.chdir(build_dir)

    # Component versions
    venv_dir = 'venv-tf-py3'

    # Load the virtual env
    venv_dir_absolute = load_venv(venv_dir)

    # First run the C++ gtests
    os.chdir("test")

    result = call(["./gtest_ngtf"])
    if (result != 0):
        raise Exception("Error running command: gtest_ngtf")

    # Next run the ngraph-tensorflow python tests
    if call(["pip", "install", "-U", "pytest"]) != 0:
        raise Exception("Error running command: pip install ")

    if call(["pip", "install", "-U", "psutil"]) != 0:
        raise Exception("Error running command: pip install psutil")

    os.chdir("python")
    if call(["python", "-m", "pytest"]) != 0:
        raise Exception("Error running test command: python -m  pytest")

    os.chdir("..")
    os.chdir("..")

    # Next patch the TensorFlow so that the tests run using ngraph_bridge
    pwd = os.getcwd()

    # Go to the site-packages
    os.chdir(glob.glob(venv_dir_absolute + "/lib/py*/site-packages")[0])
    print("CURRENT DIR: " + os.getcwd())

    patch_file = os.path.abspath(
        os.path.join(root_pwd,
                     "test/python/tensorflow/tf_unittest_ngraph.patch"))

    print("Patching TensorFlow using: %s" % patch_file)
    result = call(["patch", "-p1", "-N", "-i", patch_file])
    print("Patch result: %d" % result)
    os.chdir(pwd)

    # Now run the TensorFlow python tests
    tensorflow_src_dir = os.path.join(root_pwd, "build/tensorflow")
    test_src_dir = os.path.join(root_pwd, "test/python/tensorflow")
    test_script = os.path.join(test_src_dir, "tf_unittest_runner.py")
    test_manifest_file = os.path.join(test_src_dir, "python_tests_list.txt")

    import psutil
    num_cores = int(psutil.cpu_count(logical=False))
    print("OMP_NUM_THREADS: %s " % str(num_cores))
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'

    if call([
            "python", test_script, "--tensorflow_path", tensorflow_src_dir,
            "--run_tests_from_file", test_manifest_file
    ]) != 0:
        raise Exception("Error running TensorFlow python tests")

    os.chdir(root_pwd)


if __name__ == '__main__':
    main()
