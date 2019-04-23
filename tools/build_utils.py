#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2019 Intel Corporation
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
from argparse import RawTextHelpFormatter

import errno
import os
from subprocess import check_output, call
import sys
import shutil
import glob
import platform
import shlex


def command_executor(cmd, verbose=False, msg=None, stdout=None):
    '''
    Executes the command.
    Example: 
      - command_executor('ls -lrt')
      - command_executor(['ls', '-lrt'])
    '''
    if type(cmd) == type([]):  #if its a list, convert to string
        cmd = ' '.join(cmd)
    if verbose:
        tag = 'Running COMMAND: ' if msg is None else msg
        print(tag + cmd)
    if (call(shlex.split(cmd), stdout=stdout) != 0):
        raise Exception("Error running command: " + cmd)


def build_ngraph(build_dir, src_location, cmake_flags, verbose):
    pwd = os.getcwd()

    src_location = os.path.abspath(src_location)
    print("Source location: " + src_location)

    os.chdir(src_location)

    # mkdir build directory
    path = build_dir
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

    # Run cmake
    os.chdir(build_dir)

    cmake_cmd = ["cmake"]
    cmake_cmd.extend(cmake_flags)
    cmake_cmd.extend([src_location])

    command_executor(cmake_cmd, verbose=True)

    import psutil
    num_cores = str(psutil.cpu_count(logical=True))
    cmd = ["make", "-j" + num_cores]
    if verbose:
        cmd.extend(['VERBOSE=1'])
    command_executor(cmd, verbose=True)
    cmd = ["make", "install"]
    command_executor(cmd, verbose=True)

    os.chdir(pwd)


def install_virtual_env(venv_dir):
    # Check if we have virtual environment
    # TODO

    # Setup virtual environment
    venv_dir = os.path.abspath(venv_dir)
    # Note: We assume that we are using Python 3 (as this script is also being
    # executed under Python 3 as marked in line 1)
    command_executor(["virtualenv", "-p", "python3", venv_dir])


def load_venv(venv_dir):
    venv_dir = os.path.abspath(venv_dir)

    # Check if we are already inside the virtual environment
    # return (hasattr(sys, 'real_prefix')
    #         or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    print("Loading virtual environment from: %s" % venv_dir)

    activate_this_file = venv_dir + "/bin/activate_this.py"
    # The execfile API is for Python 2. We keep here just in case you are on an
    # obscure system without Python 3
    # execfile(activate_this_file, dict(__file__=activate_this_file))
    exec(
        compile(
            open(activate_this_file, "rb").read(), activate_this_file, 'exec'),
        dict(__file__=activate_this_file), dict(__file__=activate_this_file))

    return venv_dir


def setup_venv(venv_dir):
    load_venv(venv_dir)

    print("PIP location")
    call(['which', 'pip'])

    # Patch the MacOS pip to avoid the TLS issue
    if (platform.system() == 'Darwin'):
        get_pip = open("get-pip.py", "wb")
        call([
            "curl",
            "https://bootstrap.pypa.io/get-pip.py",
        ], stdout=get_pip)
        call(["python3", "./get-pip.py"])

    # Install the pip packages
    package_list = [
        "pip",
        "install",
        "-U",
        "pip",
        "setuptools",
        "psutil",
        "six>=1.10.0",
        "numpy>=1.13.3",
        "absl-py>=0.1.6",
        "astor>=0.6.0",
        "google_pasta>=0.1.1",
        "wheel>=0.26",
        "mock",
        "termcolor>=1.1.0",
        "protobuf>=3.6.1",
        "keras_applications>=1.0.6",
        "--no-deps",
        "keras_preprocessing==1.0.5",
        "--no-deps",
        "yapf==0.26.0",
    ]
    command_executor(package_list)

    # Print the current packages
    command_executor(["pip", "list"])


def build_tensorflow(venv_dir, src_dir, artifacts_dir, target_arch, verbosity):

    base = sys.prefix
    python_lib_path = os.path.join(base, 'lib', 'python%s' % sys.version[:3],
                                   'site-packages')
    python_executable = os.path.join(base, "bin", "python")

    print("PYTHON_BIN_PATH: " + python_executable)

    # In order to build TensorFlow, we need to be in the virtual environment
    pwd = os.getcwd()

    src_dir = os.path.abspath(src_dir)
    print("SOURCE DIR: " + src_dir)

    # Update the artifacts directory
    artifacts_dir = os.path.join(os.path.abspath(artifacts_dir), "tensorflow")
    print("ARTIFACTS DIR: %s" % artifacts_dir)

    os.chdir(src_dir)

    # Set the TensorFlow configuration related variables
    os.environ["PYTHON_BIN_PATH"] = python_executable
    os.environ["PYTHON_LIB_PATH"] = python_lib_path
    os.environ["TF_NEED_IGNITE"] = "0"
    if (platform.system() == 'Darwin'):
        os.environ["TF_ENABLE_XLA"] = "0"
    else:
        os.environ["TF_ENABLE_XLA"] = "1"
    os.environ["TF_NEED_OPENCL_SYCL"] = "0"
    os.environ["TF_NEED_COMPUTECPP"] = "0"
    os.environ["TF_NEED_ROCM"] = "0"
    os.environ["TF_NEED_MPI"] = "0"
    os.environ["TF_NEED_CUDA"] = "0"
    os.environ["TF_DOWNLOAD_CLANG"] = "0"
    os.environ["TF_SET_ANDROID_WORKSPACE"] = "0"
    os.environ["CC_OPT_FLAGS"] = "-march=" + target_arch

    command_executor("./configure")

    # Build the python package
    cmd = [
        "bazel",
        "build",
        "--config=opt",
        "//tensorflow/tools/pip_package:build_pip_package",
    ]
    if verbosity:
        cmd.extend(['-s'])

    command_executor(cmd)

    command_executor([
        "bazel-bin/tensorflow/tools/pip_package/build_pip_package",
        artifacts_dir
    ])

    # Get the name of the TensorFlow pip package
    tf_wheel_files = glob.glob(os.path.join(artifacts_dir, "tensorflow-*.whl"))
    print("TF Wheel: %s" % tf_wheel_files[0])

    # Now build the TensorFlow C++ library
    cmd = ["bazel", "build", "--config=opt", "//tensorflow:libtensorflow_cc.so"]
    command_executor(cmd)

    # Remove just in case
    try:
        doomed_file = os.path.join(artifacts_dir, "libtensorflow_cc.so")
        os.remove(doomed_file)
        doomed_file = os.path.join(artifacts_dir, "libtensorflow_framework.so")
        os.remove(doomed_file)
    except OSError:
        print("Cannot remove: %s" % doomed_file)
        pass

    # Now copy the TF libraries
    tf_cc_lib_file = "bazel-bin/tensorflow/libtensorflow_cc.so"
    print("Copying %s to %s" % (tf_cc_lib_file, artifacts_dir))
    shutil.copy(tf_cc_lib_file, artifacts_dir)

    tf_cc_fmwk_file = "bazel-bin/tensorflow/libtensorflow_framework.so"
    print("Copying %s to %s" % (tf_cc_fmwk_file, artifacts_dir))
    shutil.copy(tf_cc_fmwk_file, artifacts_dir)

    # popd
    os.chdir(pwd)


def install_tensorflow(venv_dir, artifacts_dir):

    # Load the virtual env
    load_venv(venv_dir)

    # Install tensorflow pip
    tf_pip = os.path.join(os.path.abspath(artifacts_dir), "tensorflow")

    pwd = os.getcwd()
    os.chdir(os.path.join(artifacts_dir, "tensorflow"))

    # Get the name of the TensorFlow pip package
    tf_wheel_files = glob.glob("tensorflow-*.whl")
    if (len(tf_wheel_files) != 1):
        raise Exception(
            "artifacts directory contains more than 1 version of tensorflow wheel"
        )

    command_executor(["pip", "install", "-U", tf_wheel_files[0]])

    cxx_abi = "0"
    if (platform.system() == 'Linux'):
        import tensorflow as tf
        cxx_abi = tf.__cxx11_abi_flag__
        print("LIB: %s" % tf.sysconfig.get_lib())
        print("CXX_ABI: %d" % cxx_abi)

    # popd
    os.chdir(pwd)

    return str(cxx_abi)


def build_ngraph_tf(build_dir, artifacts_location, ngtf_src_loc, venv_dir,
                    cmake_flags, verbose):
    pwd = os.getcwd()

    # Load the virtual env
    load_venv(venv_dir)

    command_executor(["pip", "list"])

    # Get the absolute path for the artifacts
    artifacts_location = os.path.abspath(artifacts_location)

    ngtf_src_loc = os.path.abspath(ngtf_src_loc)
    print("Source location: " + ngtf_src_loc)

    os.chdir(ngtf_src_loc)

    # mkdir build directory
    path = build_dir
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

    # Run cmake
    os.chdir(path)
    cmake_cmd = ["cmake"]
    cmake_cmd.extend(cmake_flags)
    cmake_cmd.extend([ngtf_src_loc])
    command_executor(cmake_cmd)

    import psutil
    num_cores = str(psutil.cpu_count(logical=True))
    make_cmd = ["make", "-j" + num_cores, "install"]
    if verbose:
        make_cmd.extend(['VERBOSE=1'])

    command_executor(make_cmd)

    os.chdir(os.path.join("python", "dist"))
    ngtf_wheel_files = glob.glob("ngraph_tensorflow_bridge-*.whl")
    if (len(ngtf_wheel_files) != 1):
        print("Multiple Python whl files exist. Please remove old wheels")
        for whl in ngtf_wheel_files:
            print("Existing Wheel: " + whl)
        raise Exception("Error getting the ngraph-tf wheel file")

    output_wheel = ngtf_wheel_files[0]
    print("OUTPUT WHL FILE: %s" % output_wheel)

    output_path = os.path.join(artifacts_location, output_wheel)
    print("OUTPUT WHL DST: %s" % output_path)
    # Delete just in case it exists
    try:
        os.remove(output_path)
    except OSError:
        pass

    # Now copy
    shutil.copy2(output_wheel, artifacts_location)

    os.chdir(pwd)
    return output_wheel


def install_ngraph_tf(venv_dir, ngtf_pip_whl):
    # Load the virtual env
    load_venv(venv_dir)

    command_executor(["pip", "install", "-U", ngtf_pip_whl])

    import tensorflow as tf
    print('\033[1;34mVersion information\033[0m')
    print('TensorFlow version: ', tf.__version__)
    print('C Compiler version used in building TensorFlow: ',
          tf.__compiler_version__)
    import ngraph_bridge
    print(ngraph_bridge.__version__)


def download_repo(target_name, repo, version):

    # First download to a temp folder
    call(["git", "clone", repo, target_name])

    # Next goto this folder nd determine the name of the root folder
    pwd = os.getcwd()

    # Go to the tree
    os.chdir(target_name)

    # checkout the specified branch
    call(["git", "fetch"])
    command_executor(["git", "checkout", version])
    os.chdir(pwd)
