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


def build_ngraph(src_location, cmake_flags):
    pwd = os.getcwd()

    src_location = os.path.abspath(src_location)
    print("Source location: " + src_location)

    os.chdir(src_location)

    # mkdir build directory
    path = 'build'
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

    # Run cmake
    os.chdir('build')

    cmake_cmd = ["cmake"]
    cmake_cmd.extend(cmake_flags)
    cmake_cmd.extend([src_location])

    print("nGraph CMAKE flags: %s" % cmake_cmd)
    result = call(cmake_cmd)
    if (result != 0):
        raise Exception("Error running command: " + str(cmake_cmd))

    result = call(["make", "-j", "install"])
    if (result != 0):
        raise Exception("Error running command: make -j install")
    os.chdir(pwd)


def install_virtual_env(venv_dir):
    # Check if we have virtual environment
    # TODO

    # Setup virtual environment
    venv_dir = os.path.abspath(venv_dir)
    # Note: We assume that we are using Python 3 (as this script is also being
    # executed under Python 3 as marked in line 1)
    result = call(["virtualenv", "-p", "python3", venv_dir])
    if (result != 0):
        raise Exception("Error running command: virtualenv -p python3 " + venv_dir)



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

def load_venv2(venv_dir):
    # Check if we are already inside the virtual environment
    return (hasattr(sys, 'real_prefix')
            or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

    venv_dir = os.path.abspath(venv_dir)
    old_os_path = os.environ.get('PATH', '')
    os.environ['PATH'] = os.path.join(venv_dir,
                                      'bin') + os.pathsep + old_os_path
    print("PATH %s" % os.environ['PATH'])
    base = venv_dir
    if sys.platform == 'win32':
        site_packages = os.path.join(base, 'Lib', 'site-packages')
    else:
        site_packages = os.path.join(base, 'lib', 'python%s' % sys.version[:3],
                                     'site-packages')
    prev_sys_path = list(sys.path)

    import site
    site.addsitedir(site_packages)
    sys.real_prefix = sys.prefix
    sys.prefix = base
    # Move the added items to the front of the path:
    new_sys_path = []
    for item in list(sys.path):
        if item not in prev_sys_path:
            new_sys_path.append(item)
            sys.path.remove(item)
    sys.path[:0] = new_sys_path


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
        ],
             stdout=get_pip)
        call(["python3", "./get-pip.py"])

    # Install the pip packages
    if call([
        "pip",
        "install",
        "-U",
        "pip",
        "setuptools",
        "six",
        "numpy",
        "wheel",
        "mock",
        "protobuf",
        "keras_applications==1.0.5",
        "--no-deps",
        "keras_preprocessing==1.0.3",
        "--no-deps",
    ]) != 0:
        raise Exception("Error running command: pip install ")


    # Print the current packages
    call(["pip", "list"])


def build_tensorflow(venv_dir, src_dir, artifacts_dir):

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
    os.environ["CC_OPT_FLAGS"] = "-march=native"

    call([
        "./configure",
    ])

    # Build the python package
    cmd = [
        "bazel",
        "build",
        "--config=opt",
        "//tensorflow/tools/pip_package:build_pip_package",
    ]
    print("Running CMD: " + str(cmd))
    result = call(cmd)
    if (result != 0):
        raise Exception("Error running command: " + str(cmd))

    # Make the pip wheel
    result = call([
        "bazel-bin/tensorflow/tools/pip_package/build_pip_package",
        artifacts_dir
    ])
    if (result != 0):
        raise Exception("Error running command: build_pip_package: " + artifacts_dir)

    # Get the name of the TensorFlow pip package
    tf_wheel_files = glob.glob(os.path.join(artifacts_dir, "tensorflow-*.whl"))
    print("TF Wheel: %s" % tf_wheel_files[0])

    # Now build the TensorFlow C++ library
    cmd = [
        "bazel", "build", "--config=opt", "//tensorflow:libtensorflow_cc.so"
    ]

    print("Running CMD: " + str(cmd))
    result = call(cmd)
    if (result != 0):
        raise Exception("Error running command: " + str(cmd))

    tf_cc_lib_file = "bazel-bin/tensorflow/libtensorflow_cc.so"

    # Remove just in case
    try:
        doomed_file = os.path.join(artifacts_dir, "libtensorflow_cc.so")
        os.remove(doomed_file)
    except OSError:
        print("Cannot remove: %s" % doomed_file)
        pass

    # Now copy
    print("Copying %s to %s" % (tf_cc_lib_file, artifacts_dir))
    shutil.copy2(tf_cc_lib_file, artifacts_dir)

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

    result = call(["pip", "install", "-U", tf_wheel_files[0]])
    if (result != 0):
        raise Exception("Error running command: " + str(cmd))

    cxx_abi = "0"
    if (platform.system() == 'Linux'):
        import tensorflow as tf
        cxx_abi = tf.__cxx11_abi_flag__
        print("LIB: %s" % tf.sysconfig.get_lib())
        print("CXX_ABI: %d" % cxx_abi)

    # popd
    os.chdir(pwd)

    return str(cxx_abi)

def build_ngraph_tf(artifacts_location, ngtf_src_loc, venv_dir, cmake_flags):
    pwd = os.getcwd()

    # Load the virtual env
    load_venv(venv_dir)

    call(["pip", "list"])

    # Get the absolute path for the artifacts
    artifacts_location = os.path.abspath(artifacts_location)

    ngtf_src_loc = os.path.abspath(ngtf_src_loc)
    print("Source location: " + ngtf_src_loc)

    os.chdir(ngtf_src_loc)

    # mkdir build directory
    path = 'build'
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

    # Run cmake
    os.chdir('build')
    cmake_cmd = ["cmake"]
    cmake_cmd.extend(cmake_flags)
    cmake_cmd.extend([ngtf_src_loc])
    print("NGRAPH_TF Cmake options: %s" + str(cmake_cmd))
    if (call(cmake_cmd) != 0):
        raise Exception("Error running cmake command: " + str(cmake_cmd))

    if (call(["make", "-j", "install"]) != 0):
        raise Exception("Error running make command ")

    os.chdir(os.path.join("python", "dist"))
    ngtf_wheel_files = glob.glob("ngraph_tensorflow_bridge-*.whl")
    if (len(ngtf_wheel_files) != 1):
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

    # Intall the ngtf_wheel
    call(["pip", "install", "-U", ngtf_pip_whl])

    import tensorflow as tf
    print('TensorFlow version: r', tf.__version__)
    print(tf.__compiler_version__)
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
    call(["git", "checkout", version])

    os.chdir(pwd)


def main():
    '''
    Builds TensorFlow, ngraph, and ngraph-tf for python 3
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug_build',
        help="Builds a debug version of the nGraph components\n",
        action="store_true")

    parser.add_argument(
        '--skip_tf_build',
        help="Don't build TensorFlow - use existing one\n",
        action="store_true")

    arguments = parser.parse_args()

    if (arguments.debug_build):
        print("Building in DEBUG mode\n")

    #-------------------------------
    # Recipe
    #-------------------------------

    # Create the build directory
    # mkdir build directory
    build_dir = 'build'
    try:
        os.makedirs(build_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(build_dir):
            pass

    pwd = os.getcwd()
    os.chdir(build_dir)

    # Component versions
    ngraph_version = "v0.11.0-rc.1"
    tf_version = "v1.12.0"
    venv_dir = 'venv-tf-py3'
    artifacts_location = 'artifacts'

    #install virtualenv
    install_virtual_env(venv_dir)

    # Load the virtual env
    load_venv(venv_dir)

    # Setup the virtual env
    setup_venv(venv_dir)

    if (not arguments.skip_tf_build):
        # Download TensorFlow
        download_repo("tensorflow",
                      "https://github.com/tensorflow/tensorflow.git",
                      tf_version)

        # Build TensorFlow
        build_tensorflow(venv_dir, "tensorflow", artifacts_location)
    else:
        print("Skipping the TensorFlow build")

    # Install tensorflow
    cxx_abi = install_tensorflow(venv_dir, artifacts_location)

    # Download nGraph
    download_repo("ngraph", "https://github.com/NervanaSystems/ngraph.git",
                  ngraph_version)

    # Now build nGraph
    artifacts_location = os.path.abspath(artifacts_location)
    print("ARTIFACTS location: " + artifacts_location)

    ngraph_cmake_flags = [
        "-DNGRAPH_INSTALL_PREFIX=" + artifacts_location,
        "-DNGRAPH_DISTRIBUTED_ENABLE=FALSE",
        "-DNGRAPH_USE_CXX_ABI=" + cxx_abi,
        "-DNGRAPH_UNIT_TEST_ENABLE=NO",
        "-DNGRAPH_DEX_ONLY=TRUE",
        "-DNGRAPH_GPU_ENABLE=NO",
        "-DNGRAPH_PLAIDML_ENABLE=NO",
        "-DNGRAPH_DEBUG_ENABLE=NO",
        "-DNGRAPH_TARGET_ARCH=native",
        "-DNGRAPH_TUNE_ARCH=native",
    ]
    if (platform.system() != 'Darwin'):
        ngraph_cmake_flags.extend(["-DNGRAPH_TOOLS_ENABLE=YES"])
    else:
        ngraph_cmake_flags.extend(["-DNGRAPH_TOOLS_ENABLE=NO"])

    if (arguments.debug_build):
        ngraph_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    build_ngraph("./ngraph", ngraph_cmake_flags)

    # Next build CMAKE options for the bridge
    tf_src_dir = os.path.abspath("tensorflow")

    ngraph_tf_cmake_flags = [
        "-DUSE_PRE_BUILT_NGRAPH=ON", "-DNGRAPH_TARGET_ARCH=native",
        "-DNGRAPH_TUNE_ARCH=native",
        "-DNGRAPH_ARTIFACTS_DIR=" + artifacts_location,
        "-DUNIT_TEST_ENABLE=ON",
        "-DTF_SRC_DIR=" + tf_src_dir, "-DUNIT_TEST_TF_CC_DIR=" + os.path.join(
            artifacts_location, "tensorflow")
    ]
    if (arguments.debug_build):
        ngraph_tf_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    # Now build the bridge
    ng_tf_whl = build_ngraph_tf(artifacts_location, "../", venv_dir,
                                ngraph_tf_cmake_flags)

    print("SUCCESSFULLY generated wheel: %s" % ng_tf_whl)

    # Run a quick test
    install_ngraph_tf(venv_dir, os.path.join(artifacts_location, ng_tf_whl))

    os.chdir(pwd)


if __name__ == '__main__':
    main()
