#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2019-2020 Intel Corporation
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
import subprocess
from subprocess import check_output, call
import sys
import shutil
import glob
import platform
import shlex
from sysconfig import get_paths


def get_tf_cxxabi():
    import tensorflow as tf
    print('Version information:')
    print('TensorFlow version: ', tf.__version__)
    print('C Compiler version used in building TensorFlow: ',
          tf.__compiler_version__)
    return str(tf.__cxx11_abi_flag__)


def is_venv():
    # https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


def command_executor(cmd,
                     verbose=False,
                     msg=None,
                     stdout=sys.stdout,
                     stderr=sys.stderr):
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
        print(tag + cmd, file=stdout)
    try:
        process = subprocess.Popen(
            shlex.split(cmd), stdout=stdout, stderr=stderr)
        so, se = process.communicate()
        retcode = process.returncode
        assert retcode == 0, "dir:" + os.getcwd(
        ) + ". Error in running command: " + cmd
    except OSError as e:
        print(
            "!!! Execution failed !!!",
            e,
            " -->  Error running command:",
            cmd,
            file=sys.stderr)
        raise
    except Exception as e:
        raise


def cmake_build(build_dir, src_location, cmake_flags, verbose):
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
        dict(__file__=activate_this_file))

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
        "psutil",
        "six>=1.12.0",
        "numpy>=1.16.0,<1.19.0",
        "wheel>=0.26",
        "setuptools",
        "mock",
        "termcolor>=1.1.0",
        "keras_applications>=1.0.6",
        "--no-deps",
        "keras_preprocessing>=1.1.1,<1.2",
        "--no-deps",
        "yapf==0.26.0",
    ]
    command_executor(package_list)

    # Print the current packages
    command_executor(["pip", "list"])


def build_tensorflow(tf_version,
                     src_dir,
                     artifacts_dir,
                     target_arch,
                     verbosity,
                     use_intel_tf,
                     target=""):
    # In order to build TensorFlow, we need to be in the virtual environment
    pwd = os.getcwd()

    src_dir = os.path.abspath(src_dir)
    print("SOURCE DIR: " + src_dir)

    # Update the artifacts directory
    artifacts_dir = os.path.join(os.path.abspath(artifacts_dir), "tensorflow")
    print("ARTIFACTS DIR: %s" % artifacts_dir)

    os.chdir(src_dir)

    base = sys.prefix
    python_lib_path = os.path.join(base, 'lib', 'python%s' % sys.version[:3],
                                   'site-packages')
    python_executable = os.path.join(base, "bin", "python")

    print("PYTHON_BIN_PATH: " + python_executable)

    # Set the TensorFlow configuration related variables
    os.environ["PYTHON_BIN_PATH"] = python_executable
    os.environ["PYTHON_LIB_PATH"] = python_lib_path
    os.environ["TF_ENABLE_XLA"] = "0"
    if (platform.system() == 'Darwin'):
        os.environ["TF_CONFIGURE_IOS"] = "0"
    os.environ["TF_NEED_OPENCL_SYCL"] = "0"
    os.environ["TF_NEED_COMPUTECPP"] = "0"
    os.environ["TF_NEED_ROCM"] = "0"
    os.environ["TF_NEED_MPI"] = "0"
    os.environ["TF_NEED_CUDA"] = "0"
    os.environ["TF_NEED_TENSORRT"] = "0"
    os.environ["TF_DOWNLOAD_CLANG"] = "0"
    os.environ["TF_SET_ANDROID_WORKSPACE"] = "0"
    os.environ["CC_OPT_FLAGS"] = "-march=" + target_arch + " -Wno-sign-compare"

    command_executor("./configure")

    cmd = [
        "bazel",
        "build",
        "--config=opt",
        "--config=noaws",
        "--config=nohdfs",
        "--config=nonccl",
    ]
    if use_intel_tf:
        print("Building Intel-Tensorflow")
        cmd.extend([
            "--config=mkl",
        ])
    # Build the python package
    if (tf_version.startswith("v2.") or tf_version.startswith("2.")):
        cmd.extend([
            "--config=v2",
        ])
    elif (tf_version.startswith("v1.") or tf_version.startswith("1.")):
        cmd.extend([
            "--config=v1",
        ])

    # If target is not specified, we assume default TF wheel build
    if target == '':
        target = "//tensorflow/tools/pip_package:build_pip_package"

    cmd.extend([target])

    if verbosity:
        cmd.extend(['-s'])

    command_executor(cmd)

    # If target is not specified, we assume default TF wheel build and copy the wheel to artifacts dir
    if target == '//tensorflow/tools/pip_package:build_pip_package':
        command_executor([
            "bazel-bin/tensorflow/tools/pip_package/build_pip_package",
            artifacts_dir
        ])

        # Get the name of the TensorFlow pip package
        tf_wheel_files = glob.glob(
            os.path.join(artifacts_dir, "tensorflow-*.whl"))
        print("TF Wheel: %s" % tf_wheel_files[0])

    # popd
    os.chdir(pwd)


def build_tensorflow_cc(tf_version,
                        src_dir,
                        artifacts_dir,
                        target_arch,
                        verbosity,
                        use_intel_tf,
                        tf_prebuilt=None):
    lib = "libtensorflow_cc.so.2"
    if (tf_version.startswith("v2.") or tf_version.startswith("2.")):
        tf_cc_lib_name = "libtensorflow_cc.so.2"
    elif (tf_version.startswith("v1.") or tf_version.startswith("1.")):
        tf_cc_lib_name = "libtensorflow_cc.so.1"

    build_tensorflow(
        tf_version,
        src_dir,
        artifacts_dir,
        target_arch,
        verbosity,
        use_intel_tf,
        target="//tensorflow:" + tf_cc_lib_name +
        " //tensorflow/core/kernels:ops_testutil")

    # In order to build TensorFlow, we need to be in the virtual environment
    pwd = os.getcwd()

    src_dir = os.path.abspath(src_dir)
    print("SOURCE DIR: " + src_dir)

    # Update the artifacts directory
    artifacts_dir = os.path.join(os.path.abspath(artifacts_dir), "tensorflow")
    print("ARTIFACTS DIR: %s" % artifacts_dir)

    os.chdir(src_dir)
    try:
        doomed_file = os.path.join(artifacts_dir, tf_cc_lib_name)
        os.unlink(doomed_file)
    except OSError:
        print("Cannot remove: %s" % doomed_file)
        pass

    # Now copy the TF libraries
    if tf_prebuilt is None:
        tf_cc_lib_file = "bazel-bin/tensorflow/" + tf_cc_lib_name
    else:
        tf_cc_lib_file = os.path.abspath(tf_prebuilt + '/' + tf_cc_lib_name)

    print("Copying %s to %s" % (tf_cc_lib_file, artifacts_dir))
    shutil.copy(tf_cc_lib_file, artifacts_dir)
    os.chdir(pwd)


def locate_tf_whl(tf_whl_loc):
    possible_whl = [i for i in os.listdir(tf_whl_loc) if '.whl' in i]
    assert len(possible_whl
              ) == 1, "Expected 1 TF whl file, but found " + len(possible_whl)
    tf_whl = os.path.abspath(tf_whl_loc + '/' + possible_whl[0])
    assert os.path.isfile(tf_whl), "Did not find " + tf_whl
    return tf_whl


def copy_tf_to_artifacts(tf_version, artifacts_dir, tf_prebuilt, use_intel_tf):
    if (tf_version.startswith("v2.")):
        tf_fmwk_lib_name = 'libtensorflow_framework.so.2'
        tf_cc_lib_name = 'libtensorflow_cc.so.2'
    elif (tf_version.startswith("v1.")):
        tf_fmwk_lib_name = 'libtensorflow_framework.so.1'
        tf_cc_lib_name = 'libtensorflow_cc.so.1'
    if (platform.system() == 'Darwin'):
        if (tf_version.startswith("v2.")):
            tf_fmwk_lib_name = 'libtensorflow_framework.2.dylib'
        elif (tf_version.startswith("v1.")):
            tf_fmwk_lib_name = 'libtensorflow_framework.1.dylib'
    try:
        doomed_file = os.path.join(artifacts_dir, tf_cc_lib_name)
        os.unlink(doomed_file)
        doomed_file = os.path.join(artifacts_dir, tf_fmwk_lib_name)
        os.unlink(doomed_file)
    except OSError:
        print("Cannot remove: %s" % doomed_file)
        pass

    # Now copy the TF libraries
    if tf_prebuilt is None:
        tf_cc_lib_file = "bazel-bin/tensorflow/" + tf_cc_lib_name
        tf_cc_fmwk_file = "bazel-bin/tensorflow/" + tf_fmwk_lib_name
        if use_intel_tf:
            opm_lib_file = "bazel-tensorflow/external/mkl_linux/lib/libiomp5.so"
            mkl_lib_file = "bazel-tensorflow/external/mkl_linux/lib/libmklml_intel.so"
    else:
        tf_cc_lib_file = os.path.abspath(tf_prebuilt + '/' + tf_cc_lib_name)
        tf_cc_fmwk_file = os.path.abspath(tf_prebuilt + '/' + tf_fmwk_lib_name)
        if use_intel_tf:
            opm_lib_file = os.path.abspath(tf_prebuilt + '/libiomp5.so')
            mkl_lib_file = os.path.abspath(tf_prebuilt + '/libmklml_intel.so')
    print("PWD: ", os.getcwd())
    print("Copying %s to %s" % (tf_cc_lib_file, artifacts_dir))
    shutil.copy(tf_cc_lib_file, artifacts_dir)

    print("Copying %s to %s" % (tf_cc_fmwk_file, artifacts_dir))
    shutil.copy(tf_cc_fmwk_file, artifacts_dir)
    if use_intel_tf:
        print("Copying %s to %s" % (opm_lib_file, artifacts_dir))
        shutil.copy(opm_lib_file, artifacts_dir)

        print("Copying %s to %s" % (mkl_lib_file, artifacts_dir))
        shutil.copy(mkl_lib_file, artifacts_dir)

    if tf_prebuilt is not None:
        tf_whl = locate_tf_whl(tf_prebuilt)
        shutil.copy(tf_whl, artifacts_dir)


def install_tensorflow(venv_dir, artifacts_dir):

    # Load the virtual env
    load_venv(venv_dir)

    # Install tensorflow pip
    tf_pip = os.path.join(os.path.abspath(artifacts_dir), "tensorflow")

    pwd = os.getcwd()
    os.chdir(os.path.join(artifacts_dir, "tensorflow"))

    # Get the name of the TensorFlow pip package
    tf_wheel_files = glob.glob("tensorflow-*.whl")
    if (len(tf_wheel_files) < 1):
        raise Exception("no tensorflow wheels found")
    elif (len(tf_wheel_files) > 1):
        raise Exception("more than 1 version of tensorflow wheels found")
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


def install_ngraph_tf(tf_version, venv_dir, ngtf_pip_whl):
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


def download_repo(target_name, repo, version, submodule_update=False):
    # First download to a temp folder
    call(["git", "clone", repo, target_name])

    pwd = os.getcwd()
    os.chdir(target_name)

    # checkout the specified branch and get the latest changes
    call(["git", "fetch"])
    command_executor(["git", "checkout", version])
    call(["git", "pull"])

    if submodule_update:
        call(["git", "submodule", "update", "--init", "--recursive"])

    os.chdir(pwd)


def apply_patch(patch_file, level=1):
    # IF patching TensorFlow unittests is done through an automation system,
    # please ensure the latest `libdvdnav-dev` or `libdvdnav-devel` is installed.
    cmd = subprocess.Popen(
        'patch -p' + str(level) + ' -N -i ' + patch_file,
        shell=True,
        stdout=subprocess.PIPE)
    printed_lines = cmd.communicate()
    # Check if the patch is being applied for the first time, in which case
    # cmd.returncode will be 0 or if the patch has already been applied, in
    # which case the string will be found, in all other cases the assertion
    # will fail
    assert cmd.returncode == 0 or 'patch detected!  Skipping patch' in str(
        printed_lines[0]), "Error applying the patch."


def get_gcc_version():
    cmd = subprocess.Popen(
        'gcc -dumpfullversion -dumpversion',
        shell=True,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True)
    output = cmd.communicate()[0].rstrip()
    return output


def get_cmake_version():
    cmd = subprocess.Popen(
        'cmake --version',
        shell=True,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True)
    output = cmd.communicate()[0].rstrip()
    # The cmake version format is: "cmake version a.b.c"
    version_tuple = output.split()[2].split('.')
    return version_tuple


def get_bazel_version():
    cmd = subprocess.Popen(
        'bazel version',
        shell=True,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True)
    # The bazel version format is a multi line output:
    #
    # Build label: 0.25.2
    # Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
    # Build time: Fri May 10 20:47:48 2019 (1557521268)
    # Build timestamp: 1557521268
    # Build timestamp as int: 1557521268
    #
    output = cmd.communicate()[0].splitlines()[0].strip()
    version_info = output.split(':')
    bazel_kind = version_info[0].strip()
    version = version_info[1].strip()
    version_tuple = version.split('.')
    return bazel_kind, version_tuple


def build_openvino(build_dir, openvino_src_dir, cxx_abi, target_arch,
                   artifacts_location, debug_enabled, verbosity):
    install_location = os.path.join(artifacts_location, "openvino")
    print("INSTALL location: " + artifacts_location)

    # Now build OpenVINO
    openvino_cmake_flags = [
        "-DENABLE_V10_SERIALIZE=ON", "-DENABLE_TESTS=OFF",
        "-DENABLE_SAMPLES=OFF", "-DENABLE_FUNCTIONAL_TESTS=OFF",
        "-DENABLE_VPU=ON", "-DENABLE_GNA=OFF",
        "-DNGRAPH_ONNX_IMPORT_ENABLE=OFF", "-DNGRAPH_TEST_UTIL_ENABLE=OFF",
        "-DNGRAPH_USE_CXX_ABI=" + cxx_abi,
        "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=" + cxx_abi + " -march=" +
        target_arch, "-DENABLE_CPPLINT=OFF", "-DENABLE_SPEECH_DEMO=FALSE",
        "-DCMAKE_INSTALL_RPATH=\"$ORIGIN\"",
        "-DCMAKE_INSTALL_PREFIX=" + install_location
    ]

    if debug_enabled:
        openvino_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    cmake_build(build_dir, openvino_src_dir, openvino_cmake_flags, verbosity)
