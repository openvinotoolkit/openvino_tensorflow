#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import argparse
from argparse import RawTextHelpFormatter

import errno
import os
import subprocess
import sys
import shutil
import glob
import platform
import shlex
import math
import psutil as psu
from sysconfig import get_paths
from subprocess import call
from wheel.vendored.packaging.tags import sys_tags


def get_tf_version():
    import tensorflow as tf
    return tf.__version__


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
                     stderr=sys.stderr,
                     shell=False):
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
            shlex.split(cmd), stdout=stdout, stderr=stderr, shell=shell)
        so, se = process.communicate()
        retcode = process.returncode
        if not retcode == 0:
            raise AssertionError("dir:" + os.getcwd() +
                                 ". Error in running command: " + cmd)
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
    if not os.path.exists(src_location):
        raise AssertionError("Path doesn't exist {0}".format(src_location))
    os.chdir(src_location)

    # mkdir build directory
    path = build_dir
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

    # Run cmake
    if not os.path.exists(build_dir):
        raise AssertionError("Path doesn't exist {0}".format(build_dir))
    os.chdir(build_dir)

    cmake_cmd = ["cmake"]
    cmake_cmd.extend(cmake_flags)
    if (platform.system() == 'Windows'):
        cmake_cmd.extend([src_location.replace("\\", "\\\\")])
    else:
        cmake_cmd.extend([src_location])

    command_executor(cmake_cmd, verbose=True)

    import psutil
    num_cores = int(psutil.cpu_count(logical=True))
    # get system's total RAM size in GB
    sys_ram = int(psutil.virtual_memory().total / (1024**3))
    # limiting num of cores to the max GBs of system RAM
    if (num_cores > sys_ram):
        num_cores = sys_ram
    num_cores = str(num_cores)

    if (platform.system() == 'Windows'):
        # TODO: Enable Debug config for windows
        cmd = [
            "cmake", "--build", ".", "--config Release", "-j" + num_cores,
            "--target install"
        ]
        if verbose:
            cmd.extend(['--verbose'])
        command_executor(cmd, verbose=True)
    else:
        cmd = ["make", "-j" + num_cores]
        if verbose:
            cmd.extend(['VERBOSE=1'])
        command_executor(cmd, verbose=True)
        cmd = ["make", "install"]
        command_executor(cmd, verbose=True)
    if not os.path.exists(pwd):
        raise AssertionError("Path doesn't exist {0}".format(pwd))
    os.chdir(pwd)


def install_virtual_env(venv_dir):
    # Check if we have virtual environment
    # TODO

    # Setup virtual environment
    venv_dir = os.path.abspath(venv_dir)
    if (platform.system() == 'Windows'):
        venv_dir = venv_dir.replace("\\", "\\\\")
    # Note: We assume that we are using Python 3 (as this script is also being
    # executed under Python 3 as marked in line 1)
    if (platform.system() == 'Windows'):
        command_executor(["python", "-m", "venv", venv_dir])
    else:
        command_executor(["python3", "-m", "venv", venv_dir])


def load_venv(venv_dir):
    venv_dir = os.path.abspath(venv_dir)

    # Check if we are already inside the virtual environment
    # return (hasattr(sys, 'real_prefix')
    #         or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    print("Loading virtual environment from: %s" % venv_dir)

    # Since activate_this.py is no longer available in python3's default venv support,
    # we bring its functionality into this load_env module
    # activate_this_file = venv_dir + "/bin/activate_this.py"
    # The execfile API is for Python 2. We keep here just in case you are on an
    # obscure system without Python 3
    # execfile(activate_this_file, dict(__file__=activate_this_file))
    # exec(
    #     compile(
    #         open(activate_this_file, "rb").read(), activate_this_file, 'exec'),
    #     dict(__file__=activate_this_file))
    # exec(open(activate_this_file).read(), {'__file__': activate_this_file})

    if (platform.system() == "Windows"):
        bin_dir = os.path.join(venv_dir, "Scripts")
        base = bin_dir[:-len(
            "Scripts"
        ) - 1]  # strip away the bin part from the __file__, plus the path separator
    else:
        bin_dir = os.path.join(venv_dir, "bin")
        base = bin_dir[:-len(
            "bin"
        ) - 1]  # strip away the bin part from the __file__, plus the path separator

    # prepend bin to PATH (this file is inside the bin directory)
    os.environ["PATH"] = os.pathsep.join(
        [bin_dir] + os.environ.get("PATH", "").split(os.pathsep))
    os.environ["VIRTUAL_ENV"] = base  # virtual env is right above bin directory

    import site

    # add the virtual environments libraries to the host python import mechanism
    prev_length = len(sys.path)
    if (platform.system() == 'Windows'):
        for lib in ("../Lib/site-packages").split(os.pathsep):
            path = os.path.realpath(os.path.join(bin_dir, lib))
            site.addsitedir(path.decode("utf-8") if "" else path)
    else:
        for lib in ("../lib/python3." + str(sys.version_info.minor) +
                    "/site-packages").split(os.pathsep):
            path = os.path.realpath(os.path.join(bin_dir, lib))
            site.addsitedir(path.decode("utf-8") if "" else path)
    sys.path[:] = sys.path[prev_length:] + sys.path[0:prev_length]

    sys.real_prefix = sys.prefix
    sys.prefix = base

    # excluding system-wide site-packages paths as they interfere when
    # venv tensorflow version is different from system-wide tensorflow version
    sys_paths = [_p for _p in sys.path if "site-packages" in _p]
    sys_paths = [_p for _p in sys_paths if "venv-tf-py3" not in _p]
    for _p in sys_paths:
        sys.path.remove(_p)

    # ignore site-package installations in user-space
    import site
    site.ENABLE_USER_SITE = False

    return venv_dir


def setup_venv(venv_dir, tf_version):
    load_venv(venv_dir)

    print("PIP location")
    if (platform.system() == 'Windows'):
        call(['where', 'pip'])
    else:
        process = subprocess.Popen(shlex.split('which pip'))
        so, se = process.communicate()

    # Install the pip packages
    if (platform.system() == "Windows"):
        command_executor(["python -m pip", "install", "-U", "pip"])
    else:
        command_executor(["pip", "install", "-U", "pip"])
    package_list = [
        "pip3",
        "install",
        "psutil",
        "six>=1.12.0",
        "numpy>=1.19.5",
        "wheel>=0.26",
        "setuptools",
        "mock",
        "termcolor>=1.1.0",
        "keras_applications>=1.0.6",
        "--no-deps",
        "keras",
        "keras_preprocessing>=1.1.1,<1.2",
        "--no-deps",
        "yapf==0.26.0",
        "opencv-python==4.5.2.54",
    ]
    command_executor(package_list)
    # TF on windows needs a higher version of numpy
    if (platform.system == "Windows"):
        command_executor(["pip", "install", "numpy>=1.21.2"])
    else:
        # TF >=2.4 and <= 2.6 requires numpy~=1.19.2
        tf_maj_version = tf_version.split(".")[0]
        tf_min_version = int(tf_version.split(".")[1])
        if ((tf_maj_version == "v2") and (3 < tf_min_version < 7)):
            command_executor(["pip", "install", "h5py"])
            command_executor(["pip", "install", "numpy~=1.19.2"])

    # Print the current packages
    command_executor(["pip", "list"])


def get_tf_build_resources(resource_usage_ratio=0.5):
    num_cores = int(psu.cpu_count(logical=True) * resource_usage_ratio)
    jobs = int(psu.cpu_count(logical=True) * resource_usage_ratio)
    #Bazel takes this flag in MB. If not given default TOTAL_RAM*0.67; 1GB -> (1<<30); 1MB -> (1<<20)
    ram_usage = math.floor((psu.virtual_memory().total /
                            (1 << 30)) * resource_usage_ratio) * (1 << 20)
    return num_cores, jobs, ram_usage


def build_tensorflow(tf_version,
                     src_dir,
                     artifacts_dir,
                     target_arch,
                     verbosity,
                     use_intel_tf,
                     cxx_abi,
                     target="",
                     resource_usage_ratio=0.5):
    # In order to build TensorFlow, we need to be in the virtual environment
    pwd = os.getcwd()

    src_dir = os.path.abspath(src_dir)
    print("SOURCE DIR: " + src_dir)

    # Update the artifacts directory
    if (platform.system() == 'Windows'):
        artifacts_dir = os.path.join(pwd, "tensorflow")
    else:
        artifacts_dir = os.path.join(
            os.path.abspath(artifacts_dir), "tensorflow")
    print("ARTIFACTS DIR: %s" % artifacts_dir)
    if not os.path.exists(src_dir):
        raise AssertionError("Path doesn't exist {0}".format(src_dir))
    os.chdir(src_dir)

    base = sys.prefix
    if (platform.system() == 'Windows'):
        python_lib_path = os.path.join(base, 'Lib', 'site-packages')
        python_executable = os.path.join(base, "Scripts", "python.exe")
    else:
        python_lib_path = os.path.join(
            base, 'lib', 'python%s' % sys.version[:3], 'site-packages')
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
    os.environ["CC_OPT_FLAGS"] = " -Wno-sign-compare"
    if (target_arch == "silvermont"):
        os.environ[
            "CC_OPT_FLAGS"] = " -mcx16 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mno-avx"

    if (platform.system() == 'Windows'):
        os.environ["TF_OVERRIDE_EIGEN_STRONG_INLINE"] = "1"
        os.environ["CC_OPT_FLAGS"] = "/arch:AVX"
        command_executor(["python", "configure.py"])
    else:
        command_executor("./configure")

    cmd = ["bazel", "build", "--config=opt"]

    if (platform.system() != 'Windows'):
        cmd.extend([
            "--config=noaws",
            "--config=nohdfs",
            "--config=nonccl",
        ])

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

    # Build Tensorflow with user-specified ABI
    # Consequent builds of Openvino-Tensorflow (OVTF) and OpenVINO will use the same ABI
    cmd.extend(["--cxxopt=\"-D_GLIBCXX_USE_CXX11_ABI=%s\"" % cxx_abi])

    # If target is not specified, we assume default TF wheel build
    if target == '':
        target = "//tensorflow/tools/pip_package:build_pip_package"

    cmd.extend([target])

    if verbosity:
        cmd.extend(['-s'])

    num_cores, jobs, ram_usage = get_tf_build_resources(resource_usage_ratio)
    cmd.extend(["--local_cpu_resources=%d" % num_cores])
    cmd.extend(["--local_ram_resources=%d" % ram_usage])
    cmd.extend(["--jobs=%d" % jobs])

    command_executor(cmd, verbose=True)

    # If target is not specified, we assume default TF wheel build and copy the wheel to artifacts dir
    if target == '//tensorflow/tools/pip_package:build_pip_package':
        if (platform.system() == 'Windows'):
            command_executor([
                "bazel-bin\\\\tensorflow\\\\tools\\\\pip_package\\\\build_pip_package",
                artifacts_dir.replace("\\", "\\\\")
            ])
        else:
            command_executor([
                "bazel-bin/tensorflow/tools/pip_package/build_pip_package",
                artifacts_dir
            ])

        # Get the name of the TensorFlow pip package
        tf_wheel_files = glob.glob(
            os.path.join(artifacts_dir, "tensorflow-*.whl"))
        print("TF Wheel: %s" % tf_wheel_files[0])

    # popd
    if not os.path.exists(pwd):
        raise AssertionError("Path doesn't exist {0}".format(pwd))
    os.chdir(pwd)


def build_tensorflow_cc(tf_version,
                        src_dir,
                        artifacts_dir,
                        target_arch,
                        verbosity,
                        use_intel_tf,
                        cxx_abi,
                        tf_prebuilt=None):
    # lib = "libtensorflow_cc.so.2"
    if (tf_version.startswith("v2.") or tf_version.startswith("2.")):
        tf_cc_lib_name = "libtensorflow_cc.so.2"
    elif (tf_version.startswith("v1.") or tf_version.startswith("1.")):
        tf_cc_lib_name = "libtensorflow_cc.so.1"

    if (platform.system() == 'Windows'):
        tf_cc_lib_name = "tensorflow_cc.dll.if.lib"

    build_tensorflow(
        tf_version,
        src_dir,
        artifacts_dir,
        target_arch,
        verbosity,
        use_intel_tf,
        cxx_abi,
        target="//tensorflow:" + tf_cc_lib_name +
        " //tensorflow/core/kernels:ops_testutil")

    # In order to build TensorFlow, we need to be in the virtual environment
    pwd = os.getcwd()

    src_dir = os.path.abspath(src_dir)
    print("SOURCE DIR: " + src_dir)

    # Update the artifacts directory
    artifacts_dir = os.path.join(os.path.abspath(artifacts_dir), "tensorflow")
    print("ARTIFACTS DIR: %s" % artifacts_dir)
    if not os.path.exists(src_dir):
        raise AssertionError("Path doesn't exist {0}".format(src_dir))
    os.chdir(src_dir)
    try:
        doomed_file = os.path.join(artifacts_dir, tf_cc_lib_name)
        try:
            if not os.path.exists(doomed_file):
                raise AssertionError(
                    "File not present for unlinking {0}".format(doomed_file))
        except Exception as e:
            print("Cannot remove: %s" % e)
            pass
        os.unlink(doomed_file)
    except OSError:
        print("Cannot remove: %s" % doomed_file)
        pass

    # Now copy the TF libraries
    if tf_prebuilt is None:
        if (platform.system() == 'Windows'):
            tf_cc_lib_file = "bazel-bin\\tensorflow\\" + tf_cc_lib_name
        else:
            tf_cc_lib_file = "bazel-bin/tensorflow/" + tf_cc_lib_name
    else:
        if (platform.system() == 'Windows'):
            tf_cc_lib_file = os.path.abspath(tf_prebuilt + '\\' +
                                             tf_cc_lib_name)
        else:
            tf_cc_lib_file = os.path.abspath(tf_prebuilt + '/' + tf_cc_lib_name)

    print("Copying %s to %s" % (tf_cc_lib_file, artifacts_dir))
    shutil.copy(tf_cc_lib_file, artifacts_dir)
    if not os.path.exists(pwd):
        raise AssertionError("Path doesn't exist {0}".format(pwd))
    os.chdir(pwd)


def locate_tf_whl(tf_whl_loc):
    if not os.path.exists(tf_whl_loc):
        raise AssertionError("path doesn't exist {0}".format(tf_whl_loc))
    possible_whl = [i for i in os.listdir(tf_whl_loc) if '.whl' in i]
    if not len(possible_whl) == 1:
        raise AssertionError("Expected 1 TF whl file, but found " +
                             len(possible_whl))
    tf_whl = os.path.abspath(tf_whl_loc + '/' + possible_whl[0])
    if not os.path.isfile(tf_whl):
        raise AssertionError("Did not find " + tf_whl)
    return tf_whl


def copy_tf_to_artifacts(tf_version, artifacts_dir, tf_prebuilt, use_intel_tf):
    if (tf_version.startswith("v2.") or (tf_version.startswith("2."))):
        tf_fmwk_lib_name = 'libtensorflow_framework.so.2'
        tf_cc_lib_name = 'libtensorflow_cc.so.2'
    elif (tf_version.startswith("v1.") or (tf_version.startswith("1."))):
        tf_fmwk_lib_name = 'libtensorflow_framework.so.1'
        tf_cc_lib_name = 'libtensorflow_cc.so.1'
    if (platform.system() == 'Darwin'):
        if (tf_version.startswith("v2.") or (tf_version.startswith("2."))):
            tf_fmwk_lib_name = 'libtensorflow_framework.2.dylib'
        elif (tf_version.startswith("v1.") or (tf_version.startswith("1."))):
            tf_fmwk_lib_name = 'libtensorflow_framework.1.dylib'
    if (platform.system() == 'Windows'):
        tf_cc_dll_name = 'tensorflow_cc.dll'
        tf_cc_lib_name = 'tensorflow_cc.dll.if.lib'
        tf_fmwk_dll_name = '_pywrap_tensorflow_internal.pyd'
        tf_fmwk_lib_name = '_pywrap_tensorflow_internal.lib'

    try:
        doomed_file = os.path.join(artifacts_dir, tf_cc_lib_name)
        # assert os.path.exists(doomed_file), "File not present for unlinking {0}".format(doomed_file)
        os.unlink(doomed_file)
        doomed_file = os.path.join(artifacts_dir, tf_fmwk_lib_name)
        try:
            if not os.path.exists(doomed_file):
                raise AssertionError(
                    "File not present for unlinking {0}".format(doomed_file))
        except Exception as e:
            print("Cannot remove: %s" % e)
            pass
        os.unlink(doomed_file)
    except OSError:
        print("Cannot remove: %s" % doomed_file)
        pass

    # Now copy the TF libraries
    if tf_prebuilt is None:
        if (platform.system() == 'Windows'):
            tf_cc_lib_file = "bazel-bin\\tensorflow\\" + tf_cc_lib_name
            tf_cc_dll_file = "bazel-bin\\tensorflow\\" + tf_cc_dll_name
            tf_fmwk_lib_file = "bazel-bin\\tensorflow\\python\\" + tf_fmwk_lib_name
            tf_fmwk_dll_file = "bazel-bin\\tensorflow\\python\\" + tf_fmwk_dll_name
        else:
            tf_cc_lib_file = "bazel-bin/tensorflow/" + tf_cc_lib_name
            tf_cc_fmwk_file = "bazel-bin/tensorflow/" + tf_fmwk_lib_name
        if use_intel_tf:
            opm_lib_file = "bazel-tensorflow/external/mkl_linux/lib/libiomp5.so"
            mkl_lib_file = "bazel-tensorflow/external/mkl_linux/lib/libmklml_intel.so"
    else:
        if (platform.system() == 'Windows'):
            tf_cc_lib_file = os.path.abspath(
                tf_prebuilt + '\\bazel-bin\\tensorflow\\' + tf_cc_lib_name)
            tf_cc_dll_file = os.path.abspath(
                tf_prebuilt + '\\bazel-bin\\tensorflow\\' + tf_cc_dll_name)
            tf_fmwk_lib_file = os.path.abspath(
                tf_prebuilt + '\\bazel-bin\\tensorflow\\python\\' +
                tf_fmwk_lib_name)
            tf_fmwk_dll_file = os.path.abspath(
                tf_prebuilt + '\\bazel-bin\\tensorflow\\python\\' +
                tf_fmwk_dll_name)
        else:
            tf_cc_lib_file = os.path.abspath(tf_prebuilt + '/' + tf_cc_lib_name)
            tf_cc_fmwk_file = os.path.abspath(tf_prebuilt + '/' +
                                              tf_fmwk_lib_name)
        if use_intel_tf:
            opm_lib_file = os.path.abspath(tf_prebuilt + '/libiomp5.so')
            mkl_lib_file = os.path.abspath(tf_prebuilt + '/libmklml_intel.so')

    print("PWD: ", os.getcwd())
    print("Copying %s to %s" % (tf_cc_lib_file, artifacts_dir))
    shutil.copy(tf_cc_lib_file, artifacts_dir)
    if (platform.system() == "Windows"):
        print("Copying %s to %s" % (tf_cc_dll_file, artifacts_dir))
        shutil.copy(tf_cc_dll_file, artifacts_dir)
        print("Copying %s to %s" % (tf_fmwk_lib_file, artifacts_dir))
        shutil.copy(tf_fmwk_lib_file, artifacts_dir)
        print("Copying %s to %s" % (tf_fmwk_dll_file, artifacts_dir))
        shutil.copy(tf_fmwk_dll_file, artifacts_dir)
    else:
        print("Copying %s to %s" % (tf_cc_fmwk_file, artifacts_dir))
        shutil.copy(tf_cc_fmwk_file, artifacts_dir)

    if use_intel_tf:
        print("Copying %s to %s" % (opm_lib_file, artifacts_dir))
        shutil.copy(opm_lib_file, artifacts_dir)

        print("Copying %s to %s" % (mkl_lib_file, artifacts_dir))
        shutil.copy(mkl_lib_file, artifacts_dir)

    if (platform.system() != 'Windows'):
        if tf_prebuilt is not None:
            tf_whl = locate_tf_whl(tf_prebuilt)
            shutil.copy(tf_whl, artifacts_dir)


def install_tensorflow(venv_dir, artifacts_dir):

    # Load the virtual env
    load_venv(venv_dir)

    # Install tensorflow pip
    tf_pip = os.path.join(os.path.abspath(artifacts_dir), "tensorflow")

    pwd = os.getcwd()
    if not os.path.exists(os.path.join(artifacts_dir, "tensorflow")):
        raise AssertionError("Path doesn't exist {0}".format(
            os.path.join(artifacts_dir, "tensorflow")))
    os.chdir(os.path.join(artifacts_dir, "tensorflow"))

    # Get the name of the TensorFlow pip package
    tf_wheel_files = glob.glob("tensorflow-*.whl")
    if (len(tf_wheel_files) < 1):
        raise Exception("no tensorflow wheels found")
    elif (len(tf_wheel_files) > 1):
        raise Exception("more than 1 version of tensorflow wheels found")
    command_executor(
        ["pip", "install", "--force-reinstall", "-U", tf_wheel_files[0]])

    import tensorflow as tf
    cxx_abi = tf.__cxx11_abi_flag__
    print("LIB: %s" % tf.sysconfig.get_lib())
    print("CXX_ABI: %d" % cxx_abi)

    # popd
    if not os.path.exists(pwd):
        raise AssertionError("Path doesn't exist {0}".format(pwd))
    os.chdir(pwd)

    return str(cxx_abi)


def build_openvino_tf(build_dir, artifacts_location, ovtf_src_loc, venv_dir,
                      cmake_flags, verbose):
    pwd = os.getcwd()

    # Load the virtual env
    load_venv(venv_dir)

    command_executor(["pip", "list"])

    # Get the absolute path for the artifacts
    artifacts_location = os.path.abspath(artifacts_location)

    ovtf_src_loc = os.path.abspath(ovtf_src_loc)
    print("Source location: " + ovtf_src_loc)
    if not os.path.exists(ovtf_src_loc):
        raise AssertionError("Path doesn't exist {0}".format(ovtf_src_loc))
    os.chdir(ovtf_src_loc)

    # mkdir build directory
    path = build_dir
    try:
        try:
            if not os.path.exists(path):
                raise AssertionError("Path doesn't exist {0}".format(path))
            os.makedirs(path)
        except Exception as e:
            print("Path doesn't exist: %s" % e)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
    # Run cmake
    if not os.path.exists(path):
        raise AssertionError("Path doesn't exist {0}".format(path))
    os.chdir(path)
    if (platform.system() == 'Windows'):
        cmake_cmd = [
            "cmake", "-G \"Visual Studio 16 2019\"",
            "-DCMAKE_BUILD_TYPE=Release"
        ]
        cmake_cmd.extend(cmake_flags)
        cmake_cmd.extend([ovtf_src_loc.replace("\\", "\\\\")])
    else:
        cmake_cmd = ["cmake"]
        cmake_cmd.extend(cmake_flags)
        cmake_cmd.extend([ovtf_src_loc])
    command_executor(cmake_cmd)

    import psutil
    num_cores = int(psutil.cpu_count(logical=True))
    # get system's total RAM size in GB
    sys_ram = int(psutil.virtual_memory().total / (1024**3))
    # limiting num of cores to the max GBs of system RAM
    if (num_cores > sys_ram):
        num_cores = sys_ram
    num_cores = str(num_cores)

    if (platform.system() == 'Windows'):
        make_cmd = [
            "cmake", "--build", ".", "--config Release", "-j" + num_cores,
            "--target install"
        ]
    else:
        make_cmd = ["make", "-j" + num_cores, "install"]
    if verbose:
        make_cmd.extend(['VERBOSE=1'])

    command_executor(make_cmd)
    if not os.path.exists(os.path.join("python", "dist")):
        raise AssertionError("Path doesn't exist {0}".format(
            os.path.join("python", "dist")))
    os.chdir(os.path.join("python", "dist"))
    ovtf_wheel_files = glob.glob("openvino_tensorflow*.whl")
    if (len(ovtf_wheel_files) != 1):
        print("Multiple Python whl files exist. Please remove old wheels")
        for whl in ovtf_wheel_files:
            print("Existing Wheel: " + whl)
        raise Exception("Error getting the openvino_tensorflow wheel file")

    output_wheel = ovtf_wheel_files[0]
    print("OUTPUT WHL FILE: %s" % output_wheel)

    output_path = os.path.join(artifacts_location, output_wheel)
    print("OUTPUT WHL DST: %s" % output_path)
    # Delete just in case it exists
    try:
        # assert os.path.exists(output_path), "Output path doesn't exist {0}".format(output_path)
        if os.path.exists(output_path):
            os.remove(output_path)
    except OSError:
        pass

    # Now copy
    shutil.copy2(output_wheel, artifacts_location)
    if not os.path.exists(pwd):
        raise AssertionError("Path doesn't exist {0}".format(pwd))
    os.chdir(pwd)
    return output_wheel


def install_openvino_tf(tf_version, venv_dir, ovtf_pip_whl):
    # Load the virtual env
    load_venv(venv_dir)

    command_executor(
        ["pip", "install", "--force-reinstall", "-U", ovtf_pip_whl])

    import tensorflow as tf
    print('\033[1;34mVersion information\033[0m')
    print('TensorFlow version: ', tf.__version__)
    print('C Compiler version used in building TensorFlow: ',
          tf.__compiler_version__)
    # [TODO] Find an alternative method to do an import check as
    # doing it before source /path/to/openvino/bin/setupvars.sh
    # results in undefined symbol
    # import openvino_tensorflow
    # print(openvino_tensorflow.__version__)


def download_repo(target_name, repo, version, submodule_update=False):
    # First download to a temp folder
    command = "git clone  {repo} {target_name}".format(
        repo=repo, target_name=target_name)
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    so, se = process.communicate()

    pwd = os.getcwd()
    if not os.path.exists(target_name):
        raise AssertionError("Path doesn't exist {0}".format(target_name))
    os.chdir(target_name)

    # checkout the specified branch and get the latest changes
    process = subprocess.Popen(shlex.split("git fetch"))
    so, se = process.communicate()
    command_executor(["git", "checkout", version])
    process = subprocess.Popen(shlex.split("git pull"))
    so, se = process.communicate()

    if submodule_update:
        process = subprocess.Popen(
            shlex.split("git submodule update --init --recursive"))
        so, se = process.communicate()
    if not os.path.exists(pwd):
        raise AssertionError("Path doesn't exist {0}".format(pwd))
    os.chdir(pwd)


def download_github_release_asset(version, asset_name):
    script = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "download_asset.sh")
    command_executor(["bash", script, version, asset_name])


def apply_patch(patch_file, level=1):
    # IF patching TensorFlow unittests is done through an automation system,
    # please ensure the latest `libdvdnav-dev` or `libdvdnav-devel` is installed.
    patch_command = ['patch', '-P', str(level), '-N', '-i', patch_file]
    cmd = subprocess.Popen(patch_command, stdout=subprocess.PIPE)
    printed_lines = cmd.communicate()
    # Check if the patch is being applied for the first time, in which case
    # cmd.returncode will be 0 or if the patch has already been applied, in
    # which case the string will be found, in all other cases the assertion
    # will fail
    if not cmd.returncode == 0 or 'patch detected!  Skipping patch' in str(
            printed_lines[0]):
        raise AssertionError("Error applying the patch.")


def get_gcc_version():
    cmd = subprocess.Popen(
        shlex.split('gcc -dumpfullversion -dumpversion'),
        shell=False,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True)
    output = cmd.communicate()[0].rstrip()
    # The cmake version format is: "gcc version a.b.c"
    version_tuple = output.split('.')
    return version_tuple


def get_cmake_version():
    cmd = subprocess.Popen(
        shlex.split('cmake --version'),
        shell=False,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True)
    output = cmd.communicate()[0].rstrip()
    # The cmake version format is: "cmake version a.b.c"
    version_tuple = output.split()[2].split('.')
    return version_tuple


def get_bazel_version():
    cmd = subprocess.Popen(
        shlex.split('bazel version'),
        shell=False,
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
    # TODO: Enable other options once 80842 ticket is resolved
    openvino_cmake_flags = [
        # "-DENABLE_TESTS=OFF",
        "-DENABLE_SAMPLES=OFF",
        # "-DENABLE_FUNCTIONAL_TESTS=OFF",
        # "-DENABLE_INTEL_GNA=OFF",
        # "-DENABLE_OV_PADDLE_FRONTEND=OFF",
        # "-DENABLE_OV_ONNX_FRONTEND=OFF",
        # "-DENABLE_OV_IR_FRONTEND=ON",
        # "-DENABLE_OV_TF_FRONTEND=OFF",
        "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=" + cxx_abi,
        # "-DENABLE_OPENCV=OFF",  #Enable opencv only for ABI 1 build if required, as it is not ABI 0 compatible
        "-DCMAKE_INSTALL_RPATH=\"$ORIGIN\""
    ]

    if (platform.system() == 'Windows'):
        openvino_cmake_flags.extend([
            "-DCMAKE_INSTALL_PREFIX=" + install_location.replace("\\", "\\\\"),
            "-G \"Visual Studio 16 2019\"", "-A x64"
        ])
    else:
        openvino_cmake_flags.extend(
            ["-DCMAKE_INSTALL_PREFIX=" + install_location])

    atom_flags = ""
    if (target_arch == "silvermont"):
        atom_flags = " -mcx16 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mno-avx"
        openvino_cmake_flags.extend(["-DCMAKE_CXX_FLAGS= -march=" + atom_flags])

    if debug_enabled:
        openvino_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    cmake_build(build_dir, openvino_src_dir, openvino_cmake_flags, verbosity)


def build_protobuf(artifacts_location, protobuf_branch, debug_enabled,
                   verbosity):
    pwd = os.getcwd()
    os.chdir(artifacts_location)
    download_repo(
        "protobuf",
        "https://github.com/protocolbuffers/protobuf.git",
        protobuf_branch,
        submodule_update=True)

    src_location = os.path.abspath(os.path.join(artifacts_location, "protobuf"))
    print("Source location: " + src_location)
    os.chdir(src_location)
    os.chdir("cmake")

    # mkdir build directory
    try:
        os.makedirs("build")
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir("build"):
            pass
    os.chdir("build")

    if debug_enabled:
        try:
            os.makedirs("debug")
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir("debug"):
                pass
        os.chdir("debug")
        cmake_cmd = [
            "cmake", "-G \"Visual Studio 16 2019\"", "-DCMAKE_BUILD_TYPE=Debug",
            "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_INSTALL_PREFIX=../../../../install", "../.."
        ]
        command_executor(cmake_cmd, verbose=True)

        cmd = ["cmake", "--build", ".", "--config Debug"]
        if verbosity:
            cmd.extend(['--verbose'])
        command_executor(cmd, verbose=True)
    else:
        try:
            os.makedirs("release")
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir("release"):
                pass
        os.chdir("release")
        cmake_cmd = [
            "cmake", "-G \"Visual Studio 16 2019\"",
            "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_INSTALL_PREFIX=../../../../install", "../.."
        ]
        command_executor(cmake_cmd, verbose=True)

        cmd = ["cmake", "--build", ".", "--config Release"]
        if verbosity:
            cmd.extend(['--verbose'])
        command_executor(cmd, verbose=True)

    os.chdir(pwd)
