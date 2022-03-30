#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

from tools.build_utils import *

# grappler related defaults
builder_version = 0.50
flag_string_map = {True: 'YES', False: 'NO'}


def version_check(use_prebuilt_tensorflow, use_tensorflow_from_location,
                  disable_cpp_api):
    # Check pre-requisites
    if use_prebuilt_tensorflow and not disable_cpp_api:
        # Check if the gcc version is at least 5.3.0
        if (platform.system() != 'Darwin' and platform.system() != 'Windows'):
            gcc_ver_list = get_gcc_version()
            gcc_ver = float(".".join(gcc_ver_list[:2]))
            gcc_desired_version = 5.3
            if gcc_ver < gcc_desired_version:
                raise Exception(
                    "Need GCC " + str(gcc_desired_version) +
                    " or newer to build using prebuilt TensorFlow\n"
                    "Gcc version installed: " + '.'.join(gcc_ver_list) + "\n"
                    "To build from source omit `use_prebuilt_tensorflow`")
    # Check cmake version
    cmake_ver_list = get_cmake_version()
    cmake_ver = float(".".join(cmake_ver_list[:2]))
    cmake_desired_version = 3.14
    if cmake_ver < cmake_desired_version:
        raise Exception("Need minimum cmake version " +
                        str(cmake_desired_version) + " \n"
                        "Got: " + '.'.join(cmake_ver_list))

    if not use_tensorflow_from_location and not disable_cpp_api and not use_prebuilt_tensorflow:
        # Check bazel version
        bazel_kind, bazel_ver = get_bazel_version()
        got_correct_bazel_version = bazel_kind == 'Bazelisk version'
        if (not got_correct_bazel_version and int(bazel_ver[0]) < 2):
            raise Exception("Need bazel version >= 2.0.0 \n" + "Got: " +
                            '.'.join(bazel_ver))


def main():
    '''
    Builds TensorFlow, OpenVINO, and OpenVINO integration with TensorFlow for Python 3
    '''

    # Component versions
    tf_version = "v2.8.0"
    ovtf_version = "v2.0.0"
    use_intel_tf = False

    # Command line parser options
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--debug_build',
        help="Builds a debug version of the components\n",
        action="store_true")

    parser.add_argument(
        '--verbose_build',
        help="Display verbose error messages\n",
        action="store_true")

    parser.add_argument(
        '--target_arch',
        help=
        "Architecture flag to use (e.g., haswell, core-avx2 etc. Default \'native\'\n"
    )

    parser.add_argument(
        '--tf_version',
        type=str,
        help=
        "Tensorflow version to be used for pulling from pypi / building from source",
        default=tf_version,
        action="store")

    parser.add_argument(
        '--build_tf_from_source',
        help="Builds TensorFlow from source. \n" +
        "You can choose to specify the version using the --tf_version flag. \n"
        + "If version isn't specified, TF version " + tf_version +
        " will be used.\n" +
        "Note: in this case C++ API, unit tests and examples will be built for "
        + "OpenVINO integration with TensorFlow",
        action="store_true")

    if (builder_version > 0.50):
        parser.add_argument(
            '--use_grappler_optimizer',
            help="Use Grappler optimizer instead of the optimization passes\n",
            action="store_true")

    parser.add_argument(
        '--artifacts_dir',
        type=str,
        help="Copy the artifacts to the given directory\n",
        action="store")

    parser.add_argument(
        '--use_tensorflow_from_location',
        help=
        "Use TensorFlow from a directory where it was already built and stored.\n"
        "NOTE: This location is expected to be populated by build_tf.py\n",
        action="store",
        default='')

    parser.add_argument(
        '--use_openvino_from_location',
        help=
        "Use OpenVINO from a directory where it was already built and stored.\n"
        "NOTE: This location is expected to be populated by build_ov.py\n",
        action="store",
        default='')

    parser.add_argument(
        '--disable_packaging_openvino_libs',
        help="Use this option to do build a standalone python package of " +
        "the OpenVINO integration with TensorFlow Library without OpenVINO libraries",
        action="store_true")

    parser.add_argument(
        '--disable_cpp_api',
        help="Disables C++ API, unit tests and examples\n",
        action="store_true")

    parser.add_argument(
        '--cxx11_abi_version',
        help="Desired version of ABI to be used while building Tensorflow, \n" +
        "OpenVINO integration with TensorFlow, and OpenVINO libraries",
        default='0')

    parser.add_argument(
        '--resource_usage_ratio',
        help="Ratio of CPU / RAM resources to utilize during Tensorflow build",
        default=0.5)

    parser.add_argument(
        '--openvino_version',
        help="Openvino version to be used for building from source",
        default='2022.1.0')

    parser.add_argument(
        '--python_executable',
        help="Use a specific python executable while building whl",
        action="store",
        default='')

    parser.add_argument(
        '--build_dir',
        help="Specify a custom directory during build",
        action="store",
        default='build_cmake')

    parser.add_argument(
        '--protobuf_branch',
        help="Protobuf branch to be used for the Windows build",
        action="store",
        default='v3.18.1')
    # Done with the options. Now parse the commandline
    arguments = parser.parse_args()

    if arguments.cxx11_abi_version == "1" and not arguments.build_tf_from_source:
        if not (tf_version == arguments.tf_version):
            raise AssertionError(
                "Currently ABI1 Tensorflow %s wheel is unavailable. " %
                arguments.tf_version +
                "Please consider adding --build_tf_from_source")

    # Update the build time tensorflow version with the user specified version
    tf_version = arguments.tf_version

    if (arguments.debug_build):
        print("Building in debug mode\n")

    verbosity = False
    if (arguments.verbose_build):
        print("Building with verbose output messages\n")
        verbosity = True

    #-------------------------------
    # Recipe
    #-------------------------------

    # Default directories
    build_dir = arguments.build_dir

    if (arguments.use_tensorflow_from_location != '' and
            arguments.build_tf_from_source):
        raise AssertionError(
            "\"use_tensorflow_from_location\" and \"build_tf_from_source\" "
            "cannot be used together.")
    if (arguments.openvino_version not in ["master", "2022.1.0"]):
        raise AssertionError(
            "Only 2022.1.0 OpenVINO version and master branch are supported")

    if arguments.use_openvino_from_location != '':
        if not os.path.isdir(arguments.use_openvino_from_location):
            raise AssertionError("Path doesn't exist {0}".format(
                arguments.use_openvino_from_location))
        ver_file = arguments.use_openvino_from_location + \
                      '/runtime/version.txt'
        if not os.path.exists(ver_file):
            raise AssertionError("Path doesn't exist {0}".format(ver_file))
        with open(ver_file) as f:
            line = f.readline()
            if not line.find(arguments.openvino_version) != -1:
                raise AssertionError("OpenVINO version " + \
                 arguments.openvino_version + \
                    " does not match the version specified in use_openvino_from_location")

    version_check((not arguments.build_tf_from_source),
                  (arguments.use_tensorflow_from_location != ''),
                  arguments.disable_cpp_api)

    if arguments.use_tensorflow_from_location != '':
        if (platform.system() == 'Windows'):
            arguments.use_tensorflow_from_location = arguments.use_tensorflow_from_location.replace(
                "\\", "\\\\")

        # Check if the prebuilt folder has necessary files
        if not os.path.isdir(arguments.use_tensorflow_from_location):
            raise AssertionError("Prebuilt TF path " +
                                 arguments.use_tensorflow_from_location +
                                 " does not exist")
        if (platform.system() == 'Windows'):
            loc = arguments.use_tensorflow_from_location + '\\\\tensorflow'  #'\\\\artifacts\\\\tensorflow'
            loc = loc + '\\\\bazel-bin\\\\tensorflow'
        else:
            loc = arguments.use_tensorflow_from_location + '/artifacts/tensorflow'
        if not os.path.isdir(loc):
            raise AssertionError(
                "Could not find artifacts/tensorflow directory")
        found_whl = False
        found_libtf_fw = False
        found_libtf_cc = False
        if not os.path.exists(loc):
            raise AssertionError("Path doesn't exist {0}".format(loc))
        for i in os.listdir(loc):
            if (platform.system() == 'Windows'):
                if 'tensorflow_cc' in i:
                    found_libtf_cc = True
                # if 'tensorflow.lib' in i:
                found_libtf_fw = True
                found_whl = True
            else:
                if '.whl' in i:
                    found_whl = True
                if 'libtensorflow_cc' in i:
                    found_libtf_cc = True
                if 'libtensorflow_framework' in i:
                    found_libtf_fw = True
        if not found_whl:
            raise AssertionError("Did not find TF whl file")
        if not found_libtf_fw:
            raise AssertionError("Did not find libtensorflow_framework")
        if not found_libtf_cc:
            raise AssertionError("Did not find libtensorflow_cc")

    try:
        os.makedirs(build_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(build_dir):
            pass

    pwd = os.getcwd()
    openvino_tf_src_dir = os.path.abspath(pwd)
    print("OVTF SRC DIR: " + openvino_tf_src_dir)
    build_dir_abs = os.path.abspath(build_dir)
    if not os.path.exists(build_dir_abs):
        raise AssertionError("Directory doesn't exist {}".format(build_dir_abs))
    os.chdir(build_dir)

    venv_dir = 'venv-tf-py3'
    artifacts_location = 'artifacts'
    if arguments.artifacts_dir:
        artifacts_location = os.path.abspath(arguments.artifacts_dir)

    artifacts_location = os.path.abspath(artifacts_location)

    #If artifacts doesn't exist create
    if not os.path.isdir(artifacts_location):
        os.mkdir(artifacts_location)

    print("ARTIFACTS location: " + artifacts_location)

    #install virtualenv
    install_virtual_env(venv_dir)

    # Load the virtual env
    load_venv(venv_dir)

    # Setup the virtual env
    setup_venv(venv_dir, tf_version)

    target_arch = 'native'
    if (arguments.target_arch):
        target_arch = arguments.target_arch

    # The cxx_abi flag is translated to _GLIBCXX_USE_CXX11_ABI
    # For gcc older than 5.3, this flag is set to 0 and for newer ones,
    # this is set to 1
    # Tensorflow still uses ABI 0 for its PyPi builds, and OpenVINO uses ABI 1
    # To maintain compatibility, a single ABI flag should be used for both builds
    cxx_abi = arguments.cxx11_abi_version

    # TensorFlow Build
    if arguments.use_tensorflow_from_location != "":
        # Some asserts to make sure the directory structure of
        # use_tensorflow_from_location is correct. The location
        # should have: ./artifacts/tensorflow, which is expected
        # to contain one TF whl file, framework.so and cc.so
        print("Using TensorFlow from " + arguments.use_tensorflow_from_location)
        tf_in_artifacts = os.path.join(
            os.path.abspath(artifacts_location), "tensorflow")
        if not os.path.isdir(tf_in_artifacts):
            os.mkdir(tf_in_artifacts)
        cwd = os.getcwd()

        # TF on windows is build separately and not using build_tf.py
        # and there is no artifacts folder in TF source location
        if (platform.system() == 'Windows'):
            tf_whl = os.path.abspath(
                arguments.use_tensorflow_from_location +
                "\\\\tensorflow\\\\tensorflow-2.8.0-cp39-cp39-win_amd64.whl")
            command_executor([
                "pip", "install", "--force-reinstall",
                tf_whl.replace("\\", "\\\\")
            ])
            tf_source_loc = os.path.abspath(
                os.path.join(arguments.use_tensorflow_from_location,
                             "tensorflow"))
            os.chdir(tf_source_loc)
            copy_tf_to_artifacts(tf_version, tf_in_artifacts, tf_source_loc,
                                 use_intel_tf)
        else:
            # The tf whl should be in use_tensorflow_from_location/artifacts/tensorflow
            tf_whl_loc = os.path.abspath(arguments.use_tensorflow_from_location
                                         + '/artifacts/tensorflow')
            if not os.path.exists(tf_whl_loc):
                raise AssertionError(
                    "path doesn't exist {0}".format(tf_whl_loc))
            possible_whl = [i for i in os.listdir(tf_whl_loc) if '.whl' in i]
            if not len(possible_whl) == 1:
                raise AssertionError("Expected one TF whl file, but found " +
                                     len(possible_whl))
            # Make sure there is exactly 1 TF whl
            tf_whl = os.path.abspath(tf_whl_loc + '/' + possible_whl[0])
            if not os.path.isfile(tf_whl):
                raise AssertionError("Did not find " + tf_whl)
            # Install the found TF whl file
            command_executor(
                ["pip", "install", "--force-reinstall", "-U", tf_whl])
            tf_cxx_abi = get_tf_cxxabi()

            if not (arguments.cxx11_abi_version == tf_cxx_abi):
                raise AssertionError(
                    "Desired ABI version and user built tensorflow library provided with "
                    "use_tensorflow_from_location are incompatible")

            if not os.path.exists(tf_whl_loc):
                raise AssertionError(
                    "Path doesn't exist {0}".format(tf_whl_loc))
            os.chdir(tf_whl_loc)

            # This function copies the .so files from
            # use_tensorflow_from_location/artifacts/tensorflow to
            # artifacts/tensorflow
            tf_version = get_tf_version()
            copy_tf_to_artifacts(tf_version, tf_in_artifacts, tf_whl_loc,
                                 use_intel_tf)
        if not os.path.exists(cwd):
            raise AssertionError("Path doesn't exist {0}".format(cwd))
        os.chdir(cwd)
    else:
        if not arguments.build_tf_from_source:
            print("Install TensorFlow")
            # get the python version tag
            tags = next(sys_tags())
            if arguments.cxx11_abi_version == "0":
                if (platform.system() == "Windows"):
                    if tags.interpreter == "cp39":
                        command_executor([
                            "pip", "install", "--force-reinstall",
                            "https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow-2.8.0-cp39-cp39-win_amd64.whl"
                        ])
                    else:
                        raise AssertionError(
                            "Only python39 is supported on Windows")

                else:
                    command_executor([
                        "pip", "install", "--force-reinstall",
                        "tensorflow==" + tf_version
                    ])
            elif arguments.cxx11_abi_version == "1":
                if tags.interpreter == "cp37":
                    command_executor([
                        "pip", "install", "--force-reinstall",
                        "https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow_abi1-2.8.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl"
                    ])
                if tags.interpreter == "cp38":
                    command_executor([
                        "pip", "install", "--force-reinstall",
                        "https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow_abi1-2.8.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl"
                    ])
                if tags.interpreter == "cp39":
                    command_executor([
                        "pip", "install", "--force-reinstall",
                        "https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.0.0/tensorflow_abi1-2.8.0-cp39-cp39-manylinux_2_12_x86_64.manylinux2010_x86_64.whl"
                    ])
                # ABI 1 TF required latest numpy
                command_executor(
                    ["pip", "install", "--force-reinstall", "-U numpy"])

            tf_cxx_abi = get_tf_cxxabi()

            if not (arguments.cxx11_abi_version == tf_cxx_abi):
                raise AssertionError(
                    "Desired ABI version and tensorflow library installed with "
                    "pip are incompatible")

            tf_src_dir = os.path.join(artifacts_location, "tensorflow")
            print("TF_SRC_DIR: ", tf_src_dir)
            # Download TF source for enabling TF python tests
            pwd_now = os.getcwd()
            if not os.path.exists(artifacts_location):
                raise AssertionError(
                    "Path doesn't exist {0}".format(artifacts_location))
            os.chdir(artifacts_location)
            print("DOWNLOADING TF: PWD", os.getcwd())
            download_repo("tensorflow",
                          "https://github.com/tensorflow/tensorflow.git",
                          tf_version)
            print("Using TensorFlow version", tf_version)
            if not os.path.exists(pwd_now):
                raise AssertionError("Path doesn't exist {0}".format(pwd_now))
            os.chdir(pwd_now)
            # Finally, copy the libtensorflow_framework.so to the artifacts
            if (tf_version.startswith("v1.") or (tf_version.startswith("1."))):
                tf_fmwk_lib_name = 'libtensorflow_framework.so.1'
            else:
                tf_fmwk_lib_name = 'libtensorflow_framework.so.2'
            if (platform.system() == 'Darwin'):
                if (tf_version.startswith("v1.") or
                    (tf_version.startswith("1."))):
                    tf_fmwk_lib_name = 'libtensorflow_framework.1.dylib'
                else:
                    tf_fmwk_lib_name = 'libtensorflow_framework.2.dylib'
            elif (platform.system() == 'Windows'):
                tf_fmwk_lib_name = '_pywrap_tensorflow_internal.lib'
                tf_fmwk_dll_name = '_pywrap_tensorflow_internal.pyd'

            import tensorflow as tf
            tf_lib_dir = tf.sysconfig.get_lib()
            if (platform.system() == 'Windows'):
                tf_lib_file = os.path.join(tf_lib_dir, "python",
                                           tf_fmwk_lib_name)
                tf_dll_file = os.path.join(tf_lib_dir, "python",
                                           tf_fmwk_dll_name)
            else:
                tf_lib_file = os.path.join(tf_lib_dir, tf_fmwk_lib_name)
            print("SYSCFG LIB: ", tf_lib_file)
            dst_dir = os.path.join(artifacts_location, "tensorflow")
            if not os.path.exists(dst_dir):
                raise AssertionError(
                    "Directory doesn't exist {0}".format(dst_dir))
            if not os.path.isdir(dst_dir):
                os.mkdir(dst_dir)
            dst = os.path.join(dst_dir, tf_fmwk_lib_name)
            shutil.copyfile(tf_lib_file, dst)
            # copy pyd file for windows
            if (platform.system() == 'Windows'):
                dst = os.path.join(dst_dir, tf_fmwk_dll_name)
                shutil.copyfile(tf_dll_file, dst)
        else:
            print("Building TensorFlow from source")
            if (platform.system() == 'Windows'):
                os.chdir(artifacts_location)
            # Download TensorFlow
            download_repo("tensorflow",
                          "https://github.com/tensorflow/tensorflow.git",
                          tf_version)

            tf_src_dir = os.path.join(os.getcwd(), "tensorflow")
            print("TF_SRC_DIR: ", tf_src_dir)

            # Build TensorFlow
            build_tensorflow(
                tf_version,
                "tensorflow",
                artifacts_location,
                target_arch,
                verbosity,
                use_intel_tf,
                arguments.cxx11_abi_version,
                resource_usage_ratio=float(arguments.resource_usage_ratio))

            # Now build the libtensorflow_cc.so - the C++ library
            build_tensorflow_cc(tf_version, tf_src_dir, artifacts_location,
                                target_arch, verbosity, use_intel_tf,
                                arguments.cxx11_abi_version)

            if (platform.system() == 'Windows'):
                os.chdir(build_dir_abs)

            tf_cxx_abi = install_tensorflow(venv_dir, artifacts_location)

            # This function copies the .so files from
            # use_tensorflow_from_location/artifacts/tensorflow to
            # artifacts/tensorflow
            cwd = os.getcwd()
            os.chdir(tf_src_dir)
            dst_dir = os.path.join(artifacts_location, "tensorflow")
            copy_tf_to_artifacts(tf_version, dst_dir, None, use_intel_tf)
            os.chdir(cwd)

    # OpenVINO Build
    if arguments.use_openvino_from_location != "":
        print("Using OpenVINO from " + arguments.use_openvino_from_location)
    else:
        print("Building OpenVINO from source")
        print(
            "NOTE: OpenVINO python module is not built when building from source."
        )
        if (arguments.openvino_version == "master"):
            openvino_release_tag = "master"
        elif (arguments.openvino_version == "2022.1.0"):
            openvino_release_tag = "2022.1.0"

        # Download OpenVINO
        download_repo(
            "openvino",
            "https://github.com/openvinotoolkit/openvino.git",
            openvino_release_tag,
            submodule_update=True)
        openvino_src_dir = os.path.join(os.getcwd(), "openvino")
        print("OV_SRC_DIR: ", openvino_src_dir)

        build_openvino(build_dir, openvino_src_dir, cxx_abi, target_arch,
                       artifacts_location, arguments.debug_build, verbosity)

    # Next build CMAKE options for the openvino-tensorflow
    if (platform.system() == 'Windows'):
        openvino_tf_cmake_flags = [
            "-DOPENVINO_TF_INSTALL_PREFIX=" + artifacts_location.replace(
                "\\", "/")
        ]
    else:
        openvino_tf_cmake_flags = [
            "-DOPENVINO_TF_INSTALL_PREFIX=" + artifacts_location
        ]

    atom_flags = ""
    if (target_arch == "silvermont"):
        atom_flags = " -mcx16 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mno-avx"
        openvino_tf_cmake_flags.extend(
            ["-DCMAKE_CXX_FLAGS= -march=" + atom_flags])

    openvino_artifacts_dir = ""
    if arguments.use_openvino_from_location == '':
        openvino_artifacts_dir = os.path.join(artifacts_location, "openvino")
    else:
        openvino_artifacts_dir = os.path.abspath(
            arguments.use_openvino_from_location)
        openvino_tf_cmake_flags.extend(["-DUSE_OPENVINO_FROM_LOCATION=TRUE"])
    print("openvino_artifacts_dir: ", openvino_artifacts_dir)
    if (platform.system() == 'Windows'):
        openvino_tf_cmake_flags.extend([
            "-DOPENVINO_ARTIFACTS_DIR='" + openvino_artifacts_dir.replace(
                "\\", "/") + "'"
        ])
    else:
        openvino_tf_cmake_flags.extend(
            ["-DOPENVINO_ARTIFACTS_DIR=" + openvino_artifacts_dir])

    openvino_tf_cmake_flags.extend(
        ["-DOPENVINO_VERSION=" + arguments.openvino_version])

    if (arguments.debug_build):
        openvino_tf_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    if arguments.use_tensorflow_from_location:
        if (platform.system() == 'Windows'):
            openvino_tf_cmake_flags.extend([
                "-DTF_SRC_DIR=" + (os.path.abspath(
                    arguments.use_tensorflow_from_location.replace(
                        "\\", "\\\\") + '\\tensorflow')).replace("\\", "\\\\")
            ])
        else:
            openvino_tf_cmake_flags.extend([
                "-DTF_SRC_DIR=" + os.path.abspath(
                    arguments.use_tensorflow_from_location + '/tensorflow')
            ])
    else:
        if not arguments.disable_cpp_api and arguments.build_tf_from_source:
            print("TF_SRC_DIR: ", tf_src_dir)
            if (platform.system() == 'Windows'):
                openvino_tf_cmake_flags.extend(
                    ["-DTF_SRC_DIR=" + tf_src_dir.replace("\\", "\\\\")])
            else:
                openvino_tf_cmake_flags.extend(["-DTF_SRC_DIR=" + tf_src_dir])

    openvino_tf_cmake_flags.extend(["-DUNIT_TEST_ENABLE=ON"])
    if not arguments.disable_cpp_api and arguments.build_tf_from_source:
        if (platform.system() == 'Windows'):
            openvino_tf_cmake_flags.extend([
                "-DUNIT_TEST_TF_CC_DIR=" + ((os.path.join(
                    artifacts_location, "tensorflow")).replace("\\", "\\\\"))
            ])
        else:
            openvino_tf_cmake_flags.extend([
                "-DUNIT_TEST_TF_CC_DIR=" + os.path.join(artifacts_location,
                                                        "tensorflow")
            ])

    if (builder_version > 0.50):
        openvino_tf_cmake_flags.extend([
            "-DOPENVINO_TF_USE_GRAPPLER_OPTIMIZER=" +
            flag_string_map[arguments.use_grappler_optimizer]
        ])

    if arguments.disable_packaging_openvino_libs:
        openvino_tf_cmake_flags.extend(["-DDISABLE_PACKAGING_OPENVINO_LIBS=1"])
    if arguments.python_executable != '':
        openvino_tf_cmake_flags.extend(
            ["-DPYTHON_EXECUTABLE=%s" % arguments.python_executable])
    if (platform.system() == 'Windows'):
        openvino_tf_cmake_flags.extend(
            ["-DTensorFlow_CXX_ABI=" + arguments.cxx11_abi_version])
        openvino_tf_cmake_flags.extend(
            ["-DTensorFlow_GIT_VERSION=" + tf_version])
        openvino_tf_cmake_flags.extend(
            ["-DTensorFlow_VERSION=" + tf_version.replace("v", "")])

    # add openvino build version as compile time definition
    openvino_tf_cmake_flags.extend(
        ["-DOPENVINO_BUILD_VERSION=%s" % str(arguments.openvino_version)])

    # Now build openvino-tensorflow
    ov_tf_whl = build_openvino_tf(build_dir, artifacts_location,
                                  openvino_tf_src_dir, venv_dir,
                                  openvino_tf_cmake_flags, verbosity)

    # Make sure that the openvino_tensorflow whl is present in the artfacts directory
    if not os.path.exists(artifacts_location):
        raise AssertionError("Path not found {}".format(artifacts_location))
    if not os.path.isfile(os.path.join(artifacts_location, ov_tf_whl)):
        raise AssertionError(
            "Cannot locate openvino-tensorflow whl in the artifacts location")
    if not os.path.isfile(os.path.join(artifacts_location, ov_tf_whl)):
        raise Exception(
            "Cannot locate openvino-tensorflow whl in the artifacts location")

    print("SUCCESSFULLY generated wheel: %s" % ov_tf_whl)
    print("PWD: " + os.getcwd())

    # Copy the TensorFlow Python code tree to artifacts directory so that they can
    # be used for running TensorFlow Python unit tests
    #
    # There are four possibilities:
    # 1. use_tensorflow_from_location is not defined
    #   2. In that case build_tf_from_source is not defined
    #       In this case we copy the entire tensorflow source to the artifacts
    #       So all we have to do is to create a symbolic link
    #       3. OR build_tf_from_source is defined
    # 4. use_tensorflow_from_location is defined
    if arguments.use_tensorflow_from_location == '':
        # Case 1
        if not arguments.build_tf_from_source:
            # Case 2
            base_dir = None
        else:
            # Case 3
            base_dir = build_dir_abs
    else:
        # Case 4
        base_dir = arguments.use_tensorflow_from_location

    if base_dir != None:
        dest_dir = os.path.join(artifacts_location, "tensorflow")
        if (platform.system() == 'Windows'):
            command_executor(
                [
                    'cp',
                    '-r',
                    base_dir +
                    '\\\\tensorflow\\\\tensorflow\\\\python',  # base_dir + '\\\\artifacts\\\\tensorflow\\\\tensorflow\\\\python',
                    dest_dir.replace("\\", "\\\\")
                ],
                verbose=True)
        else:
            command_executor([
                'cp', '-r', base_dir + '/tensorflow/tensorflow/python', dest_dir
            ],
                             verbose=True)
    else:
        # Create a sym-link to
        if (platform.system() == 'Windows'):
            link_src = os.path.join(artifacts_location,
                                    "tensorflow\\tensorflow\\python").replace(
                                        "\\", "\\\\")
            link_dst = os.path.join(artifacts_location,
                                    "tensorflow\\python").replace("\\", "\\\\")
            # if destination link already exists, then delete it
            if (os.path.exists(link_dst)):
                print("Link %s already exists, deleting it." % link_dst)
                command_executor(['rmdir /s /q', link_dst], shell=True)
            command_executor(['mklink /D', link_dst, link_src],
                             verbose=True,
                             shell=True)
        else:
            link_src = os.path.join(artifacts_location,
                                    "tensorflow/tensorflow/python")
            link_dst = os.path.join(artifacts_location, "tensorflow/python")

            command_executor(['ln', '-sf', link_src, link_dst], verbose=True)

    if not os.path.exists(artifacts_location):
        raise AssertionError("Path doesn't exist {}".format(artifacts_location))
    # Run a quick test
    if (platform.system() == 'Windows'):
        install_openvino_tf(
            tf_version, venv_dir,
            os.path.join(artifacts_location, ov_tf_whl).replace("\\", "\\\\"))
    else:
        install_openvino_tf(tf_version, venv_dir,
                            os.path.join(artifacts_location, ov_tf_whl))

    if builder_version > 0.50 and arguments.use_grappler_optimizer:
        import tensorflow as tf
        import openvino_tensorflow
        if not openvino_tensorflow.is_grappler_enabled():
            raise Exception(
                "Build failed: 'use_grappler_optimizer' specified but not used")

    print('\033[1;32mBuild successful\033[0m')
    os.chdir(pwd)


if __name__ == '__main__':
    main()
