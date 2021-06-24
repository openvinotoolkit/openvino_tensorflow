#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2021 Intel Corporation

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
        if (platform.system() != 'Darwin'):
            gcc_ver = get_gcc_version()
            if gcc_ver < '5.3.0':
                raise Exception(
                    "Need GCC 5.3.0 or newer to build using prebuilt TensorFlow\n"
                    "Gcc version installed: " + gcc_ver + "\n"
                    "To build from source omit `use_prebuilt_tensorflow`")
    # Check cmake version
    cmake_ver = get_cmake_version()
    if (int(cmake_ver[0]) < 3 or int(cmake_ver[1]) < 14):
        raise Exception("Need minimum cmake version 3.14\n"
                        "Got: " + '.'.join(cmake_ver))

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
    tf_version = "v2.5.0"
    ovtf_version = "v0.5.0"
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
        default='2021.3')

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

    # Done with the options. Now parse the commandline
    arguments = parser.parse_args()

    if arguments.cxx11_abi_version == "1" and not arguments.build_tf_from_source:
        assert (tf_version == arguments.tf_version), (
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

    assert not (
        arguments.use_tensorflow_from_location != '' and
        arguments.build_tf_from_source), (
            "\"use_tensorflow_from_location\" and \"build_tf_from_source\" "
            "cannot be used together.")

    assert not (arguments.openvino_version != "2021.3" and
                arguments.openvino_version != "2021.2"), (
                    "Only 2021.2 and 2021.3 are supported OpenVINO versions")

    if arguments.use_openvino_from_location != '':
        ver_file = arguments.use_openvino_from_location + \
                      '/deployment_tools/inference_engine/version.txt'
        with open(ver_file) as f:
            line = f.readline()
            assert line.find(arguments.openvino_version) != -1, "OpenVINO version " + \
                arguments.openvino_version + \
                " does not match the version specified in use_openvino_from_location"

    version_check((not arguments.build_tf_from_source),
                  (arguments.use_tensorflow_from_location != ''),
                  arguments.disable_cpp_api)

    if arguments.use_tensorflow_from_location != '':
        # Check if the prebuilt folder has necessary files
        assert os.path.isdir(
            arguments.use_tensorflow_from_location
        ), "Prebuilt TF path " + arguments.use_tensorflow_from_location + " does not exist"
        loc = arguments.use_tensorflow_from_location + '/artifacts/tensorflow'
        assert os.path.isdir(
            loc), "Could not find artifacts/tensorflow directory"
        found_whl = False
        found_libtf_fw = False
        found_libtf_cc = False
        for i in os.listdir(loc):
            if '.whl' in i:
                found_whl = True
            if 'libtensorflow_cc' in i:
                found_libtf_cc = True
            if 'libtensorflow_framework' in i:
                found_libtf_fw = True
        assert found_whl, "Did not find TF whl file"
        assert found_libtf_fw, "Did not find libtensorflow_framework"
        assert found_libtf_cc, "Did not find libtensorflow_cc"

    try:
        os.makedirs(build_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(build_dir):
            pass

    pwd = os.getcwd()
    openvino_tf_src_dir = os.path.abspath(pwd)
    print("OVTF SRC DIR: " + openvino_tf_src_dir)
    build_dir_abs = os.path.abspath(build_dir)
    os.chdir(build_dir)

    venv_dir = 'venv-tf-py3'
    artifacts_location = 'artifacts'
    if arguments.artifacts_dir:
        artifacts_location = os.path.abspath(arguments.artifacts_dir)

    artifacts_location = os.path.abspath(artifacts_location)
    print("ARTIFACTS location: " + artifacts_location)

    #If artifacts doesn't exist create
    if not os.path.isdir(artifacts_location):
        os.mkdir(artifacts_location)

    #install virtualenv
    install_virtual_env(venv_dir)

    # Load the virtual env
    load_venv(venv_dir)

    # Setup the virtual env
    setup_venv(venv_dir)

    target_arch = 'native'
    if (arguments.target_arch):
        target_arch = arguments.target_arch

    print("Target Arch: %s" % target_arch)

    # The cxx_abi flag is translated to _GLIBCXX_USE_CXX11_ABI
    # For gcc older than 5.3, this flag is set to 0 and for newer ones,
    # this is set to 1
    # Tensorflow still uses ABI 0 for its PyPi builds, and OpenVINO uses ABI 1
    # To maintain compatibility, a single ABI flag should be used for both builds
    cxx_abi = arguments.cxx11_abi_version

    if arguments.use_tensorflow_from_location != "":
        # Some asserts to make sure the directory structure of
        # use_tensorflow_from_location is correct. The location
        # should have: ./artifacts/tensorflow, which is expected
        # to contain one TF whl file, framework.so and cc.so
        print("Using TensorFlow from " + arguments.use_tensorflow_from_location)
        # The tf whl should be in use_tensorflow_from_location/artifacts/tensorflow
        tf_whl_loc = os.path.abspath(arguments.use_tensorflow_from_location +
                                     '/artifacts/tensorflow')
        possible_whl = [i for i in os.listdir(tf_whl_loc) if '.whl' in i]
        assert len(
            possible_whl
        ) == 1, "Expected one TF whl file, but found " + len(possible_whl)
        # Make sure there is exactly 1 TF whl
        tf_whl = os.path.abspath(tf_whl_loc + '/' + possible_whl[0])
        assert os.path.isfile(tf_whl), "Did not find " + tf_whl
        # Install the found TF whl file
        command_executor(["pip", "install", "-U", tf_whl])
        tf_cxx_abi = get_tf_cxxabi()

        assert (arguments.cxx11_abi_version == tf_cxx_abi), (
            "Desired ABI version and user built tensorflow library provided with "
            "use_tensorflow_from_location are incompatible")

        cwd = os.getcwd()
        os.chdir(tf_whl_loc)
        tf_in_artifacts = os.path.join(
            os.path.abspath(artifacts_location), "tensorflow")
        if not os.path.isdir(tf_in_artifacts):
            os.mkdir(tf_in_artifacts)
        # This function copies the .so files from
        # use_tensorflow_from_location/artifacts/tensorflow to
        # artifacts/tensorflow
        tf_version = get_tf_version()
        copy_tf_to_artifacts(tf_version, tf_in_artifacts, tf_whl_loc,
                             use_intel_tf)
        os.chdir(cwd)
    else:
        if not arguments.build_tf_from_source:
            print("Using TensorFlow version", tf_version)
            print("Install TensorFlow")

            if arguments.cxx11_abi_version == "0":
                command_executor(
                    ["pip", "install", "tensorflow==" + tf_version])
            elif arguments.cxx11_abi_version == "1":
                tags = next(sys_tags())

                if tags.interpreter == "cp36":
                    command_executor([
                        "pip", "install",
                        "https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp36-cp36m-manylinux2010_x86_64.whl"
                    ])
                if tags.interpreter == "cp37":
                    command_executor([
                        "pip", "install",
                        "https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp37-cp37m-manylinux2010_x86_64.whl"
                    ])
                if tags.interpreter == "cp38":
                    command_executor([
                        "pip", "install",
                        "https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp38-cp38-manylinux2010_x86_64.whl"
                    ])

                # ABI 1 TF required latest numpy
                command_executor(["pip", "install", "-U numpy"])

            tf_cxx_abi = get_tf_cxxabi()

            assert (arguments.cxx11_abi_version == tf_cxx_abi), (
                "Desired ABI version and tensorflow library installed with "
                "pip are incompatible")

            tf_src_dir = os.path.join(artifacts_location, "tensorflow")
            print("TF_SRC_DIR: ", tf_src_dir)
            # Download TF source for enabling TF python tests
            pwd_now = os.getcwd()
            os.chdir(artifacts_location)
            print("DOWNLOADING TF: PWD", os.getcwd())
            download_repo("tensorflow",
                          "https://github.com/tensorflow/tensorflow.git",
                          tf_version)
            os.chdir(pwd_now)
            # Finally, copy the libtensorflow_framework.so to the artifacts
            if (tf_version.startswith("v1.") or (tf_version.startswith("1."))):
                tf_fmwk_lib_name = 'libtensorflow_framework.so.1'
            else:
                tf_fmwk_lib_name = 'libtensorflow_framework.so.2'
            if (platform.system() == 'Darwin'):
                tf_fmwk_lib_name = 'libtensorflow_framework.2.dylib'
            import tensorflow as tf
            tf_lib_dir = tf.sysconfig.get_lib()
            tf_lib_file = os.path.join(tf_lib_dir, tf_fmwk_lib_name)
            print("SYSCFG LIB: ", tf_lib_file)
            dst_dir = os.path.join(artifacts_location, "tensorflow")
            if not os.path.isdir(dst_dir):
                os.mkdir(dst_dir)
            dst = os.path.join(dst_dir, tf_fmwk_lib_name)
            shutil.copyfile(tf_lib_file, dst)
        else:
            print("Building TensorFlow from source")
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

            tf_cxx_abi = install_tensorflow(venv_dir, artifacts_location)

            # This function copies the .so files from
            # use_tensorflow_from_location/artifacts/tensorflow to
            # artifacts/tensorflow
            cwd = os.getcwd()
            os.chdir(tf_src_dir)
            dst_dir = os.path.join(artifacts_location, "tensorflow")
            copy_tf_to_artifacts(tf_version, dst_dir, None, use_intel_tf)
            os.chdir(cwd)

    if arguments.use_openvino_from_location != "":
        print("Using OpenVINO from " + arguments.use_openvino_from_location)
    else:
        print("Building OpenVINO from source")
        print(
            "NOTE: OpenVINO python module is not built when building from source."
        )

        if (arguments.openvino_version == "2021.3"):
            openvino_branch = "releases/2021/3"
        elif (arguments.openvino_version == "2021.2"):
            openvino_branch = "releases/2021/2"

        # Download OpenVINO
        download_repo(
            "openvino",
            "https://github.com/openvinotoolkit/openvino",
            openvino_branch,
            submodule_update=True)
        openvino_src_dir = os.path.join(os.getcwd(), "openvino")
        print("OV_SRC_DIR: ", openvino_src_dir)

        build_openvino(build_dir, openvino_src_dir, cxx_abi, target_arch,
                       artifacts_location, arguments.debug_build, verbosity)

    # Next build CMAKE options for the bridge
    atom_flags = ""
    if (target_arch == "silvermont"):
        atom_flags = " -mcx16 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mno-avx"
    openvino_tf_cmake_flags = [
        "-DOPENVINO_TF_INSTALL_PREFIX=" + artifacts_location,
        "-DCMAKE_CXX_FLAGS=-march=" + target_arch + atom_flags,
    ]

    openvino_artifacts_dir = ""
    if arguments.use_openvino_from_location == '':
        openvino_artifacts_dir = os.path.join(artifacts_location, "openvino")
    else:
        openvino_artifacts_dir = os.path.abspath(
            arguments.use_openvino_from_location)
        openvino_tf_cmake_flags.extend(["-DUSE_OPENVINO_FROM_LOCATION=TRUE"])
    print("openvino_artifacts_dir: ", openvino_artifacts_dir)
    openvino_tf_cmake_flags.extend(
        ["-DOPENVINO_ARTIFACTS_DIR=" + openvino_artifacts_dir])

    openvino_tf_cmake_flags.extend(
        ["-DOPENVINO_VERSION=" + arguments.openvino_version])

    if (arguments.debug_build):
        openvino_tf_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    if arguments.use_tensorflow_from_location:
        openvino_tf_cmake_flags.extend([
            "-DTF_SRC_DIR=" + os.path.abspath(
                arguments.use_tensorflow_from_location + '/tensorflow')
        ])
    else:
        if not arguments.disable_cpp_api and arguments.build_tf_from_source:
            print("TF_SRC_DIR: ", tf_src_dir)
            openvino_tf_cmake_flags.extend(["-DTF_SRC_DIR=" + tf_src_dir])

    openvino_tf_cmake_flags.extend(["-DUNIT_TEST_ENABLE=ON"])
    if not arguments.disable_cpp_api and arguments.build_tf_from_source:
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

    # Now build the bridge
    ov_tf_whl = build_openvino_tf(build_dir, artifacts_location,
                                  openvino_tf_src_dir, venv_dir,
                                  openvino_tf_cmake_flags, verbosity)

    # Make sure that the openvino_tensorflow whl is present in the artfacts directory
    if not os.path.isfile(os.path.join(artifacts_location, ov_tf_whl)):
        raise Exception("Cannot locate nGraph whl in the artifacts location")

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
        command_executor(
            ['cp', '-r', base_dir + '/tensorflow/tensorflow/python', dest_dir],
            verbose=True)
    else:
        # Create a sym-link to
        link_src = os.path.join(artifacts_location,
                                "tensorflow/tensorflow/python")
        link_dst = os.path.join(artifacts_location, "tensorflow/python")
        command_executor(['ln', '-sf', link_src, link_dst], verbose=True)

    # Run a quick test
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
