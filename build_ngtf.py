#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
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

from tools.build_utils import *


def version_check(use_prebuilt_tensorflow):
    # Check pre-requisites
    if use_prebuilt_tensorflow:
        # Check if the gcc version is 4.8
        if (platform.system() != 'Darwin'):
            gcc_ver = get_gcc_version()
            if '4.8' not in gcc_ver:
                raise Exception(
                    "Need GCC 4.8 to build using prebuilt TensorFlow\n"
                    "Gcc version installed: " + gcc_ver + "\n"
                    "To build from source ommit `use_prebuilt_tensorflow`")
    # Check cmake version
    cmake_ver = get_cmake_version()
    if (int(cmake_ver[0]) < 3 or int(cmake_ver[1]) < 4):
        raise Exception("Need minimum cmake version 3.4\n"
                        "Got: " + '.'.join(cmake_ver))

    # Check bazel version
    bazel_ver = get_bazel_version()
    got_correct_bazel_version = False
    if (int(bazel_ver[1]) >= 24 and int(bazel_ver[1]) <= 25):
        if (int(bazel_ver[2]) >= 1 and int(bazel_ver[2]) <= 2):
            got_correct_bazel_version = True

    if not got_correct_bazel_version:
        raise Exception("Need bazel 0.24.1 < version < 0.25.2 \n" + "Got: " +
                        '.'.join(bazel_ver))


def main():
    '''
    Builds TensorFlow, ngraph, and ngraph-tf for python 3
    '''

    # Component versions
    ngraph_version = "v0.23.0-rc.5"
    tf_version = "v1.14.0"

    # Command line parser options
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--debug_build',
        help="Builds a debug version of the nGraph components\n",
        action="store_true")

    parser.add_argument(
        '--verbose_build',
        help="Display verbose error messages\n",
        action="store_true")

    parser.add_argument(
        '--target_arch',
        help=
        "Architecture flag to use (e.g., haswell, core-avx2 etc. Default \'native\'\n",
    )

    parser.add_argument(
        '--build_gpu_backend',
        help=
        "nGraph backends will include nVidia GPU. Use: NGRAPH_TF_BACKEND=GPU\n"
        "Note: You need to have CUDA headers and libraries available on the build system.\n",
        action="store_true")

    parser.add_argument(
        '--build_plaidml_backend',
        help=
        "nGraph backends will include PlaidML backend. Use: NGRAPH_TF_BACKEND=PLAIDML\n",
        action="store_true")

    parser.add_argument(
        '--build_intelgpu_backend',
        help=
        "nGraph backends will include Intel GPU bckend. Use: NGRAPH_TF_BACKEND=INTELGPU\n",
        action="store_true")

    parser.add_argument(
        '--use_prebuilt_tensorflow',
        help="Skip building TensorFlow and use downloaded version.\n" +
        "Note that in this case C++ unit tests won't be build for nGraph-TF bridge",
        action="store_true")

    parser.add_argument(
        '--distributed_build',
        type=str,
        help="Builds a distributed version of the nGraph components\n",
        action="store")

    parser.add_argument(
        '--enable_variables_and_optimizers',
        help=
        "Ops like variable and optimizers are supported by nGraph in this version of the bridge\n",
        action="store_true")

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
        '--ngraph_src_dir',
        type=str,
        help=
        "Local nGraph source directory to use. Overrides --ngraph_version.\n",
        action="store")

    parser.add_argument(
        '--ngraph_version',
        type=str,
        help="nGraph version to use. Overridden by --ngraph_src_dir. (Default: "
        + ngraph_version + ")\n",
        action="store")

    parser.add_argument(
        '--use_tensorflow_from_location',
        help=
        "Use TensorFlow from a directory where it was already built and stored.\n"
        "This location is expected to be populated by build_tf.py\n",
        action="store",
        default='')

    # Done with the options. Now parse the commandline
    arguments = parser.parse_args()

    if (arguments.debug_build):
        print("Building in DEBUG mode\n")

    verbosity = False
    if (arguments.verbose_build):
        print("Building in with VERBOSE output messages\n")
        verbosity = True

    #-------------------------------
    # Recipe
    #-------------------------------

    version_check(arguments.use_prebuilt_tensorflow)

    # Default directories
    build_dir = 'build_cmake'

    assert not (
        arguments.use_tensorflow_from_location != '' and
        arguments.use_prebuilt_tensorflow
    ), "\"use_tensorflow_from_location\" and \"use_prebuilt_tensorflow\" "
    "cannot be used together."

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
    ngraph_tf_src_dir = os.path.abspath(pwd)
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
    # For gcc 4.8 - this flag is set to 0 and newer ones, this is set to 1
    # The specific value is determined from the TensorFlow build
    # Normally the shipped TensorFlow is built with gcc 4.8 and thus this
    # flag is set to 0
    cxx_abi = "0"

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
        cxx_abi = get_tf_cxxabi()
        cwd = os.getcwd()
        os.chdir(tf_whl_loc)
        tf_in_artifacts = os.path.join(
            os.path.abspath(artifacts_location), "tensorflow")
        if os.path.isdir(tf_in_artifacts):
            print("TensorFlow already exists in artifacts. Using that")
        else:
            os.mkdir(tf_in_artifacts)
            # This function copies the .so files from
            # use_tensorflow_from_location/artifacts/tensorflow to
            # artifacts/tensorflow
            copy_tf_to_artifacts(tf_in_artifacts, tf_whl_loc)
        os.chdir(cwd)
    else:
        if arguments.use_prebuilt_tensorflow:
            print("Using existing TensorFlow")
            # Frst download the source. This will create the tensorfow directory as needed
            tf_src_dir = os.path.join(artifacts_location, "tensorflow")
            print("TF_SRC_DIR: ", tf_src_dir)
            # Download
            pwd_now = os.getcwd()
            os.chdir(artifacts_location)
            print("DOWNLOADING TF: PWD", os.getcwd())
            download_repo("tensorflow",
                          "https://github.com/tensorflow/tensorflow.git",
                          tf_version)
            os.chdir(pwd_now)

            # Next install the tensorflow python packge
            command_executor(
                ["pip", "install", "-U", "tensorflow==" + tf_version])
            cxx_abi = get_tf_cxxabi()

            # Copy the libtensorflow_framework.so to the artifacts so that
            # we can run c++ tests from that location later
            tf_fmwk_lib_name = 'libtensorflow_framework.so.1'
            if (platform.system() == 'Darwin'):
                tf_fmwk_lib_name = 'libtensorflow_framework.1.dylib'
            import tensorflow as tf
            tf_lib_dir = tf.sysconfig.get_lib()
            tf_lib_file = os.path.join(tf_lib_dir, tf_fmwk_lib_name)
            print("SYSCFG LIB: ", tf_lib_file)

            dst_dir = os.path.join(artifacts_location, "tensorflow")
            if not os.path.isdir(dst_dir):
                os.mkdir(dst_dir)

            dst = os.path.join(dst_dir, tf_fmwk_lib_name)
            shutil.copyfile(tf_lib_file, dst)

            # Now build the libtensorflow_cc.so - the C++ library
            build_tensorflow_cc(tf_src_dir, artifacts_location, target_arch,
                                verbosity)

        else:
            print("Building TensorFlow from source")
            # Download TensorFlow
            download_repo("tensorflow",
                          "https://github.com/tensorflow/tensorflow.git",
                          tf_version)
            tf_src_dir = os.path.join(os.getcwd(), "tensorflow")
            # Build TensorFlow
            build_tensorflow(venv_dir, "tensorflow", artifacts_location,
                             target_arch, verbosity)

            # Now build the libtensorflow_cc.so - the C++ library
            build_tensorflow_cc(tf_src_dir, artifacts_location, target_arch,
                                verbosity)

            # Install tensorflow to our own virtual env
            # Note that if gcc 4.8 is used for building TensorFlow this flag
            # will be 0
            cxx_abi = install_tensorflow(venv_dir, artifacts_location)

    if cxx_abi == 0:
        if not arguments.use_prebuilt_tensorflow:
            raise Exception(
                "Expected cxx_abi to be 0 when using 'use_prebuilt_tensorflow'")

    # Download nGraph if required.
    ngraph_src_dir = './ngraph'
    if arguments.ngraph_src_dir:
        ngraph_src_dir = arguments.ngraph_src_dir

        print("Using local nGraph source in directory ", ngraph_src_dir)
    else:
        if arguments.ngraph_version:
            ngraph_version = arguments.ngraph_version

        print("nGraph Version: ", ngraph_version)
        download_repo("ngraph", "https://github.com/NervanaSystems/ngraph.git",
                      ngraph_version)

    # Now build nGraph
    ngraph_cmake_flags = [
        "-DNGRAPH_INSTALL_PREFIX=" + artifacts_location,
        "-DNGRAPH_USE_CXX_ABI=" + cxx_abi,
        "-DNGRAPH_DEX_ONLY=TRUE",
        "-DNGRAPH_DEBUG_ENABLE=NO",
        "-DNGRAPH_UNIT_TEST_ENABLE=NO",
        "-DNGRAPH_TARGET_ARCH=" + target_arch,
        "-DNGRAPH_TUNE_ARCH=" + target_arch,
    ]

    if arguments.debug_build:
        ngraph_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    if (arguments.distributed_build == "OMPI"):
        ngraph_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=OMPI"])
    elif (arguments.distributed_build == "MLSL"):
        ngraph_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=MLSL"])
    else:
        ngraph_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=OFF"])

    if arguments.build_plaidml_backend:
        command_executor(["pip", "install", "-U", "plaidML"])

    flag_string_map = {True: 'YES', False: 'NO'}
    ngraph_cmake_flags.extend([
        "-DNGRAPH_TOOLS_ENABLE=" +
        flag_string_map[platform.system() != 'Darwin']
    ])
    ngraph_cmake_flags.extend(
        ["-DNGRAPH_GPU_ENABLE=" + flag_string_map[arguments.build_gpu_backend]])
    ngraph_cmake_flags.extend([
        "-DNGRAPH_PLAIDML_ENABLE=" +
        flag_string_map[arguments.build_plaidml_backend]
    ])
    ngraph_cmake_flags.extend([
        "-DNGRAPH_INTELGPU_ENABLE=" +
        flag_string_map[arguments.build_intelgpu_backend]
    ])

    build_ngraph(build_dir, ngraph_src_dir, ngraph_cmake_flags, verbosity)

    ngraph_tf_cmake_flags = [
        "-DNGRAPH_TF_INSTALL_PREFIX=" + artifacts_location,
        "-DUSE_PRE_BUILT_NGRAPH=ON",
        "-DUNIT_TEST_ENABLE=ON",
        "-DNGRAPH_TARGET_ARCH=" + target_arch,
        "-DNGRAPH_TUNE_ARCH=" + target_arch,
        "-DNGRAPH_ARTIFACTS_DIR=" + artifacts_location,
    ]

    if (arguments.debug_build):
        ngraph_tf_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    if not arguments.use_prebuilt_tensorflow:
        if arguments.use_tensorflow_from_location:
            ngraph_tf_cmake_flags.extend([
                "-DTF_SRC_DIR=" + os.path.abspath(
                    arguments.use_tensorflow_from_location + '/tensorflow')
            ])
        else:
            ngraph_tf_cmake_flags.extend(["-DTF_SRC_DIR=" + tf_src_dir])
        ngraph_tf_cmake_flags.extend([
            "-DUNIT_TEST_TF_CC_DIR=" + os.path.join(artifacts_location,
                                                    "tensorflow")
        ])

    # Next build CMAKE options for the bridge
    if arguments.use_tensorflow_from_location:
        ngraph_tf_cmake_flags.extend([
            "-DTF_SRC_DIR=" + os.path.abspath(
                arguments.use_tensorflow_from_location + '/tensorflow')
        ])
    else:
        print("TF_SRC_DIR: ", tf_src_dir)
        ngraph_tf_cmake_flags.extend(["-DTF_SRC_DIR=" + tf_src_dir])

    ngraph_tf_cmake_flags.extend([
        "-DUNIT_TEST_TF_CC_DIR=" + os.path.join(artifacts_location,
                                                "tensorflow")
    ])

    if ((arguments.distributed_build == "OMPI") or
        (arguments.distributed_build == "MLSL")):
        ngraph_tf_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=TRUE"])
    else:
        ngraph_tf_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=FALSE"])

    ngraph_tf_cmake_flags.extend([
        "-DNGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS=" +
        flag_string_map[arguments.enable_variables_and_optimizers]
    ])
    ngraph_tf_cmake_flags.extend([
        "-DNGRAPH_TF_USE_GRAPPLER_OPTIMIZER=" +
        flag_string_map[arguments.use_grappler_optimizer]
    ])

    # Now build the bridge
    ng_tf_whl = build_ngraph_tf(build_dir, artifacts_location,
                                ngraph_tf_src_dir, venv_dir,
                                ngraph_tf_cmake_flags, verbosity)

    # Make sure that the ngraph bridge whl is present in the artfacts directory
    if not os.path.isfile(os.path.join(artifacts_location, ng_tf_whl)):
        raise Exception("Cannot locate nGraph whl in the artifacts location")

    print("SUCCESSFULLY generated wheel: %s" % ng_tf_whl)
    print("PWD: " + os.getcwd())

    # Copy the TensorFlow Python code tree to artifacts directory so that they can
    # be used for running TensorFlow Python unit tests
    #
    # There are four possibilities:
    # 1. use_tensorflow_from_location is not defined
    #   2. In that case use_prebuilt_tensorflow is defined
    #       In this case we copy the entire tensorflow source to the artifacts
    #       So all we have to do is to create a symbolic link
    #       3. OR use_prebuilt_tensorflow is not defined
    # 4. use_tensorflow_from_location is defined
    if arguments.use_tensorflow_from_location == '':
        # Case 1
        if arguments.use_prebuilt_tensorflow:
            # Case 2
            base_dir = None
        else:
            # Case 3
            base_dir = build_dir_abs
    else:
        # Case 4
        base_dir = arguments.use_tensorflow_from_location

    if base_dir != None:
        command_executor([
            'cp', '-r', base_dir + '/tensorflow/tensorflow/python',
            os.path.join(artifacts_location, "tensorflow")
        ],
                         verbose=True)
    else:
        # Create a sym-link to
        link_src = os.path.join(artifacts_location,
                                "tensorflow/tensorflow/python")
        link_dst = os.path.join(artifacts_location, "tensorflow/python")
        command_executor(['ln', '-sf', link_src, link_dst], verbose=True)

    # Run a quick test
    install_ngraph_tf(venv_dir, os.path.join(artifacts_location, ng_tf_whl))

    if arguments.use_grappler_optimizer:
        import tensorflow as tf
        import ngraph_bridge
        if not ngraph_bridge.is_grappler_enabled():
            raise Exception(
                "Build failed: 'use_grappler_optimizer' specified but not used")

    print('\033[1;32mBuild successful\033[0m')
    os.chdir(pwd)


if __name__ == '__main__':
    main()
