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

from tools.build_utils import *


def main():
    '''
    Builds TensorFlow, ngraph, and ngraph-tf for python 3
    '''

    # Component versions
    ngraph_version = "v0.18.0-rc.2"
    tf_version = "v1.13.1"

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
        help="nGraph backends will include nVidia GPU.\n"
        "Note: You need to have CUDA headers and libraries available on the build system.\n",
        action="store_true")

    parser.add_argument(
        '--build_plaidml_backend',
        help="nGraph backends will include PlaidML bckend\n",
        action="store_true")

    parser.add_argument(
        '--use_prebuilt_tensorflow',
        help="Skip building TensorFlow and use downloaded version.\n" +
        "Note that in this case C++ unit tests won't be build for nGrapg-TF bridge",
        action="store_true")

    parser.add_argument(
        '--distributed_build',
        type=str,
        help="Builds a distributed version of the nGraph components\n",
        action="store")

    parser.add_argument(
        '--enable_variables_and_optimizers',
        help="Ops like variable and optimizers are supported by nGraph in this version of the bridge\n",
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
        '--ngraph_version',
        type=str,
        help="nGraph version to use (Default: " + ngraph_version + ")\n",
        action="store")

    parser.add_argument(
        '--skip_tensorflow_build',
        help="Use TensorFlow that's already installed" +
        "(do not build or install) \n",
        action="store_true")

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

    # Default directories
    build_dir = 'build_cmake'

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

    if arguments.use_prebuilt_tensorflow:
        print("Using existing TensorFlow")
        command_executor(["pip", "install", "-U", "tensorflow==" + tf_version])

        import tensorflow as tf
        print('Version information:')
        print('TensorFlow version: ', tf.__version__)
        print('C Compiler version used in building TensorFlow: ',
              tf.__compiler_version__)
        cxx_abi = str(tf.__cxx11_abi_flag__)
    else:
        if not arguments.skip_tensorflow_build:
            print("Building TensorFlow")
            # Download TensorFlow
            download_repo("tensorflow",
                          "https://github.com/tensorflow/tensorflow.git",
                          tf_version)

            # Build TensorFlow
            build_tensorflow(venv_dir, "tensorflow", artifacts_location,
                             target_arch, verbosity)

            # Install tensorflow
            # Note that if gcc 4.8 is used for building TensorFlow this flag
            # will be 0
            cxx_abi = install_tensorflow(venv_dir, artifacts_location)
        else:
            import tensorflow as tf
            print('Version information:')
            print('TensorFlow version: ', tf.__version__)
            print('C Compiler version used in building TensorFlow: ',
                  tf.__compiler_version__)
            cxx_abi = str(tf.__cxx11_abi_flag__)

    # Download nGraph
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
        "-DNGRAPH_TARGET_ARCH=" + target_arch,
        "-DNGRAPH_TUNE_ARCH=" + target_arch,
    ]
    if (platform.system() != 'Darwin'):
        ngraph_cmake_flags.extend(["-DNGRAPH_TOOLS_ENABLE=YES"])
    else:
        ngraph_cmake_flags.extend(["-DNGRAPH_TOOLS_ENABLE=NO"])

    if arguments.debug_build:
        ngraph_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    if (arguments.distributed_build == "OMPI"):
        ngraph_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=OMPI"])
    elif (arguments.distributed_build == "MLSL"):
        ngraph_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=MLSL"])
    else:
        ngraph_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=OFF"])

    if arguments.build_gpu_backend:
        ngraph_cmake_flags.extend(["-DNGRAPH_GPU_ENABLE=YES"])
    else:
        ngraph_cmake_flags.extend(["-DNGRAPH_GPU_ENABLE=NO"])

    if arguments.build_plaidml_backend:
        command_executor(["pip", "install", "-U", "plaidML"])
        ngraph_cmake_flags.extend(["-DNGRAPH_PLAIDML_ENABLE=YES"])
    else:
        ngraph_cmake_flags.extend(["-DNGRAPH_PLAIDML_ENABLE=NO"])

    if not arguments.use_prebuilt_tensorflow:
        ngraph_cmake_flags.extend(["-DNGRAPH_UNIT_TEST_ENABLE=YES"])
    else:
        ngraph_cmake_flags.extend(["-DNGRAPH_UNIT_TEST_ENABLE=NO"])

    build_ngraph(build_dir, "./ngraph", ngraph_cmake_flags, verbosity)

    # Next build CMAKE options for the bridge
    tf_src_dir = os.path.abspath("tensorflow")

    ngraph_tf_cmake_flags = [
        "-DNGRAPH_TF_INSTALL_PREFIX=" + artifacts_location,
        "-DUSE_PRE_BUILT_NGRAPH=ON",
        "-DNGRAPH_TARGET_ARCH=" + target_arch,
        "-DNGRAPH_TUNE_ARCH=" + target_arch,
        "-DNGRAPH_ARTIFACTS_DIR=" + artifacts_location,
    ]
    if (arguments.debug_build):
        ngraph_tf_cmake_flags.extend(["-DCMAKE_BUILD_TYPE=Debug"])

    if arguments.use_prebuilt_tensorflow:
        ngraph_tf_cmake_flags.extend(["-DUNIT_TEST_ENABLE=OFF"])
    else:
        ngraph_tf_cmake_flags.extend(["-DUNIT_TEST_ENABLE=ON"])
        ngraph_tf_cmake_flags.extend(["-DTF_SRC_DIR=" + tf_src_dir])
        ngraph_tf_cmake_flags.extend([
            "-DUNIT_TEST_TF_CC_DIR=" + os.path.join(artifacts_location,
                                                    "tensorflow")
        ])

    if ((arguments.distributed_build == "OMPI")
            or (arguments.distributed_build == "MLSL")):
        ngraph_tf_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=TRUE"])
    else:
        ngraph_tf_cmake_flags.extend(["-DNGRAPH_DISTRIBUTED_ENABLE=FALSE"])

    if (arguments.enable_variables_and_optimizers):
        ngraph_tf_cmake_flags.extend(["-DNGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS=TRUE"])
    else:
        ngraph_tf_cmake_flags.extend(["-DNGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS=FALSE"])
        
    if (arguments.use_grappler_optimizer):
        ngraph_tf_cmake_flags.extend(
            ["-DNGRAPH_TF_USE_GRAPPLER_OPTIMIZER=TRUE"])
    else:
        ngraph_tf_cmake_flags.extend(
            ["-DNGRAPH_TF_USE_GRAPPLER_OPTIMIZER=FALSE"])

    # Now build the bridge
    ng_tf_whl = build_ngraph_tf(build_dir, artifacts_location,
                                ngraph_tf_src_dir, venv_dir,
                                ngraph_tf_cmake_flags, verbosity)

    print("SUCCESSFULLY generated wheel: %s" % ng_tf_whl)
    print("PWD: " + os.getcwd())

    # Copy the TensorFlow Python code tree to artifacts directory so that they can
    # be used for running TensorFlow Python unit tests
    if not arguments.use_prebuilt_tensorflow:
        command_executor([
            'cp', '-r', build_dir_abs + '/tensorflow/tensorflow/python',
            os.path.join(artifacts_location, "tensorflow")
        ])

    # Run a quick test
    install_ngraph_tf(venv_dir, os.path.join(artifacts_location, ng_tf_whl))

    print('\033[1;32mBuild successful\033[0m')
    os.chdir(pwd)


if __name__ == '__main__':
    main()
