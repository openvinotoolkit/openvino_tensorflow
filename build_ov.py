#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

from tools.build_utils import *
import os, shutil
import argparse


def main():
    openvino_version = "releases/2021/3"
    build_dir = 'build_cmake'
    cxx_abi = "1"
    print("openVINO version: ", openvino_version)

    # Command line parser options
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--output_dir',
        type=str,
        help="Location where OpenVINO build will happen\n",
        action="store",
        required=True)
    parser.add_argument(
        '--target_arch',
        help=
        "Architecture flag to use (e.g., haswell, core-avx2 etc. Default \'native\'\n",
        default="native")
    parser.add_argument(
        '--debug_build',
        help="Builds a debug version of the components\n",
        action="store_true")
    arguments = parser.parse_args()

    if not os.path.isdir(arguments.output_dir):
        os.makedirs(arguments.output_dir)
    assert os.path.exists(
        arguments.output_dir), "path doesn't exist {0}".format(
            arguments.output_dir)
    os.chdir(arguments.output_dir)
    assert os.path.exists(
        arguments.output_dir), "Directory doesn't exist {0}".format(
            arguments.output_dir)
    if not os.path.isdir(os.path.join(arguments.output_dir, "openvino")):
        # Download OpenVINO
        download_repo(
            "openvino",
            "https://github.com/openvinotoolkit/openvino",
            openvino_version,
            submodule_update=True)
    else:
        pwd = os.getcwd()
        assert os.path.exists(
            arguments.output_dir), "path doesn't exist {0}".format(
                arguments.output_dir)
        os.chdir(os.path.join(arguments.output_dir, "openvino"))
        call(["git", "fetch"])
        command_executor(["git", "checkout", openvino_version])
        call(["git", "pull"])
        assert os.path.exists(pwd), "Path doesn't exist {0}".format(pwd)
        os.chdir(pwd)

    openvino_src_dir = os.path.join(arguments.output_dir, "openvino")
    print("OV_SRC_DIR: ", openvino_src_dir)

    verbosity = False
    artifacts_location = os.path.abspath(arguments.output_dir) + '/artifacts'

    # Build OpenVINO
    build_openvino(build_dir, openvino_src_dir, cxx_abi, arguments.target_arch,
                   artifacts_location, arguments.debug_build, verbosity)

    print('\033[1;35mOpenVINO Build finished\033[0m')

    print(
        'When building Openvino_Tensorflow using this prebuilt OpenVINO, use:')
    print('\033[3;34mpython3 build_ovtf.py --use_openvino_from_location ' +
          os.path.abspath(arguments.output_dir) + '/artifacts/openvino' +
          '\033[1;0m')


if __name__ == '__main__':
    main()
    # Usage
    # Build OV once
    # ./build_ov.py --output_dir /prebuilt/ov/dir
    #
    # Reuse OV in different openvino_tensorflow builds
    # mkdir ovtf_1; cd ovtf_1
    # git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
    # ./build_ovtf.py --use_openvino_from_location /prebuilt/ov/dir/artifacts/openvino
    # cd ..; mkdir ovtf_2; cd ovtf_2
    # git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
    # ./build_ovtf.py --use_openvino_from_location /prebuilt/ov/dir/artifacts/openvino
