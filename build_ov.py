#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
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
import os, shutil
import argparse


def main():
    openvino_version = "releases/2021/2"
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
    os.chdir(arguments.output_dir)

    if not os.path.isdir(os.path.join(arguments.output_dir, "openvino")):
        # Download OpenVINO
        download_repo(
            "openvino",
            "https://github.com/openvinotoolkit/openvino",
            openvino_version,
            submodule_update=True)
    else:
        pwd = os.getcwd()
        os.chdir(os.path.join(arguments.output_dir, "openvino"))
        call(["git", "fetch"])
        command_executor(["git", "checkout", openvino_version])
        call(["git", "pull"])
        os.chdir(pwd)

    openvino_src_dir = os.path.join(arguments.output_dir, "openvino")
    print("OV_SRC_DIR: ", openvino_src_dir)

    verbosity = False
    artifacts_location = os.path.abspath(arguments.output_dir) + '/artifacts'

    # Build OpenVINO
    build_openvino(build_dir, openvino_src_dir, cxx_abi, arguments.target_arch,
                   artifacts_location, arguments.debug_build, verbosity)

    print('\033[1;35mOpenVINO Build finished\033[0m')

    print('When building ngraph-bridge using this prebuilt OpenVINO, use:')
    print('\033[3;34mpython3 build_ngtf.py --use_openvino_from_location ' +
          os.path.abspath(arguments.output_dir) + '/artifacts/openvino' +
          '\033[1;0m')


if __name__ == '__main__':
    main()
    # Usage
    # Build OV once
    # ./build_ov.py --output_dir /prebuilt/ov/dir
    #
    # Reuse OV in different ngraph-bridge builds
    # mkdir ngtf_1; cd ngtf_1
    # git clone https://github.com/tensorflow/ngraph-bridge.git
    # ./build_ngtf.py --use_openvino_from_location /prebuilt/ov/dir/artifacts/openvino
    # cd ..; mkdir ngtf_2; cd ngtf_2
    # git clone https://github.com/tensorflow/ngraph-bridge.git
    # ./build_ngtf.py --use_openvino_from_location /prebuilt/ov/dir/artifacts/openvino
