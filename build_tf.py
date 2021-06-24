#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

from tools.build_utils import *
import os, shutil
import argparse


def main():
    # Command line parser options
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--tf_version',
        type=str,
        help="TensorFlow tag/branch/SHA\n",
        action="store",
        default="v2.5.0")
    parser.add_argument(
        '--output_dir',
        type=str,
        help="Location where TensorFlow build will happen\n",
        action="store",
        required=True)
    parser.add_argument(
        '--target_arch',
        help=
        "Architecture flag to use (e.g., haswell, core-avx2 etc. Default \'native\'\n",
        default="native")
    parser.add_argument(
        '--use_intel_tensorflow',
        help="Build using Intel TensorFlow.",
        action="store_true")
    parser.add_argument(
        '--cxx11_abi_version',
        help="Desired version of ABI to be used while building Tensorflow",
        default='0')
    parser.add_argument(
        '--resource_usage_ratio',
        help="Ratio of CPU / RAM resources to utilize during Tensorflow build",
        default=0.5)
    arguments = parser.parse_args()

    if not os.path.isdir(arguments.output_dir):
        os.makedirs(arguments.output_dir)

    assert os.path.isdir(arguments.output_dir), \
        "Did not find output directory: " + arguments.output_dir

    os.chdir(arguments.output_dir)

    venv_dir = './venv3/'

    install_virtual_env(venv_dir)
    load_venv(venv_dir)
    setup_venv(venv_dir)

    if not os.path.isdir(os.path.join(arguments.output_dir, "tensorflow")):
        # Download TensorFlow
        download_repo("tensorflow",
                      "https://github.com/tensorflow/tensorflow.git",
                      arguments.tf_version)
    else:
        pwd = os.getcwd()
        os.chdir(os.path.join(arguments.output_dir, "tensorflow"))
        call(["git", "fetch"])
        command_executor(["git", "checkout", arguments.tf_version])
        call(["git", "pull"])
        os.chdir(pwd)

    # Build TensorFlow
    build_tensorflow(
        arguments.tf_version,
        "tensorflow",
        'artifacts',
        arguments.target_arch,
        False,
        arguments.use_intel_tensorflow,
        arguments.cxx11_abi_version,
        resource_usage_ratio=float(arguments.resource_usage_ratio))

    # Build TensorFlow C++ Library
    build_tensorflow_cc(
        arguments.tf_version, "tensorflow", 'artifacts', arguments.target_arch,
        False, arguments.use_intel_tensorflow, arguments.cxx11_abi_version)

    pwd = os.getcwd()
    artifacts_dir = os.path.join(pwd, 'artifacts/tensorflow')
    os.chdir("tensorflow")

    copy_tf_to_artifacts(arguments.tf_version, artifacts_dir, None,
                         arguments.use_intel_tensorflow)

    print('\033[1;35mTensorFlow Build finished\033[0m')

    print(
        'When building openvino_tensorflow using this prebuilt tensorflow, use:'
    )
    print('\033[3;34mpython3 build_ovtf.py --use_tensorflow_from_location ' +
          os.path.abspath(arguments.output_dir) + '\033[1;0m')


if __name__ == '__main__':
    main()
    # Usage
    # Build TF once
    # ./build_tf.py --tf_version v1.15.2 --output_dir /prebuilt/tf/dir
    #
    # Reuse TF in different openvino_tensorflow builds
    # mkdir ovtf_1; cd ovtf_1
    # git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
    # ./build_ovtf.py --use_tensorflow_from_location /prebuilt/tf/dir
    # cd ..; mkdir ovtf_2; cd ovtf_2
    # git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
    # ./build_ovtf.py --use_tensorflow_from_location /prebuilt/tf/dir
