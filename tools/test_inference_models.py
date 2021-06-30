#!/usr/bin/env python3
# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

from build_utils import *


def main():
    '''
    Builds TensorFlow, ngraph, and ngraph-tf for python 3
    '''
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--model_location',
        help="Location of the model directory\n",
        action="store")

    parser.add_argument(
        '--python_location',
        help=
        "Location of Python executable (whether virtual environment or system)\n",
        action="store")

    parser.add_argument(
        '--artifacts_dir',
        type=str,
        help="Location of the artifacts\n",
        action="store")

    arguments = parser.parse_args()

    #-------------------------------
    # Recipe
    #-------------------------------

    # Save the current location
    pwd = os.getcwd()

    if not (arguments.artifacts_dir):
        raise "Need to provide artifacts directory"

    if not (arguments.python_location):
        raise "Need to provide the location of Python directory"

    if not (arguments.model_location):
        raise "Need to provide the location of models directory"

    # Check to make sure that there is TensorFlow installed and will work
    # TODO

    # Install nGraph Bridge
    ovtf_wheel_files = glob.glob(
        os.path.join(
            os.path.abspath(arguments.artifacts_dir),
            "openvino_tensorflow*.whl"))
    if (len(ovtf_wheel_files) != 1):
        raise ("Multiple Python whl files exist. Please remove old wheels")

    openvino_tensorflow_wheel = ovtf_wheel_files[0]

    print("NGRAPH Wheel: ", openvino_tensorflow_wheel)
    assert os.path.exists(arguments.python_location), "Could not find the path"
    command_executor([
        os.path.join(arguments.python_location, "pip"), "install", "-U",
        openvino_tensorflow_wheel
    ])

    # Print the version information
    print("\nnGraph-TensorFlow Information ")
    assert os.path.exists(arguments.python_location), "Path doesn't exist"
    python_exe = os.path.join(arguments.python_location, "python3")
    command_executor([
        python_exe, "-c", "\"import tensorflow as tf; " +
        "print('TensorFlow version: ',tf.__version__);" +
        "import openvino_tensorflow; print(openvino_tensorflow.__version__)" +
        ";print(openvino_tensorflow.list_backends())\n\""
    ])

    # Next is to go to the model directory
    assert os.path.exists(arguments.model_location), "Could not find the path"
    os.chdir(os.path.join(arguments.model_location, "tensorflow_scripts"))

    # Execute the inference runs
    command_executor(["/bin/bash", "run-all-models.sh"])

    # Restore
    os.chdir(pwd)


if __name__ == '__main__':
    main()
