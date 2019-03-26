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
    ngtf_wheel_files = glob.glob(
        os.path.join(
            os.path.abspath(arguments.artifacts_dir),
            "ngraph_tensorflow_bridge-*.whl"))
    if (len(ngtf_wheel_files) != 1):
        raise ("Multiple Python whl files exist. Please remove old wheels")

    ngraph_bridge_wheel = ngtf_wheel_files[0]

    print("NGRAPH Wheel: ", ngraph_bridge_wheel)
    command_executor([
        os.path.join(arguments.python_location, "pip"), "install", "-U",
        ngraph_bridge_wheel
    ])

    # Print the version information
    print("\nnGraph-TensorFlow Information ")
    python_exe = os.path.join(arguments.python_location, "python3")
    command_executor([
        python_exe, "-c", "\"import tensorflow as tf; " +
        "print('TensorFlow version: ',tf.__version__);" +
        "import ngraph_bridge; print(ngraph_bridge.__version__)" +
        ";print(ngraph_bridge.list_backends())\n\""
    ])

    # Next is to go to the model directory
    os.chdir(os.path.join(arguments.model_location, "tensorflow_scripts"))

    # Execute the inference runs
    command_executor(["/bin/bash", "run-all-models.sh"])

    # Restore
    os.chdir(pwd)


if __name__ == '__main__':
    main()
