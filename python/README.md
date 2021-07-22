# **OpenVINO™ integration with TensorFlow**

This repository contains the source code of **OpenVINO™ integration with TensorFlow**, a product needed to enable OpenVINO™ runtime and optimizations for TensorFlow. **OpenVINO™ integration with TensorFlow** enables acceleration of AI inferencing across a vast number of use cases, using a variety of AI models, on a variety of Intel<sup>®</sup> silicon such as:
- Intel<sup>®</sup> CPUs 
- Intel<sup>®</sup> integrated GPUs 
- Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs) - referred as VPU
- Intel<sup>®</sup> Vision accelerator Design with 8 Intel Movidius™ MyriadX VPUs - referred as VAD-M or HDDL.

## Installation

### Requirements

- Python 3.6, 3.7, or 3.8
- TensorFlow 2.5.0

### Use **OpenVINO™ integration with TensorFlow** alongside PyPi TensorFlow

This **OpenVINO™ integration with TensorFlow** package comes with pre-built libraries of OpenVINO™ version 2021.3. The users do not have to install OpenVINO™ separately. This package supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs).

        pip3 install -U pip==21.0.1
        pip3 install -U tensorflow==2.5.0
        pip3 install openvino-tensorflow

To use **OpenVINO™ integration with TensorFlow** with pre-installed OpenVINO™ binaries, please visit this page for detailed instructions: ([**OpenVINO™ integration with TensorFlow** - README](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/BUILD.md))
## Verify Installation

Verify that `openvino-tensorflow` installed correctly:

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This will produce something like this:

        TensorFlow version:  2.5.0
        OpenVINO integration with TensorFlow version: b'0.5.0'
        OpenVINO version used for this build: b'2021.3'
        TensorFlow version used for this build: v2.5.0
        CXX11_ABI flag used for this build: 0
        OpenVINO integration with TensorFlow built with Grappler: False

Test the installation:

        python3 test_ovtf.py

This command runs all C++ and Python unit tests from the `openvino_tensorflow` source tree.

## Usage

Once you have installed **OpenVINO™ integration with TensorFlow**, you can use TensorFlow to run inference using a trained model.
The only change required to a script is adding

    import openvino_tensorflow

To determine what backends are available on your system, use the following API:

    openvino_tensorflow.list_backends()

By default, CPU backend is enabled. You can substitute the default CPU backend with a different backend by using the following API:

    openvino_tensorflow.set_backend('backend_name')

More detailed examples on how to use **OpenVINO™ integration with TensorFlow** are located in the [**examples**](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples) directory.

## License
**OpenVINO™ integration with TensorFlow** is licensed under [Apache License Version 2.0](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.  

## Support

Please submit your questions, feature requests and bug reports via [**GitHub issues**](https://github.com/openvinotoolkit/openvino_tensorflow/issues).

## How to Contribute

We welcome community contributions to **OpenVINO™ integration with TensorFlow**. If you have an idea for how to 
improve it:

* Share your proposal via [**GitHub issues**](https://github.com/openvinotoolkit/openvino_tensorflow/issues).
* Ensure you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [**pull request**](https://github.com/openvinotoolkit/openvino_tensorflow/pulls).
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged to the repository.
* All guidelines for contributing to the OpenVINO repositories can be found [here](https://github.com/openvinotoolkit/openvino/wiki/Contribute)