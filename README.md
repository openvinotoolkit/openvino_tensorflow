<p align="center">
  <img src="images/openvino_wbgd.png">
</p>

# **OpenVINO™ integration with TensorFlow (Preview Release)**

This repository contains the source code of **OpenVINO™ integration with TensorFlow**, a product that delivers [OpenVINO™](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) inline optimizations and runtime needed for an enhanced level of TensorFlow compatibility. It is designed for developers who want to get started with OpenVINO™ in their inferencing applications to enhance inferencing performance with minimal code modifications. **OpenVINO™ integration with TensorFlow** accelerates inference across many [AI models](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/MODELS.md) on a variety of Intel<sup>®</sup> silicon such as:
- Intel<sup>®</sup> CPUs
- Intel<sup>®</sup> integrated GPUs
- Intel<sup>®</sup> Movidius™ Vision Processing Units - referred as VPU
- Intel<sup>®</sup> Vision Accelerator Design with 8 Intel Movidius™ MyriadX VPUs - referred as VAD-M or HDDL

[Note: For maximum performance, efficiency, tooling customization, and hardware control, we recommend going beyond this component to adopt native OpenVINO™ APIs and its runtime.]

## Installation
### Prerequisites

- Ubuntu 18.04, 20.04
- Python 3.6, 3.7, or 3.8
- TensorFlow v2.5.0

Check our [Interactive Installation Table](https://openvinotoolkit.github.io/openvino_tensorflow/) for a menu of installation options. The table will help you configure the installation process.

### Install **OpenVINO™ integration with TensorFlow** alongside PyPi TensorFlow

This **OpenVINO™ integration with TensorFlow** package comes with pre-built libraries of OpenVINO™ version 2021.3 meaning you don't have to install OpenVINO™ separately. This package supports:
- Intel<sup>®</sup> CPUs
- Intel<sup>®</sup> integrated GPUs
- Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs)


        pip3 install -U pip==21.0.1
        pip3 install -U tensorflow==2.5.0
        pip3 install openvino-tensorflow


If you want to leverage Intel® Vision Accelerator Design with Movidius™ (VAD-M) for inference, install [**OpenVINO™ integration with TensorFlow** alongside the Intel® Distribution of OpenVINO™ Toolkit](docs/BUILD.md#install-openvino-integration-with-tensorflow-alongside-the-intel-distribution-of-openvino-toolkit).

For more details on other modes of installation, please refer to [BUILD.md](docs/BUILD.md)

## Configuration

Once you've installed **OpenVINO™ integration with TensorFlow**, you can use TensorFlow to run inference using a trained model.

To see if **OpenVINO™ integration with TensorFlow** is properly installed, run

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This should produce an output like:

        TensorFlow version:  2.5.0
        OpenVINO integration with TensorFlow version: b'0.5.0'
        OpenVINO version used for this build: b'2021.3'
        TensorFlow version used for this build: v2.5.0
        CXX11_ABI flag used for this build: 0
        OpenVINO integration with TensorFlow built with Grappler: False

By default, Intel<sup>®</sup> CPU is used to run inference. However, you can change the default option to either Intel<sup>®</sup> integrated GPU or Intel<sup>®</sup> VPU for AI inferencing. Invoke the following function to change the hardware on which inferencing is done.

    openvino_tensorflow.set_backend('<backend_name>')

Supported backends include 'CPU', 'GPU', 'MYRIAD', and 'VAD-M'.

To determine what processing units are available on your system for inference, use the following function:

    openvino_tensorflow.list_backends()
For more API calls and environment variables, see [USAGE.md](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/USAGE.md).

## Examples

To see what you can do with **OpenVINO™ integration with TensorFlow**, explore the demos located in the [examples](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples) directory.

## License
**OpenVINO™ integration with TensorFlow** is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## Support

Submit your questions, feature requests and bug reports via [GitHub issues](https://github.com/openvinotoolkit/openvino_tensorflow/issues).

## How to Contribute

We welcome community contributions to **OpenVINO™ integration with TensorFlow**. If you have an idea for improvement:

* Share your proposal via [GitHub issues](https://github.com/openvinotoolkit/openvino_tensorflow/issues).
* Submit a [pull request](https://github.com/openvinotoolkit/openvino_tensorflow/pulls).

We will review your contribution as soon as possible. If any additional fixes or modifications are necessary, we will guide you and provide feedback. Before you make your contribution, make sure you can build **OpenVINO™ integration with TensorFlow** and run all the examples with your fix/patch. If you want to introduce a large feature, create test cases for your feature. Upon our verification of your pull request, we will merge it to the repository provided that the pull request has met the above mentioned requirements and proved acceptable.
