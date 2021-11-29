# **OpenVINO™ integration with TensorFlow**

[**OpenVINO™ integration with TensorFlow**](https://github.com/openvinotoolkit/openvino_tensorflow/) is a product designed for TensorFlow* developers who want to get started with [OpenVINO™](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) in their inferencing applications. This product delivers [OpenVINO™](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) inline optimizations which enhance inferencing performance with minimal code modifications. **OpenVINO™ integration with TensorFlow** accelerates inference across many [AI models](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/MODELS.md) on a variety of Intel<sup>®</sup> silicon such as:
- Intel<sup>®</sup> CPUs
- Intel<sup>®</sup> integrated GPUs
- Intel<sup>®</sup> Movidius™ Vision Processing Units - referred to as VPU
- Intel<sup>®</sup> Vision Accelerator Design with 8 Intel Movidius™ MyriadX VPUs - referred to as VAD-M or HDDL

[Note: For maximum performance, efficiency, tooling customization, and hardware control, we recommend the developers to adopt native OpenVINO™ APIs and its runtime.]

## Installation

### Requirements

- Ubuntu 18.04, macOS 11.2.3 or Windows<sup>1</sup> 10 - 64 bit
- Python* 3.7, 3.8 or 3.9
- TensorFlow* v2.7.0

<sup>1</sup>Windows release is in preview mode and supports only Python3.9 

This **OpenVINO™ integration with TensorFlow** package comes with pre-built libraries of OpenVINO™ version 2021.4.2 meaning you do not have to install OpenVINO™ separately.
This package supports:
- Intel<sup>®</sup> CPUs
- Intel<sup>®</sup> integrated GPUs
- Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs)

        pip3 install -U pip
        pip3 install tensorflow==2.7.0
        pip3 install -U openvino-tensorflow

To leverage Intel® Vision Accelerator Design with Movidius™ (VAD-M) for inference, please refer to: [**OpenVINO™ integration with TensorFlow** alongside the Intel® Distribution of OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/INSTALL.md#12-install-openvino-integration-with-tensorflow-alongside-the-intel-distribution-of-openvino-toolkit).

For installation instructions on Windows please refer to [**OpenVINO™ integration with TensorFlow** for Windows ](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/INSTALL.md#InstallOpenVINOintegrationwithTensorFlowalongsideTensorFlow)

For more details on installation please refer to [INSTALL.md](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/INSTALL.md), and for build from source options please refer to [BUILD.md](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/BUILD.md)

## Verify Installation

Once you have installed **OpenVINO™ integration with TensorFlow**, you can use TensorFlow to run inference using a trained model.

To check if **OpenVINO™ integration with TensorFlow** is properly installed, run

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This should produce an output like:

        TensorFlow version:  2.7.0
        OpenVINO integration with TensorFlow version: b'1.1.0'
        OpenVINO version used for this build: b'2021.4.2'
        TensorFlow version used for this build: v2.7.0
        CXX11_ABI flag used for this build: 0

## Usage

By default, Intel<sup>®</sup> CPU is used to run inference. However, you can change the default option to either Intel<sup>®</sup> integrated GPU or Intel<sup>®</sup> VPU for AI inferencing. Invoke the following function to change the hardware on which inferencing is done.

    openvino_tensorflow.set_backend('<backend_name>')

Supported backends include 'CPU', 'GPU', 'GPU_FP16', and 'MYRIAD'.

To determine what processing units are available on your system for inference, use the following function:

    openvino_tensorflow.list_backends()

For more API calls and environment variables, see [USAGE.md](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/USAGE.md).

[Note: For the best results with TensorFlow, it is advised to enable [oneDNN Deep Neural Network Library (oneDNN)](https://github.com/oneapi-src/oneDNN) by setting the environment variable `TF_ENABLE_ONEDNN_OPTS=1`]

[Note: If a CUDA capable device is present in the system then set the environment variable CUDA_VISIBLE_DEVICES to -1]  

## Examples

To see what you can do with **OpenVINO™ integration with TensorFlow**, explore the demos located in the [examples](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples) repository.

## License
**OpenVINO™ integration with TensorFlow** is licensed under [Apache License Version 2.0](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.  

## Support

Please submit your questions, feature requests and bug reports via [**GitHub issues**](https://github.com/openvinotoolkit/openvino_tensorflow/issues).

## How to Contribute

We welcome community contributions to **OpenVINO™ integration with TensorFlow**. If you have an idea for improvement:

* Share your proposal via [GitHub issues](https://github.com/openvinotoolkit/openvino_tensorflow/issues).
* Submit a [pull request](https://github.com/openvinotoolkit/openvino_tensorflow/pulls).

We will review your contribution as soon as possible. If any additional fixes or modifications are necessary, we will guide you and provide feedback. Before you make your contribution, make sure you can build **OpenVINO™ integration with TensorFlow** and run all the examples with your fix/patch. If you want to introduce a large feature, create test cases for your feature. Upon the verification of your pull request, we will merge it to the repository provided that the pull request has met the above mentioned requirements and proved acceptable.

---
\* Other names and brands may be claimed as the property of others.