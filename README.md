<p>English | <a href="./README_cn.md">简体中文</a></p>

<p align="center">
  <img src="images/openvino_wbgd.png">
</p>

# **OpenVINO™ integration with TensorFlow will no longer be supported as of OpenVINO™ 2023.0 release.**
If you are looking to deploy your TensorFlow models on Intel based devices, you have a few options.

If you prefer the native TensorFlow framework APIs, consider using the [Intel Extension for TensorFlow (ITEX)](https://github.com/intel/intel-extension-for-tensorflow). Another option is to utilize the [OpenVINO Model Conversion API](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html), which enables the automatic importation and conversion of standard TensorFlow models during runtime. It is not necessary to convert your TensorFlow models offline now..

# **OpenVINO™ integration with TensorFlow**

This repository contains the source code of **OpenVINO™ integration with TensorFlow**, designed for TensorFlow* developers who want to get started with [OpenVINO™](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) in their inferencing applications. TensorFlow* developers can now take advantage of [OpenVINO™](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) toolkit optimizations with TensorFlow inference applications across a wide range of Intel® compute devices by adding just two lines of code.

    import openvino_tensorflow
    openvino_tensorflow.set_backend('<backend_name>')

This product delivers [OpenVINO™](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) inline optimizations which enhance inferencing performance with minimal code modifications.  **OpenVINO™ integration with TensorFlow accelerates** inference across many AI models on a variety of Intel<sup>®</sup> silicon such as:

- Intel<sup>®</sup> CPUs
- Intel<sup>®</sup> integrated and discrete GPUs

Note: Support for Intel Movidius™ MyriadX VPUs is no longer maintained. Consider previous releases for running on Myriad VPUs.

[Note: For maximum performance, efficiency, tooling customization, and hardware control, we recommend the developers to adopt native OpenVINO™ APIs and its runtime.]

*New: [OpenVINO™ TensorFlow FrontEnd](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/tensorflow) can be used as an alternative to deploy your models whenever a model is fully supported by OpenVINO, and you can move from TensorFlow APIs to Native OpenVINO APIs.*

## Installation
### Prerequisites

- Ubuntu 18.04, 20.04, macOS 11.2.3 or Windows<sup>1</sup> 10 - 64 bit
- Python* 3.7, 3.8 or 3.9
- TensorFlow* v2.9.3

<sup>1</sup>Windows package supports only Python3.9 

Check our [Interactive Installation Table](https://openvinotoolkit.github.io/openvino_tensorflow/) for a menu of installation options. The table will help you configure the installation process.

The **OpenVINO™ integration with TensorFlow** package comes with pre-built libraries of OpenVINO™ version 2022.3.0. The users do not have to install OpenVINO™ separately. This package supports:
- Intel<sup>®</sup> CPUs
- Intel<sup>®</sup> integrated and discrete GPUs


        pip3 install -U pip
        pip3 install tensorflow==2.9.3
        pip3 install openvino-tensorflow==2.3.0


For installation instructions on Windows please refer to [**OpenVINO™ integration with TensorFlow** for Windows ](docs/INSTALL.md#windows)

To use Intel<sup>®</sup> integrated GPUs for inference, make sure to install the [Intel® Graphics Compute Runtime for OpenCL™ drivers](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html#install-gpu)

For more details on installation please refer to [INSTALL.md](docs/INSTALL.md), and for build from source options please refer to [BUILD.md](docs/BUILD.md)

## Configuration

Once you've installed **OpenVINO™ integration with TensorFlow**, you can use TensorFlow* to run inference using a trained model.

To see if **OpenVINO™ integration with TensorFlow** is properly installed, run

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This should produce an output like:

        TensorFlow version:  2.9.3
        OpenVINO integration with TensorFlow version: b'2.3.0'
        OpenVINO version used for this build: b'2022.3.0'
        TensorFlow version used for this build: v2.9.3

        CXX11_ABI flag used for this build: 1

By default, Intel<sup>®</sup> CPU is used to run inference. However, you can change the default option to Intel<sup>®</sup> integrated or discrete GPUs (GPU, GPU.0, GPU.1 etc). Invoke the following function to change the hardware on which inferencing is done.

    openvino_tensorflow.set_backend('<backend_name>')

Supported backends include 'CPU', 'GPU', 'GPU_FP16'

To determine what processing units are available on your system for inference, use the following function:

    openvino_tensorflow.list_backends()
For further performance improvements, it is advised to set the environment variable `OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS=1`. For more API calls and environment variables, see [USAGE.md](docs/USAGE.md).


## Examples

To see what you can do with **OpenVINO™ integration with TensorFlow**, explore the demos located in the [examples](./examples) directory.  

## Docker Support
Dockerfiles for Ubuntu* 18.04, Ubuntu* 20.04, and TensorFlow* Serving are provided which can be used to build runtime Docker* images for **OpenVINO™ integration with TensorFlow** on CPU, GPU.
For more details see [docker readme](docker/README.md).

### Prebuilt Images

- [Ubuntu 18 runtime image on Docker* Hub](https://hub.docker.com/r/openvino/openvino_tensorflow_ubuntu18_runtime)
- [Ubuntu 20 runtime image on Docker* Hub](https://hub.docker.com/r/openvino/openvino_tensorflow_ubuntu20_runtime)
- [Azure* Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvinotensorflow)

## Try it on Intel<sup>®</sup> DevCloud
Sample tutorials are also hosted on [Intel<sup>®</sup> DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/build/ovtfoverview.html). The demo applications are implemented using Jupyter Notebooks. You can interactively execute them on Intel<sup>®</sup> DevCloud nodes, compare the results of **OpenVINO™ integration with TensorFlow**, native TensorFlow and OpenVINO™. 

## License
**OpenVINO™ integration with TensorFlow** is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## Support

Submit your questions, feature requests and bug reports via [GitHub issues](https://github.com/openvinotoolkit/openvino_tensorflow/issues).

## Troubleshooting

Some known issues and troubleshooting guide can be found [here](docs/TROUBLESHOOTING.md).

## How to Contribute

We welcome community contributions to **OpenVINO™ integration with TensorFlow**. If you have an idea for improvement:

* Share your proposal via [GitHub issues](https://github.com/openvinotoolkit/openvino_tensorflow/issues).
* Submit a [pull request](https://github.com/openvinotoolkit/openvino_tensorflow/pulls).

We will review your contribution as soon as possible. If any additional fixes or modifications are necessary, we will guide you and provide feedback. Before you make your contribution, make sure you can build **OpenVINO™ integration with TensorFlow** and run all the examples with your fix/patch. If you want to introduce a large feature, create test cases for your feature. Upon our verification of your pull request, we will merge it to the repository provided that the pull request has met the above mentioned requirements and proved acceptable.

---
\* Other names and brands may be claimed as the property of others.
