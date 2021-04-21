<p align="center">
  <img src="images/openvino.png">
</p>

# **OpenVINO™ integration with TensorFlow**

This repository contains the source code of **OpenVINO™ integration with TensorFlow**, a product needed to enable OpenVINO™ runtime and optimizations for TensorFlow. **OpenVINO™ integration with TensorFlow** enables acceleration of AI inferencing across many AI models on a variety of Intel<sup>®</sup> silicon such as:
- Intel<sup>®</sup> CPUs 
- Intel<sup>®</sup> integrated GPUs
- Intel<sup>®</sup> Movidius™ Vision Processing Units - referred as VPU 
- Intel<sup>®</sup> Vision accelerator Design with 8 Intel Movidius™ MyriadX VPUs - referred as VAD-M or HDDL
  
## Installation
### Prerequisites

- Ubuntu 18.04
- Python 3.6, 3.7, or 3.8
- TensorFlow v2.4.1

Check our [__Interactive Installation Table__](https://openvinotoolkit.github.io/openvino_tensorflow/) for a menu of installation options. The table will help you configure the installation process.

### Install **OpenVINO™ integration with TensorFlow** alongside PyPi TensorFlow

This **OpenVINO™ integration with TensorFlow** package comes with pre-built libraries of OpenVINO™ version 2021.3. You don't have to install OpenVINO™ separately. This package supports: 
- Intel<sup>®</sup> CPUs
- Intel<sup>®</sup> integrated GPUs
- Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs)


        pip3 install -U pip==21.0.1
        pip3 install -U tensorflow==2.4.1
        pip3 install openvino-tensorflow


If you want to leverage Intel® Vision Accelerator Design with Movidius™ (VAD-M) for inference, install **OpenVINO™ integration with TensorFlow** alongside the Intel® Distribution of OpenVINO™ Toolkit. 

For the installation process, go to [BUILD.md](docs/BUILD.md). 


## Usage

Once you have installed **OpenVINO™ integration with TensorFlow**, you can use TensorFlow to run inference using a trained model.
The only change required to a script is adding

    import openvino_tensorflow

By default, CPU backend is enabled. You can substitute the default CPU backend with a different backend by using the following API:

    openvino_tensorflow.set_backend('backend_name')

To determine what backends are available on your system, use the following API:

    openvino_tensorflow.list_backends()

More detailed examples on how to use **OpenVINO™ integration with TensorFlow** are located in the [**examples**](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples) directory.

## License
**OpenVINO™ integration with TensorFlow** is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.  

## Support

Submit your questions, feature requests and bug reports via [**GitHub issues**](https://github.com/openvinotoolkit/openvino_tensorflow/issues).

## How to Contribute

We welcome community contributions to **OpenVINO™ integration with TensorFlow**. If you have an idea for improvement:

* Share your proposal via [**GitHub issues**](https://github.com/openvinotoolkit/openvino_tensorflow/issues).
* Submit a [**pull request**](https://github.com/openvinotoolkit/openvino_tensorflow/pulls).

We will review your contribution. If any additional fixes or modifications are necessary, we will guide you and provide you with feedback. Before you make your contribution, make sure you can build **OpenVINO™ integration with TensorFlow** and run all the examples with your patch (**what is this patch?**). If you you want to introduce a large feature, test it. Upon our verification of your pull request, we will merge it to the repository provided that the PR has met our approval.  

You can find all the guidelines for contributing to the OpenVINO repositories [here](https://github.com/openvinotoolkit/openvino/wiki/Contribute)
