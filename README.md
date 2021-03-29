<p align="center">
  <img src="images/openvino.png">
</p>

# Intel<sup>®</sup> OpenVINO™ integration with TensorFlow

This repository contains the source code of Intel<sup>®</sup> OpenVINO™ integration with TensorFlow, a product needed to enable Intel<sup>®</sup> OpenVINO™ runtime and optimizations for TensorFlow. Intel<sup>®</sup> OpenVINO™ integration with TensorFlow enables acceleration of AI inferencing across almost all imaginable use cases, using a variety of AI models, on a variety of Intel silicon such as Intel CPUs, Intel integrated GPUs and Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs) and Intel<sup>®</sup> Vision accelerator Design with 8 Intel Movidius<sup>TM</sup> MyriadX VPUs - referred as VAD-M or HDDL.

## Installation

### Requirements

**Check version numbers**
|Using pre-built packages| Building from source|
| -----------------------|-------------------|
|Python 3.6| Python 3.6|
|TensorFlow v2.2.2|GCC 7.5 (Ubuntu 18.04)|
|        |cmake 3.15 or higher|
|        |Bazelisk|
|        |virtualenv 16.0.0+|
|        |patchelf|

### Use pre-built packages

Intel<sup>®</sup> OpenVINO™ integration with TensorFlow has two releases: one built with CXX11_ABI=0 and another built with CXX11_ABI=1. TensorFlow packages available in PyPi are built with CXX11_ABI=0 and OpenVINO release packages are built with CXX11_ABI=1. To prevent ABI incompatibilities, we provide both packages built with CXX11_ABI=0 and CXX11_ABI=1. 

#### - Package built with CXX11_ABI=0

Intel<sup>®</sup> OpenVINO™ integration with TensorFlow package built with ABI=0 is compatible with PyPi TensorFlow package 2.2.2. This OpenVINO integration with TensorFlow package comes with prebuilt libraries of OpenVINO version 2021.2. The users do not have to install OpenVINO separately. This package supports Intel CPUs, Intel integrated GPUs and Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs).

Users can use TensorFlow 2.2.2 from PyPi (`pip install -U tensorflow==2.2.2`). However, TensorFlow 2.2.2 package from PyPi does not have the latest security patches. We provide a ready-to-use TensorFlow package built with security patches using CXX11_ABI=0 and recommend users to use it to avoid any security issues.


Below are the steps needed to use the packages built with CXX11_ABI=0

1. Ensure the following pip version is being used:

        pip install --upgrade pip==21.0.1

2. Install `TensorFlow`:

        pip install -U tensorflow==2.2.2
      
                    (or)

        pip install tensorflow_security_patched-2.2.2-cp36-cp36m-linux_x86_64.whl (Recommended for security patches)

    Download ([**tensorflow_security_patched-2.2.2-cp36-cp36m-linux_x86_64.whl**](https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_security_patched-2.2.2-cp36-cp36m-linux_x86_64.whl))

3. Install `openvino-tensorflow`:

        pip install openvino_tensorflow-0.5.0-py2.py3-none-manylinux2010_x86_64.whl

    Download ([**openvino_tensorflow-0.5.0-py2.py3-none-manylinux2010_x86_64.whl**](https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow-0.5.0-py2.py3-none-manylinux2010_x86_64.whl))
#### - Package built with CXX11_ABI=1

Intel<sup>®</sup> OpenVINO™ integration with TensorFlow package built with ABI=1 is compatible with OpenVINO binary releases. This OpenVINO integration with TensorFlow package is currently compatible with OpenVINO version 2021.2. This package supports Intel CPUs, Intel integrated GPUs, Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs) and Intel<sup>®</sup> Vision Accelerator Design with Movidius<sup>TM</sup> (VAD-M). 

Users can build TensorFlow from source with CXX11_ABI=1 or they can use the TensorFlow package that we provide. We provide a ready-to-use TensorFlow package built with security patches using CXX11_ABI=1 and recommend users to use it to avoid any security issues.


Below are the steps needed to use the packages built with CXX11_ABI=1

1. Ensure the following pip version is being used:

        pip install --upgrade pip==21.0.1

2. Install `TensorFlow`:

        pip3 install tensorflow_security_patched_abi1-2.2.2-cp36-cp36m-linux_x86_64.whl

    Download([**tensorflow_security_patched_abi1-2.2.2-cp36-cp36m-linux_x86_64.whl**](https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_security_patched_abi1-2.2.2-cp36-cp36m-linux_x86_64.whl))

3. Install the OpenVINO 2021.2 release along with its dependencies from ([https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)).

4. Initialize the OpenVINO environment by running the `setupvars.sh` present in <code>\<openvino\_install\_directory\>\/bin</code> using the below command:

        source setupvars.sh

3. Install `openvino-tensorflow`:

        pip install openvino_tensorflow_abi1-0.5.0-py2.py3-none-manylinux2010_x86_64.whl

    Download ([**openvino_tensorflow_abi1-0.5.0-py2.py3-none-manylinux2010_x86_64.whl**](https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-py2.py3-none-manylinux2010_x86_64.whl))

#### - Summary of the Prebuilt packages



|TensorFlow Package| OpenVINO integration with TensorFlow Package|Supported OpenVINO Flavor|Supported Hardware Backends|Comments|
| -----------------|-----------------------------------|----------------------------|---------------------------|----------------|
|tensorflow-security-patched| openvino-tensorflow|OpenVINO built from source|CPU,GPU,MYRIAD|OpenVINO libraries are built from source and included in the wheel package|
|tensorflow-security-patched-abi1| openvino-tensorflow-abi1|Links to OpenVINO binary release|CPU,GPU,MYRIAD,HDDL|OpenVINO integration with TensorFlow libraries links to OpenVINO binaries|


### Build from source

Once TensorFlow and its dependencies are installed, clone the `openvino_tensorflow` repo:

        git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
        cd openvino_tensorflow
        git submodule init
        git submodule update --recursive

Run the following Python script to build TensorFlow, OpenVINO, and the OpenVINO integration with TensorFlow. Use Python 3

        python3 build_ovtf.py

When the build finishes, a new `virtualenv` directory is created in `build_cmake/venv-tf-py3`. Build artifacts (i.e., the `openvino_tensorflow-<VERSION>-py2.py3-none-manylinux1_x86_64.whl`) are created in the `build_cmake/artifacts` directory. 

For more build options:
        
        python3 build_ovtf.py --help

To use `openvino-tensorflow`, activate the following `virtualenv` to start using OpenVINO with TensorFlow. 

        source build_cmake/venv-tf-py3/bin/activate
 
Alternatively, you can also install the TensorFlow and OpenVINO integration with TensorFlow outside of a `virtualenv`. The Python `whl` files are located in the `build_cmake/artifacts/` and `build_cmake/artifacts/tensorflow` directories, respectively.

Select the help option of `build_ovtf.py` script to learn more about various build options. 

Verify that `openvino-tensorflow` installed correctly:

    python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This will produce something like this:

        TensorFlow version:  2.2.2
        openvino_tensorflow version: b'0.5.0'
        OpenVINO version used for this build: b'2021.2'
        TensorFlow version used for this build: v2.2.2
        CXX11_ABI flag used for this build: 1
        openvino_tensorflow built with Grappler: False

Note: The version of the openvino-tensorflow is not going to be exactly 
the same as when you build from source. This is due to delay in the source 
release and publishing the corresponding Python wheel.

Test the installation:

        python3 test_ovtf.py

This command runs all C++ and Python unit tests from the `openvino_tensorflow` source tree. It also runs various TensorFlow Python tests using OpenVINO.

## Usage

Once you have installed Intel<sup>®</sup> OpenVINO™ integration with TensorFlow, you can use TensorFlow to run inference using a trained model.
The only change required to a script is adding

    import openvino_tensorflow

By default, CPU backend is enabled. You can substitute the default CPU backend with a different backend by using the following API:

    openvino_tensorflow.set_backend('backend_name')

To determine what backends are available on your system, use the following API:

    openvino_tensorflow.list_backends()

More detailed examples on how to use OpenVINO integration with TensorFlow are located in the [**examples**](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples) directory.

## Support

Please submit your questions, feature requests and bug reports via [**GitHub issues**](https://github.com/openvinotoolkit/openvino_tensorflow/issues).

## How to Contribute

We welcome community contributions to Intel<sup>®</sup> OpenVINO™ integration with TensorFlow. If you have an idea for how to 
improve it:

* Share your proposal via [**GitHub issues**](https://github.com/openvinotoolkit/openvino_tensorflow/issues).
* Ensure you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [**pull request**](https://github.com/openvinotoolkit/openvino_tensorflow/pulls).
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged to the repository.

