<p align="center">
  <img src="images/openvino.png">
</p>

# Intel<sup>®</sup> OpenVINO™ integration with TensorFlow

This repository contains the source code of Intel<sup>®</sup> OpenVINO™ integration with TensorFlow, a product needed to enable Intel<sup>®</sup> OpenVINO™ runtime and optimizations for TensorFlow. Intel<sup>®</sup> OpenVINO™ integration with TensorFlow enables acceleration of AI inferencing across almost all imaginable use cases, using a variety of AI models, on a variety of Intel silicon such as Intel CPUs, Intel integrated GPUs, Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs) and Intel<sup>®</sup> Vision accelerator Design with 8 Intel Movidius<sup>TM</sup> MyriadX VPUs - referred as VAD-M or HDDL.

## Installation
For installation table, please visit: [Link](https://openvinotoolkit.github.io/openvino_tensorflow/)
### Requirements

**Check version numbers**
|Using pre-built packages| Building from source|
| -----------------------|-------------------|
|Python 3.6, 3.7, or 3.8| Python 3.6, 3.7, or 3.8|
|TensorFlow v2.4.1|GCC 7.5 (Ubuntu 18.04)|
|        |cmake >= 3.14|
|        |Bazelisk v1.7.5|
|        |virtualenv 16.0.0+|
|        |patchelf 0.9|

### Use Pre-Built Packages

Intel<sup>®</sup> OpenVINO™ integration with TensorFlow has two releases: one built with CXX11_ABI=0 and another built with CXX11_ABI=1. Since TensorFlow packages available in PyPi are built with CXX11_ABI=0 and OpenVINO release packages are built with CXX11_ABI=1, binary releases of these packages cannot be installed together and used as is. Based on your needs, you can choose one of the two available methods:

- Use Intel<sup>®</sup> OpenVINO™ integration with TensorFlow alongside PyPi TensorFlow
  (CXX11_ABI=0, no OpenVINO installation required, disables VAD-M support)
  
- Use Intel<sup>®</sup> OpenVINO™ integration with TensorFlow alongside the Intel® Distribution of OpenVINO™ Toolkit
  (CXX11_ABI=1, needs a custom TensorFlow package, enables VAD-M support)

#### - Use Intel<sup>®</sup> OpenVINO™ integration with TensorFlow alongside PyPi TensorFlow

This OpenVINO integration with TensorFlow package comes with pre-built libraries of OpenVINO version 2021.3. The users do not have to install OpenVINO separately. This package supports Intel CPUs, Intel integrated GPUs and Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs).

1. Ensure the following pip version is being used:

        pip install --upgrade pip==21.0.1

2. Install `TensorFlow`:

        pip install -U tensorflow==2.4.1

3. Install `openvino-tensorflow`:

        pip install openvino-tensorflow

#### - Use Intel<sup>®</sup> OpenVINO™ integration with TensorFlow alongside the Intel® Distribution of OpenVINO™ Toolkit

This OpenVINO integration with TensorFlow package is currently compatible with OpenVINO version 2021.3. This package supports Intel CPUs, Intel integrated GPUs, Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs) and Intel<sup>®</sup> Vision Accelerator Design with Movidius<sup>TM</sup> (VAD-M). 

Users can build TensorFlow from source with -D_GLIBCXX_USE_CXX11_ABI=1 or they can use the TensorFlow package that we provide below.

1. Ensure the following pip version is being used:

        pip install --upgrade pip==21.0.1

2. Install `TensorFlow`. Based on your Python version, use the appropriate package below:

        pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.4.1-cp36-cp36m-manylinux2010_x86_64.whl

        or

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.4.1-cp37-cp37m-manylinux2010_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.4.1-cp38-cp38-manylinux2010_x86_64.whl

3. Download & install Intel® Distribution of OpenVINO™ Toolkit 2021.3 release along with its dependencies from ([https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)).

4. Initialize the OpenVINO environment by running the `setupvars.sh` present in <code>\<openvino\_install\_directory\>\/bin</code> using the below command:

        source setupvars.sh

3. Install `openvino-tensorflow`. Based on your Python version, use the appropriate package below:

        pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp36-cp36m-manylinux2014_x86_64.whl

        or

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp37-cp37m-manylinux2014_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp38-cp38-manylinux2014_x86_64.whl

#### - Summary of the Prebuilt packages



|TensorFlow Package| OpenVINO integration with TensorFlow Package|Supported OpenVINO Flavor|Supported Hardware Backends|Comments|
| -----------------|-----------------------------------|----------------------------|---------------------------|----------------|
|tensorflow| openvino-tensorflow|OpenVINO built from source|CPU,GPU,MYRIAD|OpenVINO libraries are built from source and included in the wheel package|
|tensorflow-abi1| openvino-tensorflow-abi1|Links to OpenVINO binary release|CPU,GPU,MYRIAD,VAD-M|OpenVINO integration with TensorFlow libraries links to OpenVINO binaries|


### Build from source

Once TensorFlow and its dependencies are installed, clone the `openvino_tensorflow` repo:

        git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
        cd openvino_tensorflow
        git submodule init
        git submodule update --recursive

Running the following Python script will build OpenVINO integration with TensorFlow with PyPi TensorFlow, and OpenVINO sources

        python3 build_ovtf.py

When the build finishes, a new `virtualenv` directory is created in `build_cmake/venv-tf-py3`. Build artifacts (ex: the `openvino_tensorflow-<VERSION>-cp36-cp36m-manylinux2014_x86_64.whl`) are created in the `build_cmake/artifacts/` directory. 

Select the help option of `build_ovtf.py` script to learn more about various build options. 
        
        python3 build_ovtf.py --help

To use `openvino-tensorflow`, activate the following `virtualenv` to start using OpenVINO integration with TensorFlow. 

        source build_cmake/venv-tf-py3/bin/activate
 
Alternatively, you can also install the TensorFlow and OpenVINO integration with TensorFlow outside of a `virtualenv`. The Python `whl` files are located in the `build_cmake/artifacts/` and `build_cmake/artifacts/tensorflow` directories, respectively.

Verify that `openvino-tensorflow` is installed correctly:

    python -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This will produce something like this:

        TensorFlow version:  2.4.1
        OpenVINO integration with TensorFlow version: b'0.5.0'
        OpenVINO version used for this build: b'2021.3'
        TensorFlow version used for this build: v2.4.1
        CXX11_ABI flag used for this build: 1
        OpenVINO integration with TensorFlow built with Grappler: False

Note: The version of the openvino-tensorflow is not going to be exactly 
the same as when you build from source. This is due to delay in the source 
release and publishing the corresponding Python wheel.

Test the installation:

        python3 test_ovtf.py

This command runs all C++ and Python unit tests from the `openvino_tensorflow` source tree. It also runs various TensorFlow Python tests using OpenVINO.

For more advanced build configurations, please refer to: ([OpenVINO integration with TensorFlow - Builds](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/README.md))

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

