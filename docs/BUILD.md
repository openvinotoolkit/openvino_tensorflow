# Installation and Build Options
## Prerequisites

|Build Type| Requirements|
|:-----------------------|-------------------|
|Use pre-built packages| Python 3.6, 3.7, or 3.8, TensorFlow v2.5.0|
|Build from source| Python 3.6, 3.7, or 3.8, GCC 7.5 (Ubuntu 18.04, 20.04),  cmake 3.14 or higher, Bazelisk v1.7.5, virtualenv 16.0.0 or higher, patchelf 0.9, libusb 1.0.0|

## Use Pre-Built Packages

**OpenVINO™ integration with TensorFlow** has two releases: one built with CXX11_ABI=0 and another built with CXX11_ABI=1.

Since TensorFlow packages available in [PyPi](https://pypi.org) are built with CXX11_ABI=0 and OpenVINO™ release packages are built with CXX11_ABI=1, binary releases of these packages **cannot be installed together**. Based on your needs, you can choose one of the two available methods:

- **OpenVINO™ integration with TensorFlow** alongside PyPi TensorFlow
  (CXX11_ABI=0, no OpenVINO™ installation required, no VAD-M support)

- **OpenVINO™ integration with TensorFlow** alongside the Intel® Distribution of OpenVINO™ Toolkit
  (CXX11_ABI=1, needs a custom TensorFlow package, enables VAD-M support)

### Install **OpenVINO™ integration with TensorFlow** alongside PyPi TensorFlow

This **OpenVINO™ integration with TensorFlow** package includes pre-built libraries of OpenVINO™ version 2021.3. The users do not have to install OpenVINO™ separately. This package supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, and Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs).


        pip3 install -U pip==21.0.1
        pip3 install -U tensorflow==2.5.0
        pip3 install openvino-tensorflow

### Install **OpenVINO™ integration with TensorFlow** alongside the Intel® Distribution of OpenVINO™ Toolkit

This **OpenVINO™ integration with TensorFlow** package is currently compatible with OpenVINO™ version 2021.3. This package supports Intel<sup>®</sup> CPUs, Intel<sup>®</sup> integrated GPUs, Intel<sup>®</sup> Movidius™ Vision Processing Units (VPUs), and Intel<sup>®</sup> Vision Accelerator Design with Movidius™ (VAD-M).

You can build TensorFlow from source with -D_GLIBCXX_USE_CXX11_ABI=1 or use the following TensorFlow package:

1. Ensure the following versions are being used for pip and numpy:

        pip3 install --upgrade pip==21.0.1
        pip3 install numpy==1.20.2

2. Install `TensorFlow`. Based on your Python version, use the appropriate package below:

        pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp36-cp36m-manylinux2010_x86_64.whl

        or

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp37-cp37m-manylinux2010_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.5.0-cp38-cp38-manylinux2010_x86_64.whl

3. Download & install Intel® Distribution of OpenVINO™ Toolkit 2021.3 release along with its dependencies from ([https://software.intel.com/en-us/openvino-toolkit/download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html)).

4. Initialize the OpenVINO™ environment by running the `setupvars.sh` located in <code>\<openvino\_install\_directory\>\/bin</code> using the command below:

        source setupvars.sh

5. Install `openvino-tensorflow`. Based on your Python version, choose the appropriate package below:

        pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp36-cp36m-linux_x86_64.whl

        or

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp37-cp37m-linux_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp38-cp38-linux_x86_64.whl



### Prebuilt packages summary

|TensorFlow Package| **OpenVINO™ integration with TensorFlow** Package|Supported OpenVINO™ Flavor|Supported Hardware Backends|Comments|
| -----------------|-----------------------------------|----------------------------|---------------------------|----------------|
|tensorflow| openvino-tensorflow|OpenVINO™ built from source|CPU,GPU,MYRIAD|OpenVINO™ libraries are built from source and included in the wheel package|
|tensorflow-abi1| openvino-tensorflow-abi1|Dynamically links to OpenVINO™ binary release|CPU,GPU,MYRIAD,VAD-M|**OpenVINO™ integration with TensorFlow** libraries are dynamically linked to OpenVINO™ binaries|


## Build From Source
Clone the `openvino_tensorflow` repository:

```bash
$ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
$ cd openvino_tensorflow
$ git submodule init
$ git submodule update --recursive
```

Install the following python packages

```bash
$ pip3 install -U psutil==5.8.0 wheel==0.36.2
```

### **OpenVINO™ integration with TensorFlow**
Use one of the following build options based on the requirements

1. Pulls compatible prebuilt TF package from PyPi, clones and builds OpenVINO™ from source.

        python3 build_ovtf.py

2. Pulls compatible prebuilt TF package from PyPi. Uses OpenVINO™ binary.

        python3 build_ovtf.py --use_openvino_from_location=/opt/intel/openvino_2021.3.394/ --cxx11_abi_version=1


3. Pulls and builds TF and OpenVINO™ from source

        python3 build_ovtf.py --build_tf_from_source

4. Pulls and builds TF from Source. Uses OpenVINO™ binary.

        python3 build_ovtf.py --build_tf_from_source --use_openvino_from_location=/opt/intel/openvino_2021.3.394/ --cxx11_abi_version=1

5. Uses pre-built TF from the given location ([refer the TensorFlow build instructions](#tensorflow)). Pulls and builds OpenVINO™ from source. Use this if you need to build **OpenVINO™ integration with TensorFlow** frequently without building TF from source everytime.

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/

6. Uses prebuilt TF from the given location ([refer the TensorFlow build instructions](#tensorflow)). Uses OpenVINO™ binary. **This is only compatible with ABI1 built TF**.

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/  --use_openvino_from_location=/opt/intel/openvino_2021/ --cxx11_abi_version=1

Select the `help` option of `build_ovtf.py` script to learn more about various build options.

        python3 build_ovtf.py --help

#### Verification
When the build is finished, a new `virtualenv` directory is created in `build_cmake/venv-tf-py3`. Build artifacts (ex: the `openvino_tensorflow-<VERSION>-cp36-cp36m-manylinux2014_x86_64.whl`) are created in the `build_cmake/artifacts/` directory.

Activate the following `virtualenv` to start using **OpenVINO™ integration with TensorFlow**.

        source build_cmake/venv-tf-py3/bin/activate

Alternatively, you may install the TensorFlow and **OpenVINO™ integration with TensorFlow** outside of a `virtualenv`. The Python `whl` files are located in the `build_cmake/artifacts/` and `build_cmake/artifacts/tensorflow` directories, respectively.

Verify that `openvino-tensorflow` is installed correctly:

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This should produce an output like:

        TensorFlow version:  2.5.0
        OpenVINO integration with TensorFlow version: b'0.5.0'
        OpenVINO version used for this build: b'2021.3'
        TensorFlow version used for this build: v2.5.0
        CXX11_ABI flag used for this build: 1
        OpenVINO integration with TensorFlow built with Grappler: False


Test the installation:

        python3 test_ovtf.py

This command runs all C++ and Python unit tests from the `openvino_tensorflow` source tree. It also runs various TensorFlow Python tests using OpenVINO™.

### Build Instructions for Intel Atom® Processor
In order to build **OpenVINO™ integration with TensorFlow** to use on Intel Atom® processor, we recommend building TF from source. The command below will build TF and OpenVINO™ from source for Intel Atom® processors.

        python3 build_ovtf.py --build_tf_from_source --cxx11_abi_version=1 --target_arch silvermont

## TensorFlow

TensorFlow can be built from source using `build_tf.py`. The build artifacts can be found under ${PATH_TO_TF_BUILD}/artifacts/

- Set your build path

        export PATH_TO_TF_BUILD=/path/to/tensorflow/build/

- For all available build options

        python3 build_tf.py -h

- Builds TF with CXX11_ABI=0.

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --cxx11_abi_version=0

- Builds TF with CXX11_ABI=1

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --cxx11_abi_version=1

- To build with a desired TF version, for example: v2.5.0

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.5.0

## OpenVINO™

OpenVINO™ can be built from source independently using `build_ov.py`

## Build ManyLinux2014 compatible **OpenVINO™ integration with TensorFlow** wheels

To build wheel files compatible with manylinux2014, use the following commands. The build artifacts will be available in your container's /whl/ folder.

```bash
cd tools/builds/
docker build --no-cache -t openvino_tensorflow/pip --build-arg OVTF_BRANCH=releases/v0.5.0 . -f Dockerfile.manylinux2014
```
