# Build Instructions

<!-- markdown-toc -->
 1. [Prerequisites](#Prerequisites)
	* 1.1. [Ubuntu](#Ubuntu)
	* 1.2. [macOS](#macOS)
 2. [OpenVINO™ integration with TensorFlow](#OpenVINOintegrationwithTensorFlow)
	* 2.1. [Build Instructions](#BuildInstructions)
	* 2.2. [Build Instructions for Intel Atom® Processor](#BuildInstructionsforIntelAtomProcessor)
	* 2.3. [ Build Verification](#BuildVerification)
 3. [OpenVINO™](#OpenVINO)
 4. [TensorFlow](#TensorFlow)
 5. [Build ManyLinux2014 compatible **OpenVINO™ integration with TensorFlow** wheels](#BuildManyLinux2014compatibleOpenVINOintegrationwithTensorFlowwheels)

<!-- markdown-toc-config
	numbering=true
	autoSave=true
	/markdown-toc-config -->
<!-- /markdown-toc -->

First, clone the `openvino_tensorflow` repository:

  ```bash
  $ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
  $ cd openvino_tensorflow
  $ git submodule init
  $ git submodule update --recursive
  ```

##  1. <a name='Prerequisites'></a>Prerequisites

###  1.1. <a name='Ubuntu'></a>Ubuntu

1. Install apt packages  

      ```bash  
      # users may need sudo access to install some of the dependencies  
      $ apt-get update
      $ apt-get install python3 python3-dev
      $ apt-get install -y --no-install-recommends ca-certificates autoconf automake build-essential \
      libtool unzip git unzip wget zlib1g zlib1g-dev bash-completion \
      build-essential cmake zip golang-go locate curl clang-format cpio libtinfo-dev jq \
      lsb-core libusb-1.0-0-dev patchelf
      ```  

      ```bash
      # install required gcc package (Optional; if gcc-7 is not installed)
      $ apt-get install gcc-7 g++-7  

      # if multiple gcc versions are installed then install alternatives
      $ update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 
      $ update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70

      # configure alternatives (select the required gcc version)
      $ update-alternatives --config gcc
      $ update-alternatives --config g++
      ```        

2. Install Python requirements  

      ```bash
      $ pip3 install -r requirements.txt
      ```  

2. Install CMake, supported version >=3.14.0 and =<3.20.0, 
        
      ```bash
      # to install CMake 3.18.2
      $ wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.tar.gz && \
      $ tar -xzvf cmake-3.18.4-Linux-x86_64.tar.gz && \
      $ cp cmake-3.18.4-Linux-x86_64/bin/* /usr/local/bin/ && \
        cp -r cmake-3.18.4-Linux-x86_64/share/cmake-3.18 /usr/local/share/
      ```

3. Install Bazelisk (Optional; required only while building TensorFlow from source)

      ```bash
      $ apt-get update && apt-get install -y openjdk-8-jdk
      $ curl -fsSL https://deb.nodesource.com/setup_12.x | bash -
      $ apt-get install -y nodejs
      $ npm install -g @bazel/bazelisk
      ```

###  1.2. <a name='macOS'></a>macOS

1. Install HomeBrew packages

      ``` bash
      $ brew install python@3.9
      $ brew install cmake autoconf automake libtool libusb wget 
      ```

2. Install Pip and requirements

      ```bash
      $ wget https://bootstrap.pypa.io/get-pip.py
      $ python3 get-pip.py
      $ pip3 install -r requirements.txt
      ```

3. Install Bazelisk v1.10.1 (Optional; required only while building TensorFlow from source)

      ```bash
      $ wget https://github.com/bazelbuild/bazelisk/releases/download/v1.10.1/bazelisk-darwin-amd64
      $ mv bazelisk-darwin-amd64 /usr/local/bin/bazel
      $ chmod 777 /usr/local/bin/bazel
      ```

4. Install Apple XCode Command Line Tools

      ```bash
      $ xcode-select --install
      ```

Notes:
        Developed and Tested on macOS version 11.2.3 with CMake version 3.21.1
        User can install Python 3.7, 3.8, or 3.9. 



##  2. <a name='OpenVINOintegrationwithTensorFlow'></a>OpenVINO™ integration with TensorFlow

###  2.1. <a name='BuildInstructions'></a>Build Instructions
Use one of the following build options based on the requirements. **OpenVINO™ integration with TensorFlow** built using PyPI TensorFlow enables only the Python APIs. TensorFlow C++ libraries built from source is required to use the C++ APIs.

1. Pulls compatible prebuilt TensorFlow package from PyPi, clones and builds OpenVINO™ from source. The arguments are optional. If any argument is not provided, then the default versions as specified in build_ovtf.py will be used. 

        python3 build_ovtf.py --tf_version=v2.7.0 --openvino_version=2021.4.2

2. Pulls compatible prebuilt TensorFlow package from Github release assets. Uses OpenVINO™ binary from specified location.

        python3 build_ovtf.py --use_openvino_from_location=/opt/intel/openvino_2021.4.752/ --cxx11_abi_version=1

3. Uses pre-built TensorFlow from the given location ([refer the TensorFlow build instructions](#tensorflow)). Pulls and builds OpenVINO™ from source. Use this if you need to build **OpenVINO™ integration with TensorFlow** frequently without building TensorFlow from source everytime.

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/

4. Uses prebuilt TensorFlow from the given location ([refer the TensorFlow build instructions](#tensorflow)). Uses OpenVINO™ binary from specified location. **This is only compatible with ABI1 built TensorFlow**.

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/  --use_openvino_from_location=/opt/intel/openvino_2021.4.752/ --cxx11_abi_version=1

5. Pulls and builds TensorFlow from Source. Uses OpenVINO™ binary from specified location.

        python3 build_ovtf.py --build_tf_from_source --use_openvino_from_location=/opt/intel/openvino_2021.4.752/ --cxx11_abi_version=1

6. Pulls and builds TensorFlow and OpenVINO™ from source

        python3 build_ovtf.py --build_tf_from_source


Select the `help` option of `build_ovtf.py` script to learn more about various build options.

        python3 build_ovtf.py --help

###  2.2. <a name='BuildInstructionsforIntelAtomProcessor'></a>Build Instructions for Intel Atom® Processor
In order to build **OpenVINO™ integration with TensorFlow** to use with the Intel Atom® processor, we recommend building TensorFlow from source, by using the following command:

        python3 build_ovtf.py --build_tf_from_source --cxx11_abi_version=1 --target_arch silvermont

###  2.3. <a name='BuildVerification'></a> Build Verification
When the build is finished, a new `virtualenv` directory with name `venv-tf-py3` is created in `build_cmake`. Build artifacts (e.g. the `openvino_tensorflow-<VERSION>-cp38-cp38-manylinux2014_x86_64.whl`) are created in the `build_cmake/artifacts/` directory.

Activate the following `virtualenv` to start using **OpenVINO™ integration with TensorFlow**.

        source build_cmake/venv-tf-py3/bin/activate

Alternatively, you may install the TensorFlow and **OpenVINO™ integration with TensorFlow** outside of a `virtualenv`. The Python `whl` files are located in the `build_cmake/artifacts/` and `build_cmake/artifacts/tensorflow` directories, respectively.

Verify that `openvino-tensorflow` is installed correctly:

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This should produce an output like:

        TensorFlow version:  2.7.0
        OpenVINO integration with TensorFlow version: b'1.1.0'
        OpenVINO version used for this build: b'2021.4.2'
        TensorFlow version used for this build: v2.7.0
        CXX11_ABI flag used for this build: 1


Test the installation:

        python3 test_ovtf.py

This command runs all C++ and Python unit tests from the `openvino_tensorflow` source tree. It also runs various TensorFlow Python tests using OpenVINO™.
  
##  3. <a name='OpenVINO'></a>OpenVINO™

OpenVINO™ can be built from source independently using `build_ov.py`
##  4. <a name='TensorFlow'></a>TensorFlow

TensorFlow can be built from source using `build_tf.py`. The build artifacts can be found under ${PATH_TO_TF_BUILD}/artifacts/

- Set your build path

        export PATH_TO_TF_BUILD=/path/to/tensorflow/build/

- For all available build options

        python3 build_tf.py -h

- Builds TensorFlow with CXX11_ABI=0.

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --cxx11_abi_version=0

- Builds TensorFlow with CXX11_ABI=1

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --cxx11_abi_version=1

- To build with a desired TensorFlow version, for example: v2.7.0

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.7.0

##  5. <a name='BuildManyLinux2014compatibleOpenVINOintegrationwithTensorFlowwheels'></a>Build ManyLinux2014 compatible **OpenVINO™ integration with TensorFlow** wheels

To build wheel files compatible with manylinux2014, use the following commands. The build artifacts will be available in your container's /whl/ folder.

```bash
cd tools/builds/
docker build --no-cache -t openvino_tensorflow/pip --build-arg OVTF_BRANCH=releases/v1.1.0 . -f Dockerfile.manylinux2014
```
