<p>English | <a href="./BUILD_cn.md">简体中文</a></p>

# Build Instructions

<!-- markdown-toc -->
 1. [Prerequisites](#Prerequisites)
	* 1.1. [Ubuntu](#Ubuntu)
	* 1.2. [macOS](#macOS)
	* 1.3. [Windows](#Windows)
 2. [OpenVINO™ integration with TensorFlow](#OpenVINOintegrationwithTensorFlow)
	* 2.1. [Build Instructions for Linux](#BuildInstructionsLinux)
	* 2.2. [Build Instructions for macOS](#BuildInstructionsmacOS)
	* 2.3. [Build Instructions for Windows](#BuildInstructionsWindows)
	* 2.4. [Build Instructions for Intel Atom® Processor](#BuildInstructionsforIntelAtomProcessor)
	* 2.5. [Build Verification](#BuildVerification)
 3. [Backwards Compatibility with TensorFlow](#BackwardsCompatibilitywithTensorFlow)
 4. [OpenVINO™](#OpenVINO)
 5. [TensorFlow](#TensorFlow)
 

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

      Install gcc-7 for Ubuntu-18.04 and in similar way gcc-9 for Ubuntu-20.04
      ```bash
      # install required gcc package (Optional; if gcc-7 is not installed for Ubuntu-18.04)
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

2. Install CMake, supported version >=3.14.0 
        
      ```bash
      # to install CMake 3.18.4
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

###  1.3. <a name='Windows'></a>Windows
1. Install [Python3.9](https://www.python.org/downloads/)

2. Install Visual C++ Build Tools 2019   
   Visual C++ build tools 2019 comes with Visual Studio 2019 but can be installed separately:
   - Go to the Visual Studio [downloads](https://visualstudio.microsoft.com/downloads/),  
   - Select Redistributables and Build Tools,
   - Download and install:  
        Microsoft Visual C++ 2019 Redistributable  
        Microsoft Build Tools 2019

3. Install [CMake](https://cmake.org/download/), supported version >=3.14.0
4. [Download and Install OpenVINO™ Toolkit 2022.1.0 LTS for Windows](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit-download.html)

<br>

##  2. <a name='OpenVINOintegrationwithTensorFlow'></a>OpenVINO™ integration with TensorFlow

###  2.1. <a name='BuildInstructionsLinux'></a>Build Instructions for **Linux**
**OpenVINO™ integration with TensorFlow** built using PyPI TensorFlow enables only the Python APIs. To enable C++ APIs, build **OpenVINO™ integration with TensorFlow** with prebuilt TensorFlow C++ libraries. For VAD-M support, build with [OpenVINO™](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html) binary and initialize the OpenVINO™ environment by running the `setupvars.sh`.  
Use one of the following common build options based on the requirements. 

1. Pulls compatible TensorFlow package from PyPi, uses OpenVINO™ binary from specified location. This is the recommended build option for most of the use cases.

        python3 build_ovtf.py --tf_version=v2.9.2 --use_openvino_from_location=/opt/intel/openvino_2022.1.0/
  
2. Pulls compatible TensorFlow package from PyPi, clones and builds OpenVINO™ from source. Make sure that the options provided are CXX11_ABI compatible. As TensorFlow is compiled with CXX11_ABI=0 for versions earlier than 2.9.0, which is incompatible with OpenVINO™ binary released package. Choose this option primarily to build **OpenVINO™ integration with TensorFlow** for TensorFlow version earlier than 2.9.0 or when OpenVINO™ binary package is not installed. If not passed the default value of "--cxx11_abi_version" argument is 1.

        python3 build_ovtf.py --tf_version=v2.8.2 --openvino_version=2022.1.0 --cxx11_abi_version=0
  
- Support for OpenVINO™ versions older than 2022.1.0 has been deprecated from the last release (version 2.0.0) of **OpenVINO™ integration with TensorFlow**. Checkout the respective release branch of **OpenVINO™ integration with TensorFlow** to use previous versions of OpenVINO™ with it
- To use the master branch of OpenVINO™, pass "master" as value to "openvino_version" argument. **OpenVINO™ integration with TensorFlow** build script will take the latest commit from OpenVINO™ master branch, however all the openvino-tensorflow APIs and functionalities would still remain same as they were for OpenVINO™ version 2022.1.0 

3. Uses prebuilt TensorFlow from the given location ([refer the TensorFlow build instructions](#LinuxAndmacOS)). Uses OpenVINO™ binary from specified location. **This is only compatible with ABI1 built TensorFlow**.

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/ --use_openvino_from_location=/opt/intel/openvino_2022.1.0/ --cxx11_abi_version=1

4. Uses prebuilt TensorFlow from the given location ([refer the TensorFlow build instructions](#LinuxAndmacOS)). Pulls and builds OpenVINO™ from source. Use this option if you need to build **OpenVINO™ integration with TensorFlow** with TensorFlow built with CXX11_ABI=0 flag.

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/ --openvino_version=2022.1.0 --cxx11_abi_version=0


Use the `help` option of `build_ovtf.py` script to learn more about various build options.

        python3 build_ovtf.py --help

###  2.2. <a name='BuildInstructionsmacOS'></a>Build Instructions for **macOS**

Pulls compatible TensorFlow package from PyPi, clones and builds OpenVINO™ from source. Currently only ABI0 is supported on macOS.

        python3 build_ovtf.py --tf_version=v2.9.2 --openvino_version=2022.1.0 --cxx11_abi_version=0

###  2.3. <a name='BuildInstructionsWindows'></a>Build Instructions for **Windows**
Use "Command Prompt" or "x64 Native Tools Command Prompt for VS 2019" as administrator, and follow below options to build from source:
1. Pulls prebuilt TensorFlow 2.9.2 python package from Github release assets. Uses OpenVINO™ binary from specified location.

        python build_ovtf.py --tf_version=v2.9.2 --use_openvino_from_location="C:\Program Files (x86)\Intel\openvino_2022.1.0" 

2. To use prebuilt TensorFlow from the given location [refer the TensorFlow build instructions](#TFWindows). It uses OpenVINO™ binary from specified location. Use this build option to run C++ examples and to integrate openvino-tensorflow in TensorFlow C++ inference applications.

        python build_ovtf.py --use_openvino_from_location="C:\Program Files (x86)\Intel\openvino_2022.1.0" --use_tensorflow_from_location="\path\to\directory\containing\tensorflow\"

###  2.4. <a name='BuildInstructionsforIntelAtomProcessor'></a>Build Instructions for Intel Atom® Processor
In order to build **OpenVINO™ integration with TensorFlow** to use with the Intel Atom® processor, we recommend building TensorFlow from source, and then build openvino-tensorflow:

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.9.2 --target_arch silvermont
        python3 build_ovtf.py --use_tensorflow_from_location=${PATH_TO_TF_BUILD} --openvino_version=2022.1.0 --target_arch silvermont

###  2.5. <a name='BuildVerification'></a> Build Verification
When the build is finished, a new `virtualenv` directory with name `venv-tf-py3` is created in `build_cmake`. Build artifacts (e.g. the `openvino_tensorflow-<VERSION>-cp38-cp38-manylinux2014_x86_64.whl`) are created in the `build_cmake/artifacts/` directory.

Activate the following `virtualenv` to start using **OpenVINO™ integration with TensorFlow**.

        source build_cmake/venv-tf-py3/bin/activate

Alternatively, you may install the TensorFlow and **OpenVINO™ integration with TensorFlow** outside of a `virtualenv`. The Python `whl` files are located in the `build_cmake/artifacts/` and `build_cmake/artifacts/tensorflow` directories, respectively.

Verify that `openvino-tensorflow` is installed correctly:

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

This should produce an output like:

        TensorFlow version:  2.9.2
        OpenVINO integration with TensorFlow version: b'2.1.0'
        OpenVINO version used for this build: b'2022.1.0'
        TensorFlow version used for this build: v2.9.2
        CXX11_ABI flag used for this build: 1


Test the installation:

        python3 test_ovtf.py

This command runs all C++ and Python unit tests from the `openvino_tensorflow` source tree. It also runs various TensorFlow Python tests using OpenVINO™.
  
##  3. <a name='BackwardsCompatibilitywithTensorFlow'></a>Backwards Compatibility with TensorFlow on Linux
**OpenVINO™ integration with TensorFlow** core library ensures backwards compatibility across **TensorFlow 2.x APIs**. This means you will be able to build its source code with the past MINOR versions of TensorFlow 2.x. (validated for TensorFlow versions **v2.4.4, v2.5.3, v2.6.3, v2.7.1, v2.8.0 and 2.9.2**). However, TensorFlow does not guarantee the binary interfaces compatibility across its MINOR versions for the C++ runtime libraries (see https://www.tensorflow.org/guide/versions). Therefore an **OpenVINO™ integration with TensorFlow** wheel that depends on a given TensorFlow version will not work with past MINOR versions of TensorFlow out-of-the-box. For example PyPi openvino-tensorflow 2.1.0 which depends on TF 2.9.2 does not work with PyPi TensorFlow 2.6.0.

Please note that PyPi openvino-tensorflow 2.1.0 is still cross-compatible with PATCH versions of TensorFlow. For example, openvino-tensorflow 2.1.0 built against TF 2.9.2 will work with PATCH versions like TF 2.9.0, and 2.9.2. This is because security and bug fixes done in PATCH versions of TensorFlow don't affect the binary interfaces of its C++ runtime libraries.

##  4. <a name='OpenVINO'></a>OpenVINO™

OpenVINO™ can be built from source independently using `build_ov.py`
##  5. <a name='TensorFlow'></a>TensorFlow

### <a name='LinuxAndmacOS'></a>For Linux and macOS

TensorFlow can be built from source using `build_tf.py`. The build artifacts can be found under ${PATH_TO_TF_BUILD}/artifacts/

- Set your build path

        export PATH_TO_TF_BUILD=/path/to/tensorflow/build/

- For all available build options

        python3 build_tf.py -h

- Builds TensorFlow with default CXX11_ABI=1 option with a desired TensorFlow version

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.9.2 

- Builds TensorFlow with CXX11_ABI=0 with a desired TensorFlow version

        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.9.2 --cxx11_abi_version=0 

### <a name='TFWindows'></a> For Windows
- Complete the setup steps: https://www.tensorflow.org/install/source_windows#setup_for_windows
- Download TensorFlow source code and apply patch
  
        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        git checkout v2.9.2
        # apply following patch to enable the symbols required to build OpenVINO™ integration with TensorFlow
        git apply \path\to\openvino_tensorflow\repo\tools\builds\tf_win_build.patch --ignore-whitespace
        # if you want to enable more symbols add them to tensorflow\tensorflow\tools\def_file_filter\def_file_filter.py.tpl file

- Configure the Build: https://www.tensorflow.org/install/source_windows#configure_the_build      
- Build the PIP Package

        bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

        # following command builds a .whl package in the C:/tmp/tensorflow_pkg directory:  
        bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg

        # copy this PIP package to working directory (tensorflow source code directory)
        cp C:/tmp/tensorflow_pkg/tensorflow-2.9.2-cp39-cp39-win_amd64.whl .\
  
- Build CC libraries
  
        bazel build --config=opt tensorflow:tensorflow_cc.lib
