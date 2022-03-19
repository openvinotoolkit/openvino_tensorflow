[English](./BUILD.md) | 简体中文

# 构建指令

<!-- markdown-toc -->
 1. [先决条件](#先决条件)
	* 1.1. [Ubuntu](#Ubuntu)
	* 1.2. [macOS](#macOS)
	* 1.3. [Windows](#Windows)
 2. [OpenVINO™ integration with TensorFlow](#OpenVINOintegrationwithTensorFlow)
 	* 2.1. [针对Linux和macOS的构建指令](#BuildInstructionsLinuxAndmacOS)
	* 2.2. [针对Windows的构建指令](#BuildInstructionsWindows)
	* 2.3. [Atom平台构建指令](#BuildInstructionsforIntelAtomProcessor)
	* 2.4. [构建验证](#BuildVerification)	
 3. [与TensorFlow的后端兼容性](#BackwardsCompatibilitywithTensorFlow)
 4. [OpenVINO™](#OpenVINO)
 5. [TensorFlow](#TensorFlow)
 

<!-- markdown-toc-config
	numbering=true
	autoSave=true
	/markdown-toc-config -->
<!-- /markdown-toc -->

首先，克隆`openvino_tensorflow`仓库：

  ```bash
  $ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
  $ cd openvino_tensorflow
  $ git submodule init
  $ git submodule update --recursive
  ```

##  1. <a name='Prerequisites'></a>Prerequisites

###  1.1. <a name='Ubuntu'></a>Ubuntu

1. 安装 apt 包  

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

2. 安装 Python 需求
      ```bash
      $ pip3 install -r requirements.txt
      ```  

2. 安装 CMake, 支持版本 >=3.14.0
        
      ```bash
      # to install CMake 3.18.4
      $ wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.tar.gz && \
      $ tar -xzvf cmake-3.18.4-Linux-x86_64.tar.gz && \
      $ cp cmake-3.18.4-Linux-x86_64/bin/* /usr/local/bin/ && \
        cp -r cmake-3.18.4-Linux-x86_64/share/cmake-3.18 /usr/local/share/
      ```

3. 安装 Bazelisk (可选; 仅在从源代码构建 TensorFlow 时需要)

      ```bash
      $ apt-get update && apt-get install -y openjdk-8-jdk
      $ curl -fsSL https://deb.nodesource.com/setup_12.x | bash -
      $ apt-get install -y nodejs
      $ npm install -g @bazel/bazelisk
      ```

###  1.2. <a name='macOS'></a>macOS

1. 安装 HomeBrew 包

      ``` bash
      $ brew install python@3.9
      $ brew install cmake autoconf automake libtool libusb wget 
      ```

2. 安装 Pip 和需求

      ```bash
      $ wget https://bootstrap.pypa.io/get-pip.py
      $ python3 get-pip.py
      $ pip3 install -r requirements.txt
      ```

3. 安装 Bazelisk v1.10.1版本 (可选; 仅在从源代码构建 TensorFlow 时需要)

      ```bash
      $ wget https://github.com/bazelbuild/bazelisk/releases/download/v1.10.1/bazelisk-darwin-amd64
      $ mv bazelisk-darwin-amd64 /usr/local/bin/bazel
      $ chmod 777 /usr/local/bin/bazel
      ```

4. 安装 Apple XCode 命令行工具

      ```bash
      $ xcode-select --install
      ```

注意:
         在 macOS 版本 11.2.3 和 CMake 版本 3.21.1 上开发和测试
         用户可以安装 Python 3.7、3.8 或 3.9。

###  1.3. <a name='Windows'></a>Windows
1. 安装 [Python3.9](https://www.python.org/downloads/)

2. 安装 Visual C++ Build Tools 2019   
   Visual C++ build tools 2019与Visual Studio 2019在一起，但可以单独安装:
   - 访问 Visual Studio [downloads](https://visualstudio.microsoft.com/downloads/),  
   - 选择 Redistributables and Build Tools,
   - 下载并安装:  
        Microsoft Visual C++ 2019 Redistributable  
        Microsoft Build Tools 2019

3. 安装 [CMake](https://cmake.org/download/), 支持版本 >=3.14.0
4. [下载并安装 OpenVINO™ Toolkit 2022.1 LTS for Windows](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit-download.html)

<br>

##  2. <a name='OpenVINOintegrationwithTensorFlow'></a>OpenVINO™ integration with TensorFlow

###  2.1. <a name='BuildInstructionsLinuxAndmacOS'></a>针对**Linux和macOS**的构建指令
根据需求使用以下构建选项之一, **OpenVINO™ integration with TensorFlow** 使用 PyPI TensorFlow 构建，仅支持 Python API。 

使用 C++ API 需要使用 TensorFlow C++ 库构建**OpenVINO™ integration with TensorFlow**(使用 --use_tensorflow_from_location 选项)。要使用VAD-M，请使用OpenVINO™ 库进行构建(使用 --use_openvino_from_location选项)。

1. 从 PyPi 中获取兼容的TensorFlow 包，从源代码克隆和构建 OpenVINO™。 参数是可选的。 如果未提供任何参数，则将使用 build_ovtf.py 中指定的默认版本。

        python3 build_ovtf.py --tf_version=v2.8.0 --openvino_version=2022.1
- 使用OpenVINO™主分支，需将"master"作为值传递至"openvino_version"参数。 **OpenVINO™ integration with TensorFlow** 构建脚本将拉取OpenVINO™主分支的最新提交，但所有openvino-tensorflow API和功能仍将同OpenVINO™版本2022.1。

2. 从 Github 发布版本中获取兼容的TensorFlow ABI1 包。 使用来自指定位置的 OpenVINO™ 二进制文件。

        python3 build_ovtf.py --use_openvino_from_location=/opt/intel/openvino_2022.1/ --cxx11_abi_version=1

3. 使用指定位置的预构建 TensorFlow（[参阅 TensorFlow 构建指令](#LinuxAndmacOS)）。通过源代码构建 OpenVINO™。如需频繁构建 **OpenVINO™ integration with TensorFlow**，可以使用该方法，无需每次通过源代码构建TensorFlow。

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/ --openvino_version=2022.1

4. 使用指定位置的预构建 TensorFlow（[参阅 TensorFlow 构建指令](#LinuxAndmacOS)）。使用指定位置的 OpenVINO™ 二进制文件。** 它仅兼容 ABI1 构建的 TensorFlow **。

        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/ --use_openvino_from_location=/opt/intel/openvino_2021.4.752/ --cxx11_abi_version=1

选择 `build_ovtf.py` 脚本的 `help` 选项，了解更多关于各种构建选项的信息。

        python3 build_ovtf.py --help

###  2.2. <a name='BuildInstructionsWindows'></a>针对 **Windows**的构建指令
将 "Command Prompt"或"x64 Native Tools Command Prompt for VS 2019"作为管理员使用，并从源代码采用以下构建选项：
1. 从 Github 发布版本中获取兼容的预构建 TensorFlow 包。 使用来自指定位置的 OpenVINO™ 二进制文件。

        python build_ovtf.py --tf_version=v2.8.0 --use_openvino_from_location="C:\Program Files (x86)\Intel\openvino_2021.4.752" 

2.  使用指定位置的预构建 TensorFlow（[参阅 TensorFlow 构建指令](#TFWindows)）。使用指定位置的OpenVINO™ 二进制文件。使用此构建选项运行C++示例，并将openvino-tensorflow集成至TensorFlow C++推理应用中。

        python build_ovtf.py --use_openvino_from_location="C:\Program Files (x86)\Intel\openvino_2022.1" --use_tensorflow_from_location="\path\to\directory\containing\tensorflow\"

###  2.3. <a name='BuildInstructionsforIntelAtomProcessor'></a>Intel Atom® 处理器的构建指令
为了构建 **OpenVINO™ integration with TensorFlow** 与英特尔凌动® 处理器一起使用，我们建议使用以下命令从源代码构建 TensorFlow：

        python3 build_ovtf.py --build_tf_from_source --cxx11_abi_version=1 --target_arch silvermont

###  2.4. <a name='BuildVerification'></a> Build Verification
构建完成后，会在 `build_cmake`内创建一个新的名为 `venv-tf-py3` 的`virtualenv` 目录。`build_cmake/artifacts/` 目录中创建了一些构建包（如 `openvino_tensorflow-<VERSION>-cp38-cp38-manylinux2014_x86_64.whl`）

激活以下 `virtualenv`，开始使用 **OpenVINO™ integration with TensorFlow** 。

        source build_cmake/venv-tf-py3/bin/activate

您还可以在 `virtualenv` 之外安装 TensorFlow 和 **OpenVINO™ integration with TensorFlow ** 。 Python `whl` 文件分别位于 `build_cmake/artifacts/` 和 `build_cmake/artifacts/tensorflow` 目录。

验证 `openvino-tensorflow` 是否安装正确：

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

它会生成以下输出：

        TensorFlow version:  2.8.0
        OpenVINO integration with TensorFlow version: b'2.0.0'
        OpenVINO version used for this build: b'2022.1'
        TensorFlow version used for this build: v2.8.0
        CXX11_ABI flag used for this build: 0


测试安装：

        python3 test_ovtf.py

该命令将运行 `openvino_tensorflow` 源代码树的所有 C++ 和 Python 单元测试。还使用 OpenVINO™ 运行各种 TensorFlow Python 测试。
  
##  3. <a name='BackwardsCompatibilitywithTensorFlow'></a>TensorFlow在Linux平台的向后兼容性
**OpenVINO™ integration with TensorFlow**核心库保证**TensorFlow 2.x APIs**的向后兼容性。这意味着您将可以使用TensorFlow 2.x.过去的副版本（已验证TensorFlow v2.4.4、v2.5.3、v2.6.3、v2.7.1和v2.8.0版本) 构建源代码。但TensorFlow不保证其副版本对C++运行时库的二进制界面兼容性（参见https://www.tensorflow.org/guide/versions）。 

因此依赖指定TensorFlow版本的 **OpenVINO™ integration with TensorFlow** wheel将无法兼容TensorFlow out-of-the-box过去的副版本（示例：依赖TF 2.8.0的PyPi openvino-tensorflow 2.0.0无法兼容PyPi TensorFlow 2.6.0）。

请注意PyPi openvino-tensorflow 2.0.0仍兼容TensorFlow PATCH版本。例如，针对 TF 2.8.0构建的openvino-tensorflow将兼容像TF 2.8.1和2.8.2 PATCH版本。 这是因为TensorFlow PATCH版本中完成的安全与漏洞修复不影响其C++运行时库的二进制接口。

##  4. <a name='OpenVINO'></a>OpenVINO™

OpenVINO™ 可以使用 `build_ov.py` 从源代码独立构建
##  5. <a name='TensorFlow'></a>TensorFlow

### <a name='LinuxAndmacOS'></a>针对Linux and macOS

TensorFlow 可以使用 `build_tf.py` 从源代码构建。 可以在 ${PATH_TO_TF_BUILD}/artifacts/ 下找到构建工件

- 设置你的构建路径

        export PATH_TO_TF_BUILD=/path/to/tensorflow/build/

- 适用于所有可用构建选项
  
        python3 build_tf.py -h

- 用 CXX11\_ABI=0 构建指定版本的TensorFlow。
  
        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.8.0 --cxx11_abi_version=0

- 用 CXX11\_ABI=1 构建指定版本的TensorFlow。
  
        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.8.0 --cxx11_abi_version=1



### <a name='TFWindows'></a> 针对Windows
- 完成设置步骤: https://www.tensorflow.org/install/source_windows#setup_for_windows
- 下载 TensorFlow 源码并打上补丁
  
        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        git checkout v2.8.0
        # apply following patch to enable the symbols required to build OpenVINO™ integration with TensorFlow
        git apply patch \path\to\openvino_tensorflow\repo\tools\builds\tf_2_7_0_build.patch
        # if you want to enable more symbols add them to tensorflow\tensorflow\tools\def_file_filter\def_file_filter.py.tpl file

- 配置构建环境: https://www.tensorflow.org/install/source_windows#configure_the_build      
- 构建 PIP 包

        bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

        # following command builds a .whl package in the C:/tmp/tensorflow_pkg directory:  
        bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg

        # copy this PIP package to working directory (tensorflow source code directory)
        cp C:/tmp/tensorflow_pkg/tensorflow-2.8.0-cp39-cp39-win_amd64.whl .\
  
- 构建 CC 库
  
        bazel build --config=opt tensorflow:tensorflow_cc.lib
