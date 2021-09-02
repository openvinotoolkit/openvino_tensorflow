# 安装和构建选项

## 前提条件

| 构建类型| 要求
|:----------|----------
| 使用预构建安装包| Python 3.6、3.7 或 3.8，TensorFlow v2.4.1
| 基于源代码构建| Python 3.6、3.7 或 3.8，GCC 7.5 (Ubuntu 18.04, 20.04)，cmake 3.14 或更高版本，Bazelisk v1.7.5，virtualenv 16.0.0 或更高版本，patchelf 0.9，libusb 1.0.0

## 使用预构建安装包

**OpenVINO™ integration with TensorFlow** 有两个版本：一个用 CXX11\_ABI=0 构建，另一个用 CXX11\_ABI=1 构建。

由于 [PyPi](https://pypi.org) 中提供的 TensorFlow 安装包用 CXX11\_ABI=0 构建，OpenVINO™ 版本的安装包用 CXX11\_ABI=1 构建，因此这些安装包的二进制版本**无法同时安装。**您可以根据自己的需求从以下两种方法中任选一种：

- **OpenVINO™ integration with TensorFlow** 以及 PyPi TensorFlow（CXX11\_ABI=0，无需安装 OpenVINO™，不支持 VAD-M）

- **OpenVINO™ integration with TensorFlow** 以及英特尔® OpenVINO™ 工具套件分发版（CXX11\_ABI=1，需要定制 TensorFlow 安装包，支持 VAD-M）

### 安装 **OpenVINO™ integration with TensorFlow** 以及 PyPi TensorFlow。

此 **OpenVINO™ integration with TensorFlow** 安装包包含 OpenVINO™ 版 2021.3 的预构建库。用户无需单独安装 OpenVINO™。此安装包支持英特尔<sup>®</sup> CPU、英特尔<sup>®</sup> 集成 GPU 和英特尔<sup>®</sup> Movidius™ 视觉处理单元 (VPU)。

        pip3 install -U pip==21.0.1
        pip3 install -U tensorflow==2.4.1
        pip3 install openvino-tensorflow

### 安装 **OpenVINO™ integration with TensorFlow** 以及英特尔® OpenVINO™ 工具套件发布版

此 **OpenVINO™ integration with TensorFlow** 安装包目前兼容 OpenVINO™ 版 2021.3。此安装包支持英特尔<sup>®</sup> CPU、英特尔<sup>®</sup> 集成 GPU 和英特尔<sup>®</sup> Movidius™ 视觉处理单元 (VPU) 和支持 Movidius™ 的英特尔<sup>®</sup> 视觉加速器设计 (VAD-M)。

您可以使用 -D\_GLIBCXX\_USE\_CXX11\_ABI=1 通过源代码构建 TensorFlow 或使用以下 TensorFlow 安装包：

1. 确保以下版本用于 pip 或 numpy:
   
        pip3 install --upgrade pip==21.0.1
        pip3 install numpy==1.20.2

2. 安装 `TensorFlow`。您可以根据自己的 Python 版本使用相应的安装包：
   
        pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.4.1-cp36-cp36m-manylinux2010_x86_64.whl
       
        or
       
        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.4.1-cp37-cp37m-manylinux2010_x86_64.whl
       
        or
       
        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/tensorflow_abi1-2.4.1-cp38-cp38-manylinux2010_x86_64.whl

3. 下载并安装英特尔® OpenVINO™ 工具套件分发版 2021.3 及其关联组件 ([https://software.intel.com/en-us/openvino-toolkit/download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html))。

4. 使用以下命令运行位于 <code>\<openvino\_install\_directory>/bin</code> 的 `setupvars.sh`，初始化 OpenVINO™ 环境：
   
        source setupvars.sh

5. 安装 `openvino-tensorflow`。您可以根据自己的 Python 版本选择相应的安装包：
   
        pip3.6 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp36-cp36m-linux_x86_64.whl
       
        or
       
        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp37-cp37m-linux_x86_64.whl
       
        or
       
        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v0.5.0/openvino_tensorflow_abi1-0.5.0-cp38-cp38-linux_x86_64.whl

### 预构建安装包汇总

| TensorFlow 安装包| **OpenVINO™ integration with TensorFlow** 安装包| 支持的 OpenVINO™ 版本| 支持的硬件后端| 注释
|----------|----------|----------|----------|----------
| tensorflow| openvino-tensorflow| 基于源代码构建的 OpenVINO™ | CPU，GPU，MYRIAD| OpenVINO™ 库通过源代码构建，包含在 wheel 安装包中
| tensorflow-abi1| openvino-tensorflow-abi1| 动态链接至 OpenVINO™ 二进制版本| CPU，GPU，MYRIAD，VAD-M| **OpenVINO™ integration with TensorFlow** 库可动态链接至 OpenVINO™ 二进制文件

## 基于源代码构建

克隆 `openvino_tensorflow` 库：

```bash
$ git clone https://github.com/openvinotoolkit/openvino_tensorflow.git
$ cd openvino_tensorflow
$ git submodule init
$ git submodule update --recursive
```

安装以下 python 安装包

```bash
$ pip3 install -U psutil==5.8.0 wheel==0.36.2
```

### **OpenVINO™ integration with TensorFlow**

根据要求选择以下任一构建选项

1. 从 PyPi 中取出兼容的预构建 TF 安装包，克隆并通过源代码构建 OpenVINO™。
   
        python3 build_ovtf.py

2. 从 PyPi 中取出兼容的预构建 TF 安装包。使用 OpenVINO™ 二进制文件。
   
        python3 build_ovtf.py --use_openvino_from_location=/opt/intel/openvino_2021.3.394/ --cxx11_abi_version=1

3. 通过源代码取出并构建 TF 和 OpenVINO™
   
        python3 build_ovtf.py --build_tf_from_source

4. 通过源代码取出并构建 TF。使用 OpenVINO™ 二进制文件。
   
        python3 build_ovtf.py --build_tf_from_source --use_openvino_from_location=/opt/intel/openvino_2021.3.394/ --cxx11_abi_version=1

5. 使用指定位置的预构建 TF（[参阅 TensorFlow 构建指令](#tensorflow)）。通过源代码取出并构建 OpenVINO™。如需频繁构建 **OpenVINO™ integration with TensorFlow**，可以使用该方法，无需每次通过源代码构建 TF。
   
        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/

6. 使用指定位置的预构建 TF（[参阅 TensorFlow 构建指令](#tensorflow)）。使用 OpenVINO™ 二进制文件。它仅兼容 ABI1 构建的 TF。
   
        python3 build_ovtf.py --use_tensorflow_from_location=/path/to/tensorflow/build/  --use_openvino_from_location=/opt/intel/openvino_2021/ --cxx11_abi_version=1

选择 `build_ovtf.py` 脚本的 `help` 选项，了解更多关于各种构建选项的信息。

        python3 build_ovtf.py --help

#### 验证

构建完成后，`build_cmake/venv-tf-py3` 中创建了一个新的 `virtualenv` 目录。`build_cmake/artifacts/` 目录中创建了一些 Build artifact（如 `openvino_tensorflow-<VERSION>-cp36-cp36m-manylinux2014_x86_64.whl`）

激活以下 `virtualenv`，开始使用 **OpenVINO™ integration with TensorFlow**。

        source build_cmake/venv-tf-py3/bin/activate

您还可以在 `virtualenv` 之外安装 TensorFlow 和 **OpenVINO™ integration with TensorFlow。**Python `whl` 文件分别位于 `build_cmake/artifacts/` 和 `build_cmake/artifacts/tensorflow` 目录。

验证 `openvino-tensorflow` 是否安装正确：

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

它会生成以下输出：

        TensorFlow version:  2.4.1
        OpenVINO integration with TensorFlow version: b'0.5.0'
        OpenVINO version used for this build: b'2021.3'
        TensorFlow version used for this build: v2.4.1
        CXX11_ABI flag used for this build: 1
        OpenVINO integration with TensorFlow built with Grappler: False

测试安装：

        python3 test_ovtf.py

该命令将运行 `openvino_tensorflow` 源树的所有 C++ 和 Python 单元测试。还使用 OpenVINO™ 运行各种 TensorFlow Python 测试。

### 构建用于英特尔® 凌动® 处理器的指令

为了构建 **OpenVINO™ integration with TensorFlow** 以便用于英特尔® 凌动® 处理器，我们建议通过源代码构建 TF。以下命令将通过源代码构建用于英特尔® 凌动® 处理器的 TF 和 OpenVINO™。

        python3 build_ovtf.py --build_tf_from_source --cxx11_abi_version=1 --target_arch silvermont

## TensorFlow

TensorFlow 可使用 `build_tf.py` 通过源代码来构建。build artifact 位于 ${PATH\_TO\_TF\_BUILD}/artifacts/ 下方

- 设置构建路径
  
        export PATH_TO_TF_BUILD=/path/to/tensorflow/build/

- 适用于所有可用构建选项
  
        python3 build_tf.py -h

- 用 CXX11\_ABI=0 构建 TF。
  
        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --cxx11_abi_version=0

- 用 CXX11\_ABI=1 构建 TF。
  
        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --cxx11_abi_version=1

- 为使用所需的 TF 版本（如 v2.4.1）来构建
  
        python3 build_tf.py --output_dir=${PATH_TO_TF_BUILD} --tf_version=v2.4.1

## OpenVINO™

OpenVINO™ 可使用 `build_ov.py` 通过源代码单独构建。

## 构建兼容 ManyLinux2014 的 **OpenVINO™ integration with TensorFlow** wheel

如要构建兼容 manylinux2014 的 wheel 文件，可使用以下命令。build artifact 位于容器的 /whl/ 文件夹。

```bash
cd tools/builds/
docker build --no-cache -t openvino_tensorflow/pip --build-arg OVTF_BRANCH=releases/v0.5.0 . -f Dockerfile.manylinux2014
```