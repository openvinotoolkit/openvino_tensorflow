[English]|(./INSTALL.md) | 简体中文
# <a name='Pre-BuiltPackages'></a>使用预编译软件包安装

**OpenVINO™ integration with TensorFlow** 有 Linux, macOS 和 Windows 发行版. 你可以根据需要从下列安装方法中选择一种。

## Linux

Linux环境下的**OpenVINO™ integration with TensorFlow** 以两种不同的版本发布：一种使用 CXX11_ABI=0 构建，另一种使用 CXX11_ABI=1 构建。

由于 [PyPi](https://pypi.org) 中可用的 TensorFlow 包是使用 CXX11_ABI=0 构建的，而 OpenVINO™ 发布包是使用 CXX11_ABI=1 构建的，因此这些包的二进制版本 **不能一起安装**。 

- [**OpenVINO™ integration with TensorFlow** PyPi 与 PyPi TensorFlow 一起安装](#InstallOpenVINOintegrationwithTensorFlowalongsidePyPiTensorFlow)
    * 包括 OpenVINO™ 2021.4.2 版的预建库。 用户无需单独安装 OpenVINO™ 
    * 支持 Intel<sup>®</sup> CPU、Intel<sup>®</sup> 集成 GPU 和 Intel<sup>®</sup> Movidius™ 视觉处理单元 (VPU)，但不支持 VAD-M。
    * 使用 CXX11_ABI=0 构建  

<br/>  

- [**OpenVINO™ integration with TensorFlow** Github版本与 Intel® Distribution of OpenVINO™ Toolkit 一起安装](#InstallOpenVINOintegrationwithTensorFlowalongsidetheIntelDistributionofOpenVINOToolkit)
    * 兼容 OpenVINO™ 版本 2021.4.2
    * 支持 Intel<sup>®</sup> CPU、Intel<sup>®</sup> 集成 GPU、Intel<sup>®</sup> Movidius™ 视觉处理单元 (VPU) 和 Intel<sup>®</sup> sup> 使用 Movidius™ (VAD-M) 的视觉加速器设计
    * 使用 CXX11_ABI=1 构建  
    * 需要一个自定义的 TensorFlow ABI1 包，该包在 Github 版本中可用 

<br/>  

## macOS

  - [**OpenVINO™ integration with TensorFlow** PyPi 与 PyPi TensorFlow 一起安装](#InstallOpenVINOintegrationwithTensorFlowalongsidePyPiTensorFlow)
  * 包括 OpenVINO™ 2021.4.2 版的预建库。 用户无需单独安装 OpenVINO™ 
  * 支持 Intel<sup>®</sup> CPU、Intel<sup>®</sup> 集成 GPU 和 Intel<sup>®</sup> Movidius™ 视觉处理单元 (VPU)，但不支持 VAD-M。

<br/>  

## Windows

  - [**OpenVINO™ integration with TensorFlow** PyPi 与 TensorFlow Github版本一起安装](#InstallOpenVINOintegrationwithTensorFlowalongsideTensorFlow)
  * 包括 OpenVINO™ 2021.4.2 版的预建库。 用户无需单独安装 OpenVINO™ 
  * 支持 Intel<sup>®</sup> CPU、Intel<sup>®</sup> 集成 GPU 和 Intel<sup>®</sup> Movidius™ 视觉处理单元 (VPU)，但不支持 VAD-M。
  * PyPi上的TensorFlow Windows安装包中没有包含**OpenVINO™ integration with TensorFlow**所需的所有API符号。用户需要从Github release页面中下载安装TensorFlow。
  
<br/> 

## <a name='Prebuiltpackagessummary'></a>预构建安装包汇总
  
|TensorFlow 安装包| **OpenVINO™ integration with TensorFlow** 安装包|支持的 OpenVINO™ 版本|支持的硬件后端|注释|
| -----------------|-----------------------------------|----------------------------|---------------------------|----------------|
|tensorflow| openvino-tensorflow| 基于源代码构建的 OpenVINO™|CPU,iGPU,MYRIAD|**OpenVINO™** 库通过源代码构建，包含在 wheel 安装包中|
|tensorflow-abi1| openvino-tensorflow-abi1|动态链接至 OpenVINO™ 二进制版本|CPU,iGPU,MYRIAD,VAD-M|**OpenVINO™ integration with TensorFlow** 库可动态链接至 **OpenVINO™** 二进制文件|
<br/>  

##  1.1. <a name='InstallOpenVINOintegrationwithTensorFlowalongsidePyPiTensorFlow'></a>安装 **OpenVINO™ integration with TensorFlow** 及 PyPi TensorFlow (Linux, macOS上可用)

        pip3 install -U pip
        pip3 install tensorflow==2.7.0
        pip3 install -U openvino-tensorflow==1.1.0
<br/> 

##  1.2. <a name='InstallOpenVINOintegrationwithTensorFlowalongsideTensorFlow'></a>**OpenVINO™ integration with TensorFlow** PyPi 与 TensorFlow Github版本一起安装 (Windows可用)

        pip3.9 install -U pip
        pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/tensorflow-2.7.0-cp39-cp39-win_amd64.whl
        pip3.9 install -U openvino-tensorflow
<br/> 

##  1.3. <a name='InstallOpenVINOintegrationwithTensorFlowalongsidetheIntelDistributionofOpenVINOToolkit'></a>安装 **OpenVINO™ integration with TensorFlow** 及 Intel® Distribution of OpenVINO™ Toolkit (Linux可用)

1. 确保 pip 和 numpy版本如下：

        pip3 install -U pip
        pip3 install numpy==1.20.2

2. 根据您的 Python 版本安装 `TensorFlow`，您可以使用 -D_GLIBCXX_USE_CXX11_ABI=1 构建 [TensorFlow from source](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/BUILD_cn.md#tensorflow) 或按照以下说明使用适当的包：

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/tensorflow_abi1-2.7.0-cp37-cp37m-manylinux2010_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/tensorflow_abi1-2.7.0-cp38-cp38-manylinux2010_x86_64.whl

        or

        pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/tensorflow_abi1-2.7.0-cp39-cp39-manylinux2010_x86_64.whl

3. 下载并安装英特尔® OpenVINO™ Toolkit 2021.4.2 发行版及其依赖项 ([https://software.intel.com/en-us/openvino-toolkit/download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html)).

4. 使用位于 <code>\<openvino\_install\_directory\>\/bin</code> 中的 `setupvars.sh` 命令来初始化 OpenVINO™ 环境：

        source setupvars.sh

5. 安装“openvino-tensorflow”，根据您的 Python 版本，在下面选择合适的包：

        pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/openvino_tensorflow_abi1-1.1.0-cp37-cp37m-linux_x86_64.whl

        or

        pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/openvino_tensorflow_abi1-1.1.0-cp38-cp38-linux_x86_64.whl

        or

        pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0/openvino_tensorflow_abi1-1.1.0-cp39-cp39-linux_x86_64.whl
