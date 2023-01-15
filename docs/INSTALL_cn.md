[English](./INSTALL.md) | 简体中文

# <a name='Pre-BuiltPackages'></a>使用预构建软件包安装

**OpenVINO™ integration with TensorFlow** 有 Linux, macOS 和 Windows 发行版. 你可以根据需要从下列安装方法中选择一种。

## Linux

  ### 安装 **OpenVINO™ integration with TensorFlow** PyPi 发布版
  * 包括 Intel<sup>®</sup> OpenVINO™ 2022.3.0 版的预建库，用户无需单独安装 OpenVINO™。
  * 支持 Intel<sup>®</sup> CPU、Intel<sup>®</sup> 集成 GPU 和 Intel<sup>®</sup> Movidius™ 视觉处理单元 (VPU)，但不支持 VAD-M。

        pip3 install -U pip
        pip3 install tensorflow==2.9.3
        pip3 install openvino-tensorflow==2.3.0
    openvino-tensorflow PyPi 安装包对tensorflow的补丁版交叉兼容。例如 openvino-tensorflow 针对TF 2.9.2 的模块依然可以在像 TF 2.9.2 和 2.9.3 这样的补丁版本上运行。
<br/>

  ### 安装 **OpenVINO™ integration with TensorFlow** PyPi 发布版与独立安装Intel® OpenVINO™ 发布版以支持VAD-M
    * 兼容 Intel<sup>®</sup> OpenVINO™ 2022.3.0版本
    * 支持 Intel<sup>®</sup> Movidius™ (VAD-M) 的视觉加速器设计 同时支持 Intel<sup>®</sup> CPU、Intel<sup>®</sup> 集成 GPU、Intel<sup>®</sup> Movidius™ 视觉处理单元 (VPU)。 
    * 使用方法：
    1. 按照上述方法从PyPi安装tensorflow 和 openvino-tensorflow。
    2. 下载安装Intel<sup>®</sup> OpenVINO™ 2022.3.0发布版，一并安装其依赖([https://software.intel.com/en-us/openvino-toolkit/download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html))。
    3. 初始化Intel<sup>®</sup> OpenVINO™, 可以运行位于<code>\<openvino\_install\_directory\></code> 的命令脚本`setupvars.sh` 。该步骤需要在运行openvino-tensorflow做推理的同一个环境下执行。

        source setupvars.sh  
      
  
## macOS

  安装 **OpenVINO™ integration with TensorFlow** PyPi 发布版
  * 包括 Intel<sup>®</sup> OpenVINO™ 2022.3.0 版的预建库，用户无需单独安装Intel<sup>®</sup> OpenVINO™ 
  * 支持 Intel<sup>®</sup> CPU、Intel<sup>®</sup> 集成 GPU 和 Intel<sup>®</sup> Movidius™ 视觉处理单元 (VPU)，但不支持 VAD-M。

        pip3 install -U pip
        pip3 install tensorflow==2.9.3
        pip3 install openvino-tensorflow==2.3.0


## Windows

  安装 **OpenVINO™ integration with TensorFlow** PyPi 版本与独立安装TensorFlow Github版本
  * 基于Windows 的TensorFlow PyPi 安装版并没有使能 **OpenVINO™ integration with TensorFlow** 需要的所有API。用户需要从Github 发布中安装TensorFlow wheel。
  * 包括 OpenVINO™ 2022.3.0 版的预建库。 用户无需单独安装 Intel<sup>®</sup> OpenVINO™ 。
  * 支持 Intel<sup>®</sup> CPU、Intel<sup>®</sup> 集成 GPU 和 Intel<sup>®</sup> Movidius™ 视觉处理单元 (VPU)，但不支持 VAD-M。

        pip3.9 install -U pip
        pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v2.2.0/tensorflow-2.9.2-cp39-cp39-win_amd64.whl
        pip3.9 install openvino-tensorflow==2.2.0
