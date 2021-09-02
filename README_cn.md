[English](./README.md) | 简体中文

<p align="center">
  <img src="images/openvino_wbgd.png">
</p>

# **OpenVINO™ integration with TensorFlow（预发布版）**

该仓库包含 **OpenVINO™ integration with TensorFlow** 的源代码，该产品可提供所需的 [OpenVINO™](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) 内联优化和运行时，显著增强对TensorFlow 的兼容性。该产品专为开发人员设计，支持他们将 OpenVINO™ 运用在自己的推理应用，只需稍微修改代码，就可显著增强推理性能。**OpenVINO™ integration with TensorFlow** 可在各种英特尔<sup>®</sup> 芯片上加速AI模型（如下所示） [AI 模型](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/MODELS_cn.md)的推理速度：

- 英特尔<sup>®</sup> CPU
- 英特尔<sup>®</sup> 集成 GPU
- 英特尔<sup>®</sup> Movidius™ 视觉处理单元 (VPU)
- 支持 8 颗英特尔 Movidius™ MyriadX VPU 的英特尔<sup>®</sup> 视觉加速器设计（称作 VAD-M 或 HDDL）

\[注：为实现最佳的性能、效率、工具定制和硬件控制，我们建议使用原生 OpenVINO™ API 及其运行时。]

## 安装

### 前提条件

- Ubuntu 18.04, 20.04
- Python 3.6, 3.7 或 3.8
- TensorFlow v2.4.1

请参阅我们的[交互式安装表](https://openvinotoolkit.github.io/openvino_tensorflow/)，查看安装选项菜单。该表格将引导您完成安装过程。

### 安装 **OpenVINO™ integration with TensorFlow** 以及 PyPi TensorFlow

**OpenVINO™ integration with TensorFlow** 安装包附带 OpenVINO™ 2021.3 的预构建库，这意味着您无需单独安装 OpenVINO™。该安装包支持：

- 英特尔<sup>®</sup> CPU

- 英特尔<sup>®</sup> 集成 GPU

- 英特尔<sup>®</sup> Movidius™ 视觉处理单元 (VPU)
  
        pip3 install -U pip==21.0.1
        pip3 install -U tensorflow==2.4.1
        pip3 install openvino-tensorflow

如果您想使用支持 Movidius™ 的英特尔® 视觉加速器设计 (VAD-M) 进行推理，请安装 [**OpenVINO™ integration with TensorFlow** 以及英特尔® OpenVINO™ 工具套件发布版](docs/BUILD.md#install-openvino-integration-with-tensorflow-alongside-the-intel-distribution-of-openvino-toolkit)。

更多关于其他安装模式的详情，请参阅 [BUILD.md](docs/BUILD.md)

## 配置

安装 **OpenVINO™ integration with TensorFlow** 后，您可以在TensorFlow上对训练好的模型运行推理操作。

如要查看 **OpenVINO™ integration with TensorFlow** 是否安装正确，请运行

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

它会生成以下输出：

        TensorFlow version:  2.4.1
        OpenVINO integration with TensorFlow version: b'0.5.0'
        OpenVINO version used for this build: b'2021.3'
        TensorFlow version used for this build: v2.4.1
        CXX11_ABI flag used for this build: 0
        OpenVINO integration with TensorFlow built with Grappler: False

默认情况下，英特尔<sup>®</sup> CPU 用于运行推理。您也可以将默认选项改为英特尔<sup>®</sup> 集成 GPU 或英特尔<sup>®</sup> VPU 来进行 AI 推理。调用以下函数，更改执行推理的硬件。

    openvino_tensorflow.set_backend('<backend_name>')

支持的后端包括‘CPU'、‘GPU'、‘MYRIAD’和‘VAD-M'。

如要确定系统上的哪些处理单元用于推理，可使用以下函数：

    openvino_tensorflow.list_backends()

如欲了解更多 API 调用和环境变量的信息，请查看 [USAGE.md](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/USAGE_cn.md)。

## 示例

如欲了解 **OpenVINO™ integration with TensorFlow** 的具体功能，请查看[示例](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples)目录中的演示。

## 许可

**OpenVINO™ integration with TensorFlow** 依照 [Apache 许可版本 2.0](LICENSE)。通过贡献项目，您同意其中包含的许可和版权条款，并根据这些条款发布您的贡献。

## 支持

通过 [GitHub 问题](https://github.com/openvinotoolkit/openvino_tensorflow/issues)提交您的问题、功能请求和漏洞报告。

## 如何贡献

我们欢迎您为 **OpenVINO™ integration with TensorFlow** 做出社区贡献。如您在改进方面有好的想法：

* 请通过 [GitHub 问题](https://github.com/openvinotoolkit/openvino_tensorflow/issues)分享您的建议。
* 提交 [pull 请求](https://github.com/openvinotoolkit/openvino_tensorflow/pulls)。

我们将以最快的速度审核您的贡献！如果需要进行其他修复或修改，我们将为您提供引导和反馈。贡献之前，请确保您可以构建 **OpenVINO™ integration with TensorFlow** 并运行所有示例和修复/补丁。如果您想推出重要特性，可以创建特性测试案例。您的 pull 请求经过验证之后，我们会将其合并到存储库中，前提是 pull 请求满足上述要求并经过认可。