# **OpenVINO™ integration with TensorFlow**

该仓库包含 **OpenVINO™ integration with TensorFlow** 的源代码，该产品可支持 OpenVINO™ 运行时并实现 TensorFlow 优化。**OpenVINO™ integration with TensorFlow** 支持在各种英特尔<sup>®</sup> 芯片（如下所示）上使用各种 AI 模型，加快各种用例中的 AI 推理速度：

- 英特尔<sup>®</sup> CPU
- 英特尔<sup>®</sup> 集成 GPU
- 英特尔<sup>®</sup> Movidius™ 视觉处理单元 (VPU)
- 支持 8 颗英特尔 Movidius™ MyriadX VPU 的英特尔<sup>®</sup> 视觉加速器设计（称作 VAD-M 或 HDDL）

## 安装

### 要求

- Python 3.6、3.7 或 3.8
- TensorFlow 2.4.1

### 使用 **OpenVINO™ integration with TensorFlow** 以及 PyPi TensorFlow。

此 **OpenVINO™ integration with TensorFlow** 安装包随附 OpenVINO™ 2021.3 的预构建库。用户无需单独安装 OpenVINO™。此安装包支持英特尔<sup>®</sup> CPU、英特尔<sup>®</sup> 集成 GPU 和英特尔<sup>®</sup> Movidius™ 视觉处理单元 (VPU)。

        pip3 install -U pip==21.0.1
        pip3 install -U tensorflow==2.4.1
        pip3 install openvino-tensorflow

如要将 **OpenVINO™ integration with TensorFlow** 用于预安装的二进制文件，请访问此页了解详细说明： ([**OpenVINO™ integration with TensorFlow** - README](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/BUILD_cn.md))

## 验证安装

验证 `openvino-tensorflow` 是否安装正确：

    python3 -c "import tensorflow as tf; print('TensorFlow version: ',tf.__version__);\
                import openvino_tensorflow; print(openvino_tensorflow.__version__)"

它将生成如下结果：

        TensorFlow version:  2.4.1
        OpenVINO integration with TensorFlow version: b'0.5.0'
        OpenVINO version used for this build: b'2021.3'
        TensorFlow version used for this build: v2.4.1
        CXX11_ABI flag used for this build: 0
        OpenVINO integration with TensorFlow built with Grappler: False

测试安装：

        python3 test_ovtf.py

该命令将运行 `openvino_tensorflow` 源树的所有 C++ 和 Python 单元测试。

## 用途

安装 **OpenVINO™ integration with TensorFlow** 后，您可以通过 TensorFlow，使用经过训练的模型来运行推理。需要对脚本做的惟一更改是添加

    import openvino_tensorflow

如要确定系统上可用的后端，可使用以下 API：

    openvino_tensorflow.list_backends()

默认情况下支持 CPU 后端。您可以使用以下 API，将默认的 CPU 后端替换为其他后端：

    openvino_tensorflow.set_backend('backend_name')

更多关于如何使用 **OpenVINO™ integration with TensorFlow** 的详细示例，请前往[**示例**](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples)目录。

## 许可

**OpenVINO™ integration with TensorFlow** 依照 [Apache 许可版本 2.0](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/LICENSE)。通过贡献项目，您同意其中包含的许可和版权条款，并根据这些条款发布您的贡献。

## 支持

请通过 [****GitHub 问题****](https://github.com/openvinotoolkit/openvino_tensorflow/issues)提交您的问题、功能请求和漏洞报告。

## 如何贡献

我们欢迎您为 **OpenVINO™ integration with TensorFlow** 做出社区贡献。如您在改进方面有好的想法：

* 请通过 [****GitHub 问题****](https://github.com/openvinotoolkit/openvino_tensorflow/issues)分享您的建议。
* 确保您可以通过补丁构建产品并运行所有示例。
* 如果是较大的特性，可创建测试。
* 提交 [****pull 请求****](https://github.com/openvinotoolkit/openvino_tensorflow/pulls)。
* 我们将审核您的贡献，如有必要进行其他修复或修改，我们将为您提供引导和反馈。pull 请求经过认可后，将会合并到存储库中。
* 关于贡献 OpenVINO 开源项目的所有指南，请访问[此处](https://github.com/openvinotoolkit/openvino/wiki/Contribute)