[English](./USAGE.md) | 简体中文

# **OpenVINO™ integration with TensorFlow** 的 API 和环境变量

本文介绍了适用于 **OpenVINO™ integration with TensorFlow** 的 Python API。第一节主要介绍在 TensorFlow 应用中发挥 **OpenVINO™ integration with TensorFlow** 的功能所需的基本 API 和代码行。

## 关于基本功能的 API

如要将环境变量安装包添加至 TensorFlow python 应用，需使用以下代码行导入安装包：

    import openvino_tensorflow

[注意：这将把**CUDA_VISIBLE_DEVICES**环境变量设置为-1。当禁用**OpenVINO™ integration with TensorFlow**后，该变量将恢复其原先的状态。]

默认情况下禁用 CPU 后端。您可以使用以下 API 设置其他后端：

    openvino_tensorflow.set_backend('<backend_name>')

支持的后端包括‘CPU'、‘GPU'、'GPU_FP16', ‘MYRIAD’和‘VAD-M'。


## 其他 API

如要确定系统上可用的后端，可使用以下 API：

    openvino_tensorflow.list_backends()

如要禁用 **OpenVINO™ integration with TensorFlow** ，可使用以下 API：

    openvino_tensorflow.disable()

如要启用**OpenVINO™ integration with TensorFlow** ，可使用以下 API：

    openvino_tensorflow.enable()

如要查看是否已禁用**OpenVINO™ integration with TensorFlow**，可使用以下 API：

    openvino_tensorflow.is_enabled()

如要获取分配的后端，可使用以下 API：

    openvino_tensorflow.get_backend()

如要启用 TensorFlow pipeline和布局阶段以及 **OpenVINO™ integration with TensorFlow** 详细的执行日志，可使用以下 API：

    openvino_tensorflow.start_logging_placement()

如要禁用 TensorFlow pipline 和布局阶段以及 **OpenVINO™ integration with TensorFlow** 详细的执行日志，可使用以下 API：

    openvino_tensorflow.stop_logging_placement()

如要查看布局日志是否已启用，可使用以下 API：

    openvino_tensorflow.is_logging_placement()

如要查看用于编译 **OpenVINO™ integration with TensorFlow** 的 CXX11\_ABI，可使用以下 API：

    openvino_tensorflow.cxx11_abi_flag()

如要禁用 OpenVINO™ 后端上部分算子的执行，可使用以下 API 在原生 TensorFlow 运行时上运行它们：

    openvino_tensorflow.set_disabled_ops(<string_of_operators_separated_by_commas>)

算子串应该用逗号隔开，并以上述 API 的参数形式提供。

如要查看后端声明支持，但程序上已禁用的算子列表，可使用以下 API：

    openvino_tensorflow.get_disabled_ops()

要启用或禁用动态回退，请使用以下 API（启用后，在运行期间导致错误的集群可以回退到原生 TensorFlow，尽管它们被分配为在 OpenVINO™ 上运行）。

    openvino_tensorflow.enable_dynamic_fallback()
    openvino_tensorflow.disable_dynamic_fallback()

要将已转换的集群中间表示文件 (IR) 导出到目录，请使用下面的 API。 此 API 将最近执行的模型中的 IR 导出并保存为“.xml”和“.bin”文件，稍后可用于 OpenVINO™ 应用程序。 此 API 的第一个参数是输出目录。 如果相应目录中有任何预先存在的 IR 文件，它会在覆盖任何旧文件之前要求用户确认。 要禁用此检查，请传递“False”值作为第二个参数（可选）。 然后，如果 IR 文件名相同，则任何预先存在的 IR 文件将被覆盖而无需任何确认。

    openvino_tensorflow.export_ir("output/directory/path")

或

    openvino_tensorflow.export_ir("output/directory/path", False)

## 环境变量

**OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS**
默认条件下禁用该变量，且在图表解析阶段将来自TensorFlow's ReadVariableOp的变量冻结为常量。强烈建议启用该变量，以保证需迫切执行模型上的最佳推理延时。加载推理模型后当模型权值修改时禁用此变量。

**OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS:**
形成集群后，由于某些原因（如集群太小，目标设备不支持某些条件），部分集群可能仍然会返回原生 TensorFlow。如果已设置此变量，集群将不会被删除，而是强行在 OpenVINO™ 后端上运行。这可能会导致性能降低，一定情况下还会导致执行崩溃。

示例：

    OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS="1"

**OPENVINO_TF_VLOG_LEVEL:**
此变量用于打印执行日志。设为 1 将打印极少细节，设为 5 将打印最详细的日志。

示例：

    OPENVINO_TF_VLOG_LEVEL="4"

**OPENVINO_TF_LOG_PLACEMENT:**
如果此变量设为 1，将打印与集群形成和封装相关的日志。

示例：

    OPENVINO_TF_LOG_PLACEMENT="1"

**OPENVINO_TF_BACKEND:** 
可使用此变量设置后端设备名称。可以设为“CPU”，“GPU”，"GPU_FP16",“MYRIAD”或“VAD-M”。

示例：

    OPENVINO_TF_BACKEND="MYRIAD"

**OPENVINO_TF_DISABLED_OPS:**
使用此变量传递已禁用算子列表。这些算子不考虑进行集群化，而是返回原生 TensorFlow。

示例：

    OPENVINO_TF_DISABLED_OPS="Squeeze,Greater,Gather,Unpack"

**OPENVINO_TF_CONSTANT_FOLDING：** 
它将启用/禁用已解析集群上constant的folding pass（默认禁用）。

示例：

    OPENVINO_TF_CONSTANT_FOLDING="1"

**OPENVINO_TF_TRANSPOSE_SINKING:**
它将启用/禁用已解析集群上的 transpose sinking pass（默认启用）。

示例：

    OPENVINO_TF_TRANSPOSE_SINKING="0"

**OPENVINO_TF_ENABLE_BATCHING:** 
如果此参数设为 1 且 VAD-M 用作后端，后端引擎会将输入分成多个异步请求，以利用 VAD-M 中的所有设备来提升性能。

示例：

    OPENVINO_TF_ENABLE_BATCHING="1"

**OPENVINO_TF_DUMP_GRAPHS:**
设置此参数将在optimization pass的所有阶段序列化整个图表，并将其保存至当前目录。

示例：

    OPENVINO_TF_DUMP_GRAPHS=1

**OPENVINO_TF_DUMP_CLUSTERS:**
此变量设为 1 将以“.pbtxt”格式序列化所有集群。

示例：

    OPENVINO_TF_DUMP_CLUSTERS=1

**OPENVINO_TF_DISABLE:**
此变量设为 1 将禁用 **OpenVINO™ integration with TensorFlow**。

示例：

    OPENVINO_TF_DISABLE=1

**OPENVINO_TF_MIN_NONTRIVIAL_NODES:**
此变量设置集群中可以存在的最少算子数。如果算子数量小于指定数量，集群将退回至 TensorFlow。默认情况下，该数量根据总图形大小来计算，但不能小于 6，除非手动设置（启用非常小的集群没有任何性能优势）。

示例：

    OPENVINO_TF_MIN_NONTRIVIAL_NODES=10

**OPENVINO_TF_DYNAMIC_FALLBACK**
此变量启用或禁用动态回退功能。 应设置为“0”以禁用，设置为“1”以启用动态回退。 启用后，在运行期间导致错误的集群可以回退到原生 TensorFlow，尽管它们被分配为在 OpenVINO™ 上运行。 默认启用。

示例:

    OPENVINO_TF_DYNAMIC_FALLBACK=0

## GPU 数据精度

Intel<sup>®</sup> 集成 GPU (iGPU) 的默认精度为 FP32。 因此，如果您将后端名称设置为 **'GPU'**，则 iGPU 上的执行将在 FP32 精度上运行。 要将 iGPU 精度更改为 FP16，请使用设备名称 **'GPU_FP16'**。

将 iGPU 精度设置为 FP32 的示例：

    openvino_tensorflow.set_backend('GPU')

或

    OPENVINO_TF_BACKEND="GPU"

将 iGPU 精度设置为 FP16 的示例：

    openvino_tensorflow.set_backend('GPU_FP16')

或

    OPENVINO_TF_BACKEND="GPU_FP16"