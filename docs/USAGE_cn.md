# 面向 **OpenVINO™ integration with TensorFlow** 的 API 和环境变量

本文介绍了适用于 **OpenVINO™ integration with TensorFlow** 的 Python API。第一节主要介绍在 TensorFlow 应用中发挥 **OpenVINO™ integration with TensorFlow** 的功能所需的基本 API 和代码行。

## 关于基本功能的 API

如要将 **OpenVINO™ integration with TensorFlow** 安装包添加至 TensorFlow python 应用，需使用以下代码行导入安装包：

    import openvino_tensorflow

默认情况下支持 CPU 后端。您可以使用以下 API 设置其他后端：

    openvino_tensorflow.set_backend('<backend_name>')

支持的后端包括‘CPU'、‘GPU'、‘MYRIAD’和‘VAD-M'。

## 其他 API

如要确定系统上可用的后端，可使用以下 API：

    openvino_tensorflow.list_backends()

如要查看 **OpenVINO™ integration with TensorFlow** 是否已启用，可使用以下 API：

    openvino_tensorflow.is_enabled()

如要获取分配的后端，可使用以下 API：

    openvino_tensorflow.get_backend()

如要启用 TensorFlow 管道和布局阶段以及 **OpenVINO™ integration with TensorFlow** 详细的执行日志，可使用以下 API：

    openvino_tensorflow.start_logging_placement()

如要禁用 TensorFlow 管道和布局阶段以及 **OpenVINO™ integration with TensorFlow** 详细的执行日志，可使用以下 API：

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

## 环境变量

**OPENVINO\_TF\_DISABLE\_DEASSIGN\_CLUSTERS：** 形成集群后，由于某些原因（如集群太小，目标设备不支持某些条件），部分集群可能仍然会返回原生 TensorFlow。如果已设置此变量，集群将不会被删除，而是强行在 OpenVINO™ 后端上运行。这可能会导致性能降低，一定情况下还会导致执行崩溃。

示例：

    OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS="1"

**OPENVINO\_TF\_VLOG\_LEVEL：** 此变量用于打印执行日志。设为 1 将打印极少细节，设为 5 将打印最详细的日志。

示例：

    OPENVINO_TF_VLOG_LEVEL="4"

**OPENVINO\_TF\_LOG\_PLACEMENT：** 如果此变量设为 1，将打印与集群形成和封装相关的日志。

示例：

    OPENVINO_TF_LOG_PLACEMENT="1"

**OPENVINO\_TF\_BACKEND：** 可使用此变量设置后端设备名称。可以设为“CPU”，“GPU”，“MYRIAD”或“VAD-M”。

示例：

    OPENVINO_TF_BACKEND="MYRIAD"

**OPENVINO\_TF\_DISABLED\_OPS：** 使用此变量传递已禁用操作列表。这些操作不考虑进行集群化，而是返回原生 TensorFlow。

示例：

    OPENVINO_TF_DISABLED_OPS="Squeeze,Greater,Gather,Unpack"

**OPENVINO\_TF\_CONSTANT\_FOLDING：** 它将启用/禁用已解析集群上的连续的folding pass（默认禁用）。

示例：

    OPENVINO_TF_CONSTANT_FOLDING="1"

**OPENVINO\_TF\_TRANSPOSE\_SINKING：** 它将启用/禁用已解析集群上的 transpose sinking pass（默认启用）。

示例：

    OPENVINO_TF_TRANSPOSE_SINKING="0"

**OPENVINO\_TF\_ENABLE\_BATCHING：** 如果此参数设为 1 且 VAD-M 用作后端，后端引擎会将输入分成多个异步请求，以利用 VAD-M 中的所有设备来提升性能。

示例：

    OPENVINO_TF_ENABLE_BATCHING="1"

**OPENVINO\_TF\_DUMP\_GRAPHS：** 设置此参数将在optimization pass的所有阶段序列化整个图表。

示例：

    OPENVINO_TF_DUMP_GRAPHS=1

**OPENVINO\_TF\_DUMP\_CLUSTERS：** 此变量设为 1 将以“.pbtxt”格式序列化所有集群。

示例：

    OPENVINO_TF_DUMP_CLUSTERS=1

**OPENVINO\_TF\_DISABLE：** 此变量设为 1 将禁用 **OpenVINO™ integration with TensorFlow**。

示例：

    OPENVINO_TF_DISABLE=1

**OPENVINO\_TF\_MIN\_NONTRIVIAL\_NODES：** 此变量设置集群中可以存在的最少操作数。如果操作数量小于指定数量，集群将退回至 TensorFlow。默认情况下，该数量根据总图形大小来计算，但不能小于 6，除非手动设置（启用非常小的集群没有任何性能优势）。

示例：

    OPENVINO_TF_MIN_NONTRIVIAL_NODES=10