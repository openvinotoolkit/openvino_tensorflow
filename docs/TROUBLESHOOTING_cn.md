[English](./TROUBLESHOOTING.md) | 中文简体
# 模型故障排除提示
如果模型无法在 **OpenVINO™ Integration with TensorFlow** 上运行，或者您遇到性能/准确性问题，请按照以下步骤调试问题。

## 1. 激活日志消息

**OpenVINO™ Integration with TensorFlow** 中有两种类型的日志消息。 第一种将向您展示有关图聚类阶段的步骤和详细信息。 您可以通过设置以下环境变量来激活此日志放置：

    OPENVINO_TF_LOG_PLACEMENT=1

激活日志放置后，您将看到几行关于集群和封装阶段的输出。 要检查集群如何形成的简要摘要，请参阅以“OVTF_SUMMARY”开头的行。 这些输出将提供总体统计信息，例如集群化的原因、集群的节点数量、集群总数、每个集群的节点数量等。

```
OVTF_SUMMARY: Summary of reasons why a pair of edge connected encapsulates did not merge
OVTF_SUMMARY: DEADNESS: 1466, STATICINPUT: 203, PATHEXISTS: 1033
OVTF_SUMMARY: Summary of reasons why a pair of edge connected clusters did not merge
OVTF_SUMMARY: NOTANOP: 173, UNSUPPORTED: 722, DEADNESS: 1466, SAMECLUSTER: 4896, STATICINPUT: 203, PATHEXISTS: 1033

OVTF_SUMMARY: Number of nodes in the graph: 4250
OVTF_SUMMARY: Number of nodes marked for clustering: 3943 (92% of total nodes)
OVTF_SUMMARY: Number of nodes assigned a cluster: 2093 (49% of total nodes)      (53% of nodes marked for clustering) 
OVTF_SUMMARY: Number of ngraph clusters :3
OVTF_SUMMARY: Nodes per cluster: 697.667
OVTF_SUMMARY: Size of nGraph Cluster[0]:        1351
OVTF_SUMMARY: Size of nGraph Cluster[35]:       361
OVTF_SUMMARY: Size of nGraph Cluster[192]:      381
OVTF_SUMMARY: Op_deassigned:  Gather -> 549, Reshape -> 197, Cast -> 183, Const -> 152, StridedSlice -> 107, Shape -> 106, Add -> 98, ZerosLike -> 92, Minimum -> 91, Split -> 90, NonMaxSuppressionV2 -> 90, Mul -> 13, Identity -> 12, Range -> 12, ConcatV2 -> 9, Pack -> 7, Sub -> 7, ExpandDims -> 6, Fill -> 4, Unpack -> 4, Transpose -> 3, Slice -> 3, Greater -> 3, Equal -> 2, Exp -> 2, Less -> 2, Squeeze -> 2, Tile -> 1, Size -> 1, Sigmoid -> 1, TopKV2 -> 1
```

日志的初始行与分配集群有关。 在这个阶段，图中的所有节点都被访问并分配了集群边界。 像下面这样的输出日志意味着相应的一对连续节点被分组到同一个集群中：

    NONCONTRACTION: SAMECLUSTER: Preprocessor/mul/x<Const>[0] -> Preprocessor/mul<Mul>[1]

如下一行说明这些节点不能分组到同一个集群中，因为 OpenVINO™ 不支持第二个节点。

    NONCONTRACTION: UNSUPPORTED: MultipleGridAnchorGenerator/assert_equal/Assert/Assert<Assert>[-1] -> Postprocessor/ExpandDims<ExpandDims>[-1]

除了日志放置，您还可以启用 VLOG 级别以打印运行时执行的详细信息。 VLOG 可以设置为 1 到 5 的任何级别。将 VLOG 设置为 1 将打印最少的细节，将其设置为 5 将打印最详细的日志。 例如，您可以通过设置以下环境变量将 VLOG 级别设置为 1：

    OPENVINO_TF_VLOG_LEVEL=1

这将为 **OpenVINO™ integration with TensorFlow** 上执行的每个集群打印一些详细信息，如下所示：
    OPENVINO_TF_MEM_PROFILE:  OP_ID: 0 Step_ID: 8 Cluster: ovtf_cluster_0 Input Tensors created: 0 MB Total process memory: 1 GB
    OPENVINO_TF_TIMING_PROFILE: OP_ID: 0 Step_ID: 8 Cluster: ovtf_cluster_0 Time-Compute: 10 Function-Create-or-Lookup: 0 Create-and-copy-tensors: 0 Execute: 10

## 2. 转储图/集群

要在聚类化的每个步骤中转储完整图，请设置以下环境变量：

    OPENVINO_TF_DUMP_GRAPHS=1

这将在聚类化的每个步骤中以“pbtxt”格式序列化 TensorFlow 图。

- unmarked_<graph_id>.pbtxt: 这是 **OpenVINO™ Integration with TensorFlow** optimization pass 的初始图。
- marked_<graph_id>.pbtxt: 这是标记了支持的节点后的图表。
- clustered_<graph_id>.pbtxt: 这是聚类化完成后的图，在这一步之后，所有支持的节点都应该被分组到集群中。
- declustered_<graph_id>.pbtxt: 这是一些集群被取消分配后的图表。 例如，在此步骤之后，操作数非常少的集群将被取消分配。
- encapsulated_<graph_id>.pbtxt: 这是封装完成后的图，每个现有集群都应该封装到一个“_nGraphEncapsulate”操作中。

OpenVINO™ 中间表示 (IR) 文件（“ovtf_cluster_<cluster_id>.xml”和“ovtf_cluster_<cluster_id>.bin”）将针对每个创建的集群进行序列化。

此外，可以通过设置以下环境变量来序列化每个集群的 TensorFlow 图以进行进一步调试：

    OPENVINO_TF_DUMP_CLUSTERS=1

这将为生成的每个集群生成一个文件（“ovtf_cluster_<cluster_id>.pbtxt”）。



## 3. 禁用算子
禁用导致问题的节点。 如果您能够确定导致问题的算子类型，则可以尝试禁用该特定类型的算子。 您可以设置环境变量“OPENVINO_TF_DISABLED_OPS”以禁用导致问题的算子（请参见下面的示例）。

    OPENVINO_TF_DISABLED_OPS="Squeeze,Greater,Gather,Unpack"

## 4. 设置集群大小限制
如果有多个集群在执行并且较小的集群导致了问题，您可以设置集群大小限制，该限制将仅在 OpenVINO™ 上执行较大的集群。 这样，较小的集群将在原生 TensorFlow 上执行，您仍然可以享受在 OpenVINO™ 上执行较大集群的性能优势。 您应该通过设置下面的环境变量来设置集群大小限制。 调整最适合您的模型的值（请参见下面的示例）。

    OPENVINO_TF_MIN_NONTRIVIAL_NODES=25

## 5. 为 OpenVINO™ integration with TensorFlow 优化 Keras 模型

一些 Keras 模型可能包含训练操作，这会导致 TensorFlow 生成控制流操作。 由于 OpenVINO™ 不支持控制流操作，因此该图可能被划分为较小的集群。 冻结模型可以移除这些操作，并使用 **OpenVINO™ integration with TensorFlow** 提高整体性能。

下面是使用 Keras API 的 DenseNet121 推理应用程序示例。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121
# Add two lines of code to enable OpenVINO integration with TensorFlow
import openvino_tensorflow
openvino_tensorflow.set_backend("CPU")
.
.
.
model = DenseNet121(weights='imagenet')
.
.
.
# Run the inference using Keras API    
model.predict(input_data)
```

下面是一个示例代码，用于冻结和运行 Keras 模型以进一步优化它，从而使用 **OpenVINO™ integration with TensorFlow** 实现最佳性能。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# Add two lines of code to enable OpenVINO integration with TensorFlow
import openvino_tensorflow
openvino_tensorflow.set_backend("CPU")
.
.
.
model = DenseNet121(weights='imagenet')
.
.
.
# Freeze the model first to achieve the best performance using OpenVINO integration with TensorFlow    
full_model = tf.function(lambda x: self.model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name=model.inputs[0].name))
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
session = tf.compat.v1.Session(graph=frozen_func.graph)
prob_tensor = frozen_func.graph.get_tensor_by_name(full_model.outputs[0].name)
.
.
.
# 在冻结模型上运行推理
session.run(prob_tensor, feed_dict={full_model.inputs[0].name : input_data})
```
