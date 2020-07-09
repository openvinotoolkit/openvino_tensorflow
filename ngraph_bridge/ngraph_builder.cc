/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/dequantize_builder.hpp"
#include "ngraph/builder/numpy_transpose.hpp"
#include "ngraph/builder/quantize_builder.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/experimental/layers/interpolate.hpp"
#include "ngraph/op/util/logical_reduction.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/slice_plan.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_conversions.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_utils.h"

using tensorflow::int32;
using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

static bool VecStrCmp(const std::vector<string>& a,
                      const std::vector<string>& b) {
  return a == b;
}

static Status ValidateInputCount(const Node* op, tensorflow::int32 count) {
  if (op->num_inputs() != count) {
    return errors::InvalidArgument("\"", op->name(), "\" requires ", count,
                                   " input(s), got ", op->num_inputs(),
                                   " instead");
  }
  return Status::OK();
}

static Status ValidateInputCountMin(const Node* op, tensorflow::int32 count) {
  if (op->num_inputs() < count) {
    return errors::InvalidArgument("\"", op->name(), "\" requires at least ",
                                   count, " input(s), got ", op->num_inputs(),
                                   " instead");
  }
  return Status::OK();
}
//
// Helper for storing ops in ng_op_map.
// For most of the cases, op would have one output so
// vector ng_op_map[op_name] would contain one element.
//
// If storing more than one output_nodes, make sure it's in
// the same order as tensorflow would do that.
//
// Parameters:
//    Builder::OpMap& ng_op_map        - The TF-to-nGraph op map.
//    std::string op_name              - Name of the op.
//
//    shared_ptr<ng::Node> output_node - ng::Node to store
//

static void SaveNgOp(Builder::OpMap& ng_op_map, const std::string& op_name,
                     const shared_ptr<ng::Node>& output_node) {
  // no need to try-catch, map[key] will create vector object
  // if not exists
  ng_op_map[op_name].push_back(output_node);
}

void Builder::SetTracingInfo(const std::string& op_name,
                             const shared_ptr<ng::Node> ng_node) {
  ng_node->set_friendly_name(op_name);
  ng_node->add_provenance_tag(op_name);
  if (config::IsLoggingPlacement()) {
    cout << "TF_to_NG: " << op_name << " --> " << ng_node->get_name() << "\n";
  }
}

template <class TOpType, class... TArg>
std::shared_ptr<TOpType> ConstructNgNode(const std::string& op_name,
                                         TArg&&... Args) {
  auto ng_node = std::make_shared<TOpType>(std::forward<TArg>(Args)...);
  Builder::SetTracingInfo(op_name, ng_node);
  return ng_node;
}

// Helper for fetching correct input node from ng_op_map.
// Handles edge checking to make sure correct input node is
// fetched.
//
// Reduces some boilerplate code (incorrect from now) like this:
//
//      Node* tf_input;
//      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
//
//      shared_ptr<ng::Node> ng_input;
//      try {
//        ng_input = ng_op_map.at(tf_input->name());
//      } catch (const std::out_of_range&) {
//        return errors::NotFound(tf_input->name(),
//                                    " is not found in the ng_op_map");
//      }
//
// Into 2 lines:
//
//      shared_ptr<ng::node> ng_input;
//      TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input))
//
//
//
// Parameters:
//    Builder::OpMap& ng_op_map     - The TF-to-nGraph op map.
//    Node* op                  - TF op being translated.
//    input_idx                     - index of input
//
//    shared_ptr<ng::Node> *result  - ng::Node pointer where result
//                                    will be written
//
//

static Status GetInputNode(const Builder::OpMap& ng_op_map, const Node* op,
                           size_t input_idx, shared_ptr<ng::Node>* result) {
  // input op may have resulted in more than one ng::Node (eg. Split)
  // we need to look at Edge to check index of the input op
  std::vector<const Edge*> edges;
  TF_RETURN_IF_ERROR(op->input_edges(&edges));
  size_t src_output_idx;
  try {
    src_output_idx = edges.at(input_idx)->src_output();
  } catch (const out_of_range&) {
    return Status(error::NOT_FOUND, "Edge not found");
  }

  Node* tf_input;
  TF_RETURN_IF_ERROR(op->input_node(input_idx, &tf_input));
  const std::vector<shared_ptr<ng::Node>>* ng_op = nullptr;
  try {
    ng_op = &ng_op_map.at(tf_input->name());
  } catch (const out_of_range&) {
    return Status(error::NOT_FOUND,
                  string("Ngraph op not found for ") + tf_input->name());
  }
  try {
    *result = ng_op->at(src_output_idx);
  } catch (const out_of_range&) {
    return Status(error::NOT_FOUND, string("Input node not found at index ") +
                                        to_string(src_output_idx));
  }
  return Status::OK();
}

namespace detail {
static Status GetInputNodes(const Builder::OpMap&, const Node*, size_t) {
  return Status::OK();
}

template <typename... Arguments>
static Status GetInputNodes(const Builder::OpMap& ng_op_map, const Node* op,
                            size_t index, shared_ptr<ng::Node>* result,
                            Arguments&&... remaining) {
  if (result != nullptr) {
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, index, result));
  }
  return GetInputNodes(ng_op_map, op, index + 1, remaining...);
}
}  // namespace detail

template <typename... Arguments>
static Status GetInputNodes(const Builder::OpMap& ng_op_map, const Node* op,
                            Arguments&&... remaining) {
  constexpr size_t args_len = sizeof...(Arguments);
  TF_RETURN_IF_ERROR(ValidateInputCount(op, args_len));
  return detail::GetInputNodes(ng_op_map, op, 0, remaining...);
}

static Status GetStaticNodeTensor(
    const Node* node, const std::vector<const Tensor*>& static_input_map,
    Tensor* result) {
  if (node->IsArg()) {
    int arg_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &arg_index));
    const Tensor* source_tensor = static_input_map[arg_index];
    if (source_tensor == nullptr) {
      return errors::Internal(
          "GetStaticNodeTensor called on _Arg but input tensor is missing from "
          "static input map");
    }
    *result = *source_tensor;
    return Status::OK();
  } else if (node->type_string() == "Const") {
    if (!result->FromProto(node->def().attr().at("value").tensor())) {
      return errors::Internal(
          "GetStaticNodeTensor: Const tensor proto parsing failed");
    }
    return Status::OK();
  } else {
    return errors::Internal("GetStaticNodeTensor called on node with type ",
                            node->type_string(), "; _Arg or Const expected");
  }
}

template <typename Ttensor, typename Tvector>
static void ConvertTensorDataToVector(const Tensor& tensor,
                                      std::vector<Tvector>* vector) {
  const Ttensor* data = tensor.flat<Ttensor>().data();
  vector->resize(tensor.NumElements());
  for (int64 i = 0; i < tensor.NumElements(); i++) {
    (*vector)[i] = Tvector(data[i]);
  }
}

template <typename T>
static Status TensorDataToVector(const Tensor& tensor, std::vector<T>* vector) {
  DataType dt = tensor.dtype();

  // If dt and T match, we can just copy.
  if (dt == DataTypeToEnum<T>::value) {
    *vector = std::vector<T>(tensor.flat<T>().data(),
                             tensor.flat<T>().data() + tensor.NumElements());
  }
  // Else we have to convert.
  else {
    switch (dt) {
      case DT_FLOAT:
        ConvertTensorDataToVector<float, T>(tensor, vector);
        break;
      case DT_DOUBLE:
        ConvertTensorDataToVector<double, T>(tensor, vector);
        break;
      case DT_INT8:
        ConvertTensorDataToVector<int8, T>(tensor, vector);
        break;
      case DT_INT16:
        ConvertTensorDataToVector<int16, T>(tensor, vector);
        break;
      case DT_INT32:
        ConvertTensorDataToVector<int32, T>(tensor, vector);
        break;
      case DT_INT64:
        ConvertTensorDataToVector<int64, T>(tensor, vector);
        break;
      case DT_UINT8:
        ConvertTensorDataToVector<uint8, T>(tensor, vector);
        break;
      case DT_UINT16:
        ConvertTensorDataToVector<uint16, T>(tensor, vector);
        break;
      case DT_UINT32:
        ConvertTensorDataToVector<uint32, T>(tensor, vector);
        break;
      case DT_UINT64:
        ConvertTensorDataToVector<uint64, T>(tensor, vector);
        break;
      case DT_BOOL:
        ConvertTensorDataToVector<bool, T>(tensor, vector);
        break;
      default:
        return errors::Internal("TensorDataToVector: tensor has element type ",
                                DataType_Name(dt), ", vector has type ",
                                DataType_Name(DataTypeToEnum<T>::value),
                                "; don't know how to convert");
    }
  }
  return Status::OK();
}

template <typename T>
static Status GetStaticInputVector(
    const Node* op, int64 input_index,
    const std::vector<const Tensor*>& static_input_map,
    std::vector<T>* vector) {
  Node* input_node;
  TF_RETURN_IF_ERROR(op->input_node(input_index, &input_node));
  Tensor input_tensor;
  TF_RETURN_IF_ERROR(
      GetStaticNodeTensor(input_node, static_input_map, &input_tensor));
  TF_RETURN_IF_ERROR(TensorDataToVector(input_tensor, vector));
  return Status::OK();
}

// Helper for Builder::TranslateGraph ("Const" op)
template <typename T, typename VecT = T>
static Status MakeConstOp(const Node* op, ng::element::Type et,
                          std::shared_ptr<ng::Node>* ng_node) {
  vector<VecT> const_values;
  TensorShapeProto shape_proto;

  TF_RETURN_IF_ERROR(
      ValuesFromConstNode<T, VecT>(op->def(), &shape_proto, &const_values));

  TensorShape const_shape(shape_proto);

  ng::Shape ng_shape;
  TF_RETURN_IF_ERROR(TFTensorShapeToNGraphShape(const_shape, &ng_shape));

  *ng_node = ConstructNgNode<ng::opset3::Constant>(op->name(), et, ng_shape,
                                                   const_values);
  return Status::OK();
}

const std::map<DataType,
               std::pair<std::function<Status(const Node*, ng::element::Type,
                                              std::shared_ptr<ng::Node>*)>,
                         const ngraph::element::Type>>&
Builder::TF_NGRAPH_CONST_MAP() {
  static const std::map<
      DataType, std::pair<std::function<Status(const Node*, ng::element::Type,
                                               std::shared_ptr<ng::Node>*)>,
                          const ngraph::element::Type>>
      the_map = {
          {DataType::DT_FLOAT, make_pair(MakeConstOp<float>, ng::element::f32)},
          {DataType::DT_DOUBLE,
           make_pair(MakeConstOp<double>, ng::element::f64)},
          {DataType::DT_INT8, make_pair(MakeConstOp<int8>, ng::element::i8)},
          {DataType::DT_INT16, make_pair(MakeConstOp<int16>, ng::element::i16)},
          {DataType::DT_QINT8, make_pair(MakeConstOp<qint8>, ng::element::i8)},
          {DataType::DT_QUINT16,
           make_pair(MakeConstOp<quint8>, ng::element::u8)},
          {DataType::DT_INT32, make_pair(MakeConstOp<int32>, ng::element::i32)},
          {DataType::DT_INT64, make_pair(MakeConstOp<int64>, ng::element::i64)},
          {DataType::DT_UINT8, make_pair(MakeConstOp<uint8>, ng::element::u8)},
          {DataType::DT_UINT16,
           make_pair(MakeConstOp<uint16>, ng::element::u16)},
          {DataType::DT_BOOL,
           make_pair(MakeConstOp<bool, char>, ng::element::boolean)}};
  return the_map;
}

std::pair<std::shared_ptr<ng::Node>, std::shared_ptr<ng::Node>>
Builder::PerformNgBroadcast(const string& prov_tag,
                            std::shared_ptr<ng::Node> ng_lhs,
                            std::shared_ptr<ng::Node> ng_rhs) {
  // builder::numpy_broadcast is the only known builder that has the possibility
  // that the output node is same as the input node
  // So we take special care to check, before calling SetTracingInfo
  std::shared_ptr<ng::Node> ng_lhs_new, ng_rhs_new;
  std::tie(ng_lhs_new, ng_rhs_new) =
      ng::builder::numpy_broadcast(std::make_pair(ng_lhs, ng_rhs));
  if (ng_lhs_new != ng_lhs) {
    Builder::SetTracingInfo(prov_tag, ng_lhs_new);
  }
  if (ng_rhs_new != ng_rhs) {
    Builder::SetTracingInfo(prov_tag, ng_rhs_new);
  }
  return make_pair(ng_lhs_new, ng_rhs_new);
}

ng::AxisSet ConvertMaskToAxes(const int mask) {
  ng::AxisSet axes{};
  for (auto i = 0; i < sizeof(int) * 8; ++i) {
    if ((unsigned char)(mask >> i & 0x01) == 1) {
      axes.emplace(i);
    }
  }
  return axes;
};

struct strided_slice_mask_attrs {
  int begin;
  int end;
  int new_axis;
  int shrink_axis;
  int ellipsis;
};

static Status GetStridedSliceAttrs(const Node* op,
                                   strided_slice_mask_attrs& attrs) {
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "begin_mask", &attrs.begin));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "end_mask", &attrs.end));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "new_axis_mask", &attrs.new_axis));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "shrink_axis_mask", &attrs.shrink_axis));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "ellipsis_mask", &attrs.ellipsis));

  NGRAPH_VLOG(5) << "Strided Slice Mask Attributes: "
                 << "  begin mask: " << attrs.begin
                 << "  end mask: " << attrs.end
                 << "  new axis mask: " << attrs.new_axis
                 << "  shrink axis mask: " << attrs.shrink_axis
                 << "  ellipsis mask: " << attrs.ellipsis;

  return Status::OK();
}

ng::SlicePlan GetSlicePlan(const ng::Shape& shape,
                           const std::vector<int64_t>& begin,
                           const std::vector<int64_t>& end,
                           const std::vector<int64_t>& stride,
                           const strided_slice_mask_attrs& mask_attrs) {
  ng::SlicePlan slice_plan = ng::make_slice_plan(
      shape, begin, end, stride, ConvertMaskToAxes(mask_attrs.begin),
      ConvertMaskToAxes(mask_attrs.end), ConvertMaskToAxes(mask_attrs.new_axis),
      ConvertMaskToAxes(mask_attrs.shrink_axis),
      ConvertMaskToAxes(mask_attrs.ellipsis));

  // To handle cases like x[2:2], where shape(x) = [1],
  // TF returns shape = [0], empty vector
  // make_slice_plan returns begin=2, end=2, but that is > 1
  // So must clamp them
  // Another example:
  // for dimension 3, Also 2:3:-1 gives 4:4, which will also fail if we try to
  // construct slice. So must clamp to 2:2 etc
  auto clamp = [](int64_t x, int64_t min, int64_t max) {
    return x > max ? max : (x < min ? min : x);
  };
  for (int i = 0; i < shape.size(); i++) {
    slice_plan.begins[i] = clamp(slice_plan.begins[i], 0, shape[i]);
    slice_plan.ends[i] = clamp(slice_plan.ends[i], 0, shape[i]);
  }

  NGRAPH_VLOG(5) << "Slice_plan: begin: " << ng::join(slice_plan.begins)
                 << ", end: " << ng::join(slice_plan.ends)
                 << ", stride: " << ng::join(slice_plan.strides)
                 << ", reshape input shape: " << slice_plan.reshape_in_shape
                 << ", reshape output shape: " << slice_plan.reshape_out_shape
                 << ", reverse axis: " << slice_plan.reverse_axes;

  return slice_plan;
}

// Helper function to translate a unary op.
//
// Parameters:
//
//    Node* op                   - TF op being translated. Must have one input.
//    const std::vector<const Tensor*>& static_input_map
//                               - the static input map
//    Builder::OpMap& ng_op_map  - The TF-to-nGraph op map.
//
//    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>>
//      create_unary_op           - Function to construct the graph implementing
//                                 the unary op, given the input to the unop
//                                 as an argument.
//
// Example usage:
//
//  if (n->type_string == "Square") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp(n, static_input_map, ng_op_map,
//                       [] (std::shared_ptr<ng::Node> n) {
//                           return
//                           (std::make_shared<ng::opset3::Multiply>(n,n));
//                       });
//  }
static Status TranslateUnaryOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>)>
        create_unary_op) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));
  auto ng_node = create_unary_op(ng_input);
  if (ng_node != ng_input) {
    Builder::SetTracingInfo(op->name(), ng_node);
  }
  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

// Helper function to translate a unary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Abs") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp<ng::op::Abs>(n, static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
static Status TranslateUnaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(op, static_input_map, ng_op_map,
                          [&op](std::shared_ptr<ng::Node> n) {
                            return ConstructNgNode<T>(op->name(), n);
                          });
}

// Helper function to translate a binary op
// Parameters:
//
//    Node* op               - TF op being translated. Must have only two
//    inputs.
//    const std::vector<const Tensor*>& static_input_map - the static input map
//    Builder::OpMap& ng_op_map  - The TF-to-nGraph op map.
//    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>,
//    std::shared_ptr<ng::Node>)>
//    create_binary_op           - Function to construct the graph implementing
//                                 the binary op, given the 2 ng_inputs to the
//                                 binaryop
// Example Usage:
//
// if (op->type_string() == "SquaredDifference") {
//      TF_RETURN_IF_ERROR(TranslateBinaryOp(op, ng_op_map,
//         [](std::shared_ptr<ng::Node> ng_input1, std::shared_ptr<ng::Node>
//         ng_input2) {
//           auto ng_diff = std::make_shared<ng::opset3::Subtract>(input1,
//           input2);
//           return std::make_shared<ng::opset3::Multiply>(ng_diff,ng_diff);
//         }));
//    }
//

static Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>,
                                            std::shared_ptr<ng::Node>)>
        create_binary_op) {
  std::shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));
  auto ng_node = create_binary_op(ng_lhs, ng_rhs);
  if (ng_node != ng_lhs && ng_node != ng_rhs) {
    Builder::SetTracingInfo(op->name(), ng_node);
  }

  SaveNgOp(ng_op_map, op->name(), ng_node);

  return Status::OK();
}

// Helper function to translate a binary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Add") {
//    TF_RETURN_IF_ERROR(TranslateBinaryOp<ng::opset3::Add>(op,
//    static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
static Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateBinaryOp(
      op, static_input_map, ng_op_map, [&op](std::shared_ptr<ng::Node> ng_lhs,
                                             std::shared_ptr<ng::Node> ng_rhs) {
        return ConstructNgNode<T>(op->name(), ng_lhs, ng_rhs);
      });
}

// Helper function for translating QuantizedAvgPool and QuantizedMaxPool
static Status TranslateQuantizedPoolOp(const Node* op,
                                       const std::vector<const Tensor*>&,
                                       Builder::OpMap& ng_op_map,
                                       std::string pooling_name) {
  bool is_quantizedAvgPool = pooling_name == "QuantizedAvgPool";
  bool is_quantizedMaxPool = pooling_name == "QuantizedMaxPool";

  if (!(is_quantizedAvgPool || is_quantizedMaxPool)) {
    return errors::InvalidArgument(
        "Expected quantized pooling type node to be ScaledQuantizedAvgPool or "
        "ScaledQuantizedMaxPool");
  }
  shared_ptr<ng::Node> ng_input, ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_min, &ng_max));
  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;

  bool is_nhwc = true;  // The input data format is always NHWC
  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_input->get_output_shape(0),
                         ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  // Creating and passing dummy nodes to quantized pool operation because it
  // does
  // not use them. If it ever starts using min/max, the dummy min-max would
  // cause it to fail
  shared_ptr<ng::Node> dummy_min(nullptr), dummy_max(nullptr);

  std::shared_ptr<ng::Node> ng_quant_pool;
  if (is_quantizedAvgPool) {
    // QuantizeAvgPool
    // TF doesn't include padding in avg calculation
    ng_quant_pool = ConstructNgNode<ng::op::AvgPool>(
        op->name(), ng_input, ng_kernel_shape, ng_strides, ng_padding_below,
        ng_padding_above, false);
  } else {
    // QuantizeMaxPool
    ng_quant_pool = ConstructNgNode<ng::op::MaxPool>(
        op->name(), ng_input, ng_kernel_shape, ng_strides, ng_padding_below,
        ng_padding_above);
  }
  Builder::SetTracingInfo(op->name(), ng_quant_pool);

  BatchToTensorflow(op->name(), is_nhwc, ng_quant_pool);
  SaveNgOp(ng_op_map, op->name(), ng_quant_pool);
  // For QuantizedAvgPool and QuantizedMaxPool input min-max remains unchanged
  // and is just propagated along
  // https://github.com/tensorflow/tensorflow/blob/9590c4c32dd4346ea5c35673336f5912c6072bf2/tensorflow/core/kernels/quantized_pooling_ops.cc#L99
  SaveNgOp(ng_op_map, op->name(), ng_min);
  SaveNgOp(ng_op_map, op->name(), ng_max);
  return Status::OK();
}

static Status TranslateAddNOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  std::vector<shared_ptr<ng::Node>> ng_arg_vec(op->num_inputs());

  for (int inp_idx = 0; inp_idx < op->num_inputs(); inp_idx++)
    TF_RETURN_IF_ERROR(
        GetInputNode(ng_op_map, op, inp_idx, &ng_arg_vec[inp_idx]));

  SaveNgOp(
      ng_op_map, op->name(),
      std::accumulate(
          std::next(ng_arg_vec.begin()), ng_arg_vec.end(), ng_arg_vec.at(0),
          [&op](shared_ptr<ng::Node> a, shared_ptr<ng::Node> b) {
            return ConstructNgNode<ng::opset3::Add>(op->name(), a, b);
          }));  // accumulation: start with
                // first element. default op is
                // addition
  return Status::OK();
}

template <typename T>
static Status TranslateArgMinMaxOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  bool is_argmin = std::is_same<T, ng::op::ArgMin>::value;
  bool is_argmax = std::is_same<T, ng::op::ArgMax>::value;
  if (!(is_argmin || is_argmax)) {
    return errors::InvalidArgument("Expected node to be argmin or argmax type");
  }

  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input));

  std::vector<int64> tf_dim;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &tf_dim));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  if (tf_dim.size() != 1) {
    return errors::InvalidArgument(
        (is_argmin ? "ArgMin" : "ArgMax"),
        " Op: dimension must be scalar, operates on a single axis");
  }

  // If input dimension is negative, make it positive
  if (tf_dim[0] < 0) {
    tf_dim[0] = (int64)input_rank + tf_dim[0];
  }
  size_t input_dims = tf_dim[0];

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "output_type", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<T>(op->name(), ng_input, input_dims, ng_et));
  return Status::OK();
}

static Status TranslateAvgPoolOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "AvgPool data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(op->name(), is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for AvgPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};
  ng::op::PadType ng_pad_type = ng::op::PadType::EXPLICIT;
  if (tf_padding_type == "VALID") {
    ng_pad_type = ng::op::PadType::VALID;
  } else {
    Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_padding_below, ng_padding_above);
  }

  std::shared_ptr<ng::Node> ng_avgpool = ConstructNgNode<ng::opset3::AvgPool>(
      op->name(), ng_input, ng_strides, ng_padding_below, ng_padding_above,
      ng_kernel_shape, true, ng::op::RoundingType::FLOOR, ng_pad_type);

  BatchToTensorflow(op->name(), is_nhwc, ng_avgpool);
  NGRAPH_VLOG(3) << "avgpool outshape: {" << ng::join(ng_avgpool->get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_avgpool);
  return Status::OK();
}

static Status TranslateBatchMatMulOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));

  std::string backend_name;
  TF_RETURN_IF_ERROR(ngraph_bridge::GetNodeBackend(op, &backend_name));

  auto ng_lhs_shape = ng_lhs->get_shape();
  auto ng_rhs_shape = ng_rhs->get_shape();

  if (ng_lhs_shape.size() != ng_rhs_shape.size()) {
    return errors::InvalidArgument(
        "Dimensions of two input args are not the same for BatchMatMul. Left "
        "shape is ",
        ng::join(ng_lhs_shape), " of rank ", ng_lhs_shape.size(),
        " and Right shape is ", ng::join(ng_rhs_shape), " of rank ",
        ng_rhs_shape.size());
  }
  size_t n_dims = ng_lhs_shape.size();
  if (n_dims < 2) {
    return errors::InvalidArgument(
        "Dimensions of input args for BatchMatMul must be >=2 but is ", n_dims);
  }

  for (size_t i = 0; i < n_dims - 2; ++i) {
    if (ng_lhs_shape[i] != ng_rhs_shape[i]) {
      return errors::InvalidArgument(
          "ng_lhs_shape and ng_rhs_shape must be the same for BatchMatMul "
          "for each dimension but found ",
          i, "th dimension different. Left shape is ", ng::join(ng_lhs_shape),
          "and Right shape is ", ng::join(ng_rhs_shape));
    }
  }

  bool tf_adj_x = false;
  bool tf_adj_y = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "adj_x", &tf_adj_x));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "adj_y", &tf_adj_y));

  if (n_dims == 2) {
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<ngraph::op::MatMul>(op->name(), ng_lhs, ng_rhs,
                                                 tf_adj_x, tf_adj_y));
  } else if (n_dims == 3) {
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<ngraph::op::BatchMatMulTranspose>(
                 op->name(), ng_lhs, ng_rhs, tf_adj_x, tf_adj_y));
  } else {
    ng::AxisVector out_axes(n_dims);
    std::iota(out_axes.begin(), out_axes.end(), 0);

    size_t compound_size = 1;
    for (size_t i = 0; i < n_dims - 2; i++) {
      compound_size *= ng_lhs_shape[i];
    }

    ng::Shape tmp_lhs_shape = {compound_size, ng_lhs_shape[n_dims - 2],
                               ng_lhs_shape[n_dims - 1]};
    ng::Shape tmp_rhs_shape = {compound_size, ng_rhs_shape[n_dims - 2],
                               ng_rhs_shape[n_dims - 1]};

    auto output_shape = ng_lhs_shape;
    output_shape[n_dims - 2] = ng_lhs_shape[n_dims - (tf_adj_x ? 1 : 2)];
    output_shape[n_dims - 1] = ng_rhs_shape[n_dims - (tf_adj_y ? 2 : 1)];
    ng::AxisVector tmp_axes = {0, 1, 2};

    std::shared_ptr<ng::Node> lhs_reshape =
        ConstructNgNode<ngraph::op::Reshape>(op->name(), ng_lhs, out_axes,
                                             tmp_lhs_shape);
    std::shared_ptr<ng::Node> rhs_reshape =
        ConstructNgNode<ngraph::op::Reshape>(op->name(), ng_rhs, out_axes,
                                             tmp_rhs_shape);
    std::shared_ptr<ng::Node> batchmatmul_transpose =
        ConstructNgNode<ngraph::op::BatchMatMulTranspose>(
            op->name(), lhs_reshape, rhs_reshape, tf_adj_x, tf_adj_y);
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<ngraph::op::Reshape>(
                 op->name(), batchmatmul_transpose, tmp_axes, output_shape));
  }
  return Status::OK();
}

static Status TranslateBatchMatMulV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));

  auto ng_lhs_shape = ng_lhs->get_shape();
  auto ng_rhs_shape = ng_rhs->get_shape();
  size_t n_dims = ng_lhs_shape.size();

  if (ng_lhs_shape.size() != ng_rhs_shape.size()) {
    return errors::InvalidArgument(
        "Dimensions of two input args are not the same for BatchMatMul. Left "
        "shape is ",
        ng::join(ng_lhs_shape), " of rank ", ng_lhs_shape.size(),
        " and Right shape is ", ng::join(ng_rhs_shape), " of rank ",
        ng_rhs_shape.size());
  }

  for (size_t i = 0; i < n_dims - 2; ++i) {
    if (!((ng_lhs_shape[i] == ng_rhs_shape[i]) || (ng_lhs_shape[i] == 1) ||
          (ng_rhs_shape[i] == 1))) {
      return errors::InvalidArgument(
          "ng_lhs_shape and ng_rhs_shape must be broadcastable for "
          "BatchMatMulV2 "
          "for each dimension. Failed to match dimension ",
          i, " where lhs was ", ng_lhs_shape[i], " and rhs was ",
          ng_rhs_shape[i]);
    } else if (ng_lhs_shape[i] != ng_lhs_shape[i]) {
      return errors::Unimplemented(
          "ng_lhs_shape and ng_rhs_shape must be the same for each dimension, "
          "for current implementation of BatchMatMulV2. "
          "Failed to match dimension ",
          i, " where lhs was ", ng_lhs_shape[i], " and rhs was ",
          ng_rhs_shape[i]);
    }
  }
  return TranslateBatchMatMulOp(op, static_input_map, ng_op_map);
}

static Status TranslateBiasAddOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_bias;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_bias));

  std::string tf_data_format;
  if (GetNodeAttr(op->attrs(), "data_format", &tf_data_format) !=
      Status::OK()) {
    tf_data_format = "NHWC";
  }

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "BiasAdd data format is neither NHWC nor NCHW");
  }

  auto ng_input_shape = ng_input->get_shape();
  auto ng_bias_shape = ng_bias->get_shape();
  if (ng_bias_shape.size() != 1) {
    return errors::InvalidArgument(
        "Bias argument to BiasAdd does not have one dimension");
  }

  // We'll choose reshape over broadcast
  // Reshape the bias to (1, C, 1, ...) if input is channels-first.
  shared_ptr<ng::Node> ng_bias_reshaped = ng_bias;
  if (tf_data_format == "NCHW") {
    auto channel_dim = ng_input_shape[1];
    std::vector<int64> target_shape(ng_input_shape.size());
    for (int64_t i = 0; i < ng_input_shape.size(); i++) {
      if (i == 1) {
        target_shape[i] = channel_dim;
      } else {
        target_shape[i] = 1;
      }
    }
    auto target_shape_node = make_shared<ng::opset3::Constant>(
        ng::element::i64, ng::Shape{ng_input_shape.size()}, target_shape);
    ng_bias_reshaped = ConstructNgNode<ng::opset3::Reshape>(
        op->name(), ng_bias, target_shape_node, false);
  }

  shared_ptr<ng::Node> ng_add =
      ConstructNgNode<ng::opset3::Add>(op->name(), ng_input, ng_bias_reshaped);

  SaveNgOp(ng_op_map, op->name(), ng_add);
  return Status::OK();
}

static Status TranslateCastOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "DstT", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  try {
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<ng::opset3::Convert>(op->name(), ng_input, ng_et));
  } catch (const std::out_of_range&) {
    return errors::Unimplemented("Failed to convert TF data type: ",
                                 DataType_Name(dtype));
  }
  return Status::OK();
}

static Status TranslateConcatV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 2));

  std::vector<int64> tf_concat_axis_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(
      op, op->num_inputs() - 1, static_input_map, &tf_concat_axis_vec));

  int64 concat_axis = tf_concat_axis_vec[0];

  if (concat_axis < 0) {
    shared_ptr<ng::Node> ng_first_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_first_arg));

    concat_axis += int64(ng_first_arg->get_shape().size());
  }

  ng::NodeVector ng_args;

  for (int i = 0; i < op->num_inputs() - 1; i++) {
    shared_ptr<ng::Node> ng_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, &ng_arg));
    ng_args.push_back(ng_arg);
  }

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::opset3::Concat>(op->name(), ng_args,
                                               size_t(concat_axis)));
  return Status::OK();
}

static Status TranslateConstOp(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dtype", &dtype));

  std::shared_ptr<ng::Node> ng_node;

  // For some reason the following do not work (no specialization of
  // tensorflow::checkpoint::SavedTypeTraits...)
  // case DataType::DT_UINT32:
  //   TF_RETURN_IF_ERROR(MakeConstOp<uint32>(op, ng::element::u32,
  //   &ng_node));
  //   break;
  // case DataType::DT_UINT64:
  //   TF_RETURN_IF_ERROR(MakeConstOp<uint64>(op, ng::element::u64,
  //   &ng_node));
  //   break;
  try {
    const auto& func_param = Builder::TF_NGRAPH_CONST_MAP().at(dtype);
    TF_RETURN_IF_ERROR(func_param.first(op, func_param.second, &ng_node));
  } catch (const std::out_of_range&) {
    return errors::Unimplemented("Failed to translate Constant with TF type: ",
                                 DataType_Name(dtype));
  }

  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

static Status TranslateConv2DOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_filter));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2D data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  // TF Kernel Test Checks
  // Strides in the batch and depth dimension is not supported
  if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
    return errors::InvalidArgument(
        "Strides in batch and depth dimensions is not supported: ",
        op->type_string());
  }

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Strides ng_dilations(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);

  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
  BatchToNGraph(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};

  ng::op::PadType ng_pad_type = ng::op::PadType::EXPLICIT;
  if (tf_padding_type == "VALID") {
    ng_pad_type = ng::op::PadType::VALID;
  } else {
    Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_dilations, ng_padding_below,
                         ng_padding_above);
  }

  std::shared_ptr<ng::Node> ng_conv = ConstructNgNode<ng::opset3::Convolution>(
      op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
      ng_padding_above, ng_dilations, ng_pad_type);

  BatchToTensorflow(op->name(), is_nhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateConv2DBackpropInputOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_filter, ng_out_backprop;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, nullptr, &ng_filter, &ng_out_backprop));

  // TODO: refactor me to be less redundant with other convolution ops
  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2DBackpropInput data format is neither NHWC nor NCHW: %s",
        tf_data_format);
  }

  std::vector<int64> tf_input_sizes;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 0, static_input_map, &tf_input_sizes));

  if (std::any_of(tf_input_sizes.begin(), tf_input_sizes.end(),
                  [](int32 size) { return size <= 0; })) {
    return errors::InvalidArgument(
        "Conv2DBackpropInput input sizes must be positive integers");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Strides ng_dilations(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);
  ng::Shape ng_batch_shape(4);

  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, tf_input_sizes, ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
  BatchToNGraph(op->name(), is_nhwc, ng_out_backprop);
  if (is_nhwc) {
    ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                      static_cast<unsigned long>(tf_input_sizes[3]),
                      static_cast<unsigned long>(tf_input_sizes[1]),
                      static_cast<unsigned long>(tf_input_sizes[2])};
  } else {
    ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                      static_cast<unsigned long>(tf_input_sizes[1]),
                      static_cast<unsigned long>(tf_input_sizes[2]),
                      static_cast<unsigned long>(tf_input_sizes[3])};
  }

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};

  ng::op::PadType ng_pad_type = ng::op::PadType::EXPLICIT;
  if (tf_padding_type == "VALID") {
    ng_pad_type = ng::op::PadType::VALID;
  } else {
    Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_dilations, ng_padding_below,
                         ng_padding_above);
  }

  auto ng_output_shape = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::i64, ng::Shape{ng_batch_shape.size() - 2},
      vector<size_t>(ng_batch_shape.begin() + 2, ng_batch_shape.end()));

  std::shared_ptr<ng::Node> ng_data =
      ConstructNgNode<ng::opset3::ConvolutionBackpropData>(
          op->name(), ng_out_backprop, ng_filter, ng_output_shape, ng_strides,
          ng_padding_below, ng_padding_above, ng_dilations, ng_pad_type);

  BatchToTensorflow(op->name(), is_nhwc, ng_data);

  SaveNgOp(ng_op_map, op->name(), ng_data);
  return Status::OK();
}

// Translate Conv3D Op
static Status TranslateConv3DOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_filter));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NDHWC" && tf_data_format != "NCDHW") {
    return errors::InvalidArgument(
        "Conv3D data format is neither NDHWC nor NCDHW");
  }

  bool is_ndhwc = (tf_data_format == "NDHWC");

  // TODO: in 3D
  // TF Kernel Test Checks
  // // Strides in the batch and depth dimension is not supported
  // if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
  //   return errors::InvalidArgument(
  //       "Strides in batch and depth dimensions is not supported: ",
  //       op->type_string());
  // }

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(3);
  ng::Strides ng_dilations(3);
  ng::Shape ng_image_shape(3);
  ng::Shape ng_kernel_shape(3);

  BatchedOpParam3DToNGraph(is_ndhwc, tf_strides, ng_strides);
  BatchedOpParam3DToNGraph(is_ndhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParam3DToNGraph(is_ndhwc, tf_dilations, ng_dilations);
  BatchToNGraph3D(op->name(), is_ndhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  ng_kernel_shape[2] = ng_filter_shape[2];
  Transpose3D<4, 3, 0, 1, 2>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0, 0};

  ng::op::PadType ng_pad_type = ng::op::PadType::EXPLICIT;
  if (tf_padding_type == "VALID") {
    ng_pad_type = ng::op::PadType::VALID;
  } else {
    Builder::MakePadding3D(tf_padding_type, ng_image_shape, ng_kernel_shape,
                           ng_strides, ng_dilations, ng_padding_below,
                           ng_padding_above);
  }

  std::shared_ptr<ng::Node> ng_conv = ConstructNgNode<ng::opset3::Convolution>(
      op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
      ng_padding_above, ng_dilations, ng_pad_type);

  BatchToTensorflow3D(op->name(), is_ndhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

// Translate TranslateCropAndResizeOp op
static Status TranslateCropAndResizeOp(const Node* op,
                                       const std::vector<const Tensor*>&,
                                       Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> image, boxes, box_ind, crop_size;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &image, &boxes, &box_ind, &crop_size));

  // Get the attributes
  float extrapolation_value;
  std::string method;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "extrapolation_value", &extrapolation_value));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "method", &method));

  ng::op::CropAndResize::ResizeMethod ng_method =
      ng::op::CropAndResize::ResizeMethod::unspecified;
  if (method == "bilinear") {
    ng_method = ng::op::CropAndResize::ResizeMethod::bilinear;
  } else if (method == "nearest") {
    ng_method = ng::op::CropAndResize::ResizeMethod::nearest;
  } else {
    return errors::Internal(
        "Expected crop and resize's interpolation mode to be bilinear or "
        "nearest, but got ",
        extrapolation_value, " in op ", op->name());
  }

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::CropAndResize>(op->name(), image, boxes,
                                                  box_ind, crop_size, ng_method,
                                                  extrapolation_value));
  return Status::OK();
}

static Status TranslateCumsumOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_x, ng_axis;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_x, &ng_axis));
  bool exclusive, reverse;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "exclusive", &exclusive));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "reverse", &reverse));

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::opset3::CumSum>(op->name(), ng_x, ng_axis,
                                               exclusive, reverse));
  return Status::OK();
}

// Translate DepthToSpace op
static Status TranslateDepthToSpaceOp(const Node* op,
                                      const std::vector<const Tensor*>&,
                                      Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  // Get the attributes
  int64 block_size;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "block_size", &block_size));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  ng::Shape input_shape = ng_input->get_shape();
  std::map<std::string, int> format_to_int_map = {
      {"NHWC", 0}, {"NCHW", 1}, {"NCHW_VECT_C", 1}};

  int channel_dimension;
  int num_spatial_dimensions = 2;  // H, W are spatial dimensions

  switch (format_to_int_map[tf_data_format]) {
    // NHWC
    case 0:
      channel_dimension = 3;
      break;
    // NCHW
    case 1:
      channel_dimension = 1;
      break;
    // NCHW_VEC_C
    case 2:
      return errors::InvalidArgument(
          "NCHW_VECT_C is not supported in DepthToSpace for now");
    default:
      return errors::InvalidArgument(
          "DepthToSpace supported data format is NCHW, NHWC, or NCHW_VECT_C");
  }

  // Error checking : depth must be divisible by square of the block_size
  if (input_shape[channel_dimension] % (block_size * block_size) != 0) {
    return errors::InvalidArgument(
        "Input tensor's channel dimension ,", input_shape[channel_dimension],
        " is not divisible by square of the block_size ", block_size);
  }

  ng::AxisVector ng_reshape_shape;
  ng::AxisVector ng_transpose_permutation;
  ng::AxisVector ng_output_shape;

  switch (format_to_int_map[tf_data_format]) {
    // NHWC
    case 0: {
      // ng_reshape_shape = [batch_size,
      //                     height,
      //                     width,
      //                     block_size,
      //                     block_size,
      //                     channel / (block_size * block_size)]
      ng_reshape_shape.push_back(input_shape[0]);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_reshape_shape.push_back(input_shape[i + 1]);
      }

      int64 num_blocks = 1;
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_reshape_shape.push_back(block_size);
        num_blocks *= block_size;
      }
      ng_reshape_shape.push_back(input_shape[channel_dimension] / num_blocks);

      // ng_transpose_shape = [batch_size,
      //                       height,
      //                       block_size,
      //                       width,
      //                       block_size,
      //                       channel / (block_size * block_size)]
      ng_transpose_permutation.push_back(0);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_transpose_permutation.push_back(i + 1);
        ng_transpose_permutation.push_back(i + 1 + num_spatial_dimensions);
      }
      ng_transpose_permutation.push_back(channel_dimension +
                                         num_spatial_dimensions);

      // ng_output_shape = [batch_size,
      //                    height * block_size,
      //                    width * block_size,
      //                    channel / (block_size * block_size)]
      ng_output_shape.push_back(input_shape[0]);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_output_shape.push_back(input_shape[i + 1] * block_size);
      }
      ng_output_shape.push_back(input_shape[channel_dimension] / num_blocks);
      break;
    }  // end of case NHWC

    // NCHW
    case 1: {
      // ng_reshape_shape = [batch_size,
      //                     block_size,
      //                     block_size,
      //                     channel / (block_size * block_size),
      //                     height,
      //                     width]
      int64 num_blocks = 1;
      ng_reshape_shape.push_back(input_shape[0]);  // N dimension
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_reshape_shape.push_back(block_size);
        num_blocks *= block_size;
      }
      ng_reshape_shape.push_back(input_shape[channel_dimension] / num_blocks);

      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_reshape_shape.push_back(input_shape[i + 2]);
      }

      // ng_transpose_shape = [batch_size,
      //                       channel / (block_size * block_size)
      //                       height,
      //                       block_size,
      //                       width,
      //                       block_size]
      ng_transpose_permutation.push_back(0);
      ng_transpose_permutation.push_back(1 + num_spatial_dimensions);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_transpose_permutation.push_back(i + 2 + num_spatial_dimensions);
        ng_transpose_permutation.push_back(i + 1);
      }

      // ng_output_shape = [batch_size,
      //                    channel / (block_size * block_size)
      //                    height * block_size,
      //                    width * block_size]
      ng_output_shape.push_back(input_shape[0]);
      ng_output_shape.push_back(input_shape[channel_dimension] / num_blocks);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_output_shape.push_back(input_shape[i + 2] * block_size);
      }
      break;
    }  // end of case NCHW
  }

  ng::AxisVector ng_axis_order(input_shape.size());
  std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
  auto reshaped = ConstructNgNode<ng::op::Reshape>(
      op->name(), ng_input, ng_axis_order, ng_reshape_shape);

  auto transposed =
      ng::builder::numpy_transpose(reshaped, ng_transpose_permutation);
  Builder::SetTracingInfo(op->name(), transposed);

  ng::AxisVector ng_axis_order_second_reshape(transposed->get_shape().size());
  std::iota(ng_axis_order_second_reshape.begin(),
            ng_axis_order_second_reshape.end(), 0);
  auto final_reshape = ConstructNgNode<ng::op::Reshape>(
      op->name(), transposed, ng_axis_order_second_reshape, ng_output_shape);
  SaveNgOp(ng_op_map, op->name(), final_reshape);

  return Status::OK();
}

static Status TranslateDepthwiseConv2dNativeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_filter));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "DepthwiseConv2D data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Strides ng_dilations(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);

  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
  BatchToNGraph(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  // ng input shape is NCHW
  auto& input_shape = ng_input->get_shape();
  // ng filter shape is OIHW
  auto& filter_shape = ng_filter->get_shape();
  ng::NodeVector ng_args;

  for (size_t i = 0; i < input_shape[1]; i++) {
    const std::vector<size_t> lower_bound{0, i, 0, 0};
    const std::vector<size_t> upper_bound{input_shape[0], i + 1, input_shape[2],
                                          input_shape[3]};
    auto ng_sliced_input = ConstructNgNode<ng::op::Slice>(
        op->name(), ng_input, lower_bound, upper_bound);

    const std::vector<size_t> f_lower_bound{0, i, 0, 0};
    const std::vector<size_t> f_upper_bound{filter_shape[0], i + 1,
                                            filter_shape[2], filter_shape[3]};
    auto ng_sliced_filter = ConstructNgNode<ng::op::Slice>(
        op->name(), ng_filter, f_lower_bound, f_upper_bound);

    NGRAPH_VLOG(3) << "depthwise conv 2d.";
    NGRAPH_VLOG(3) << "sliced shape " << ng::join(ng_sliced_input->get_shape());
    NGRAPH_VLOG(3) << "filter shape "
                   << ng::join(ng_sliced_filter->get_shape());

    auto ng_conv = ConstructNgNode<ng::op::Convolution>(
        op->name(), ng_sliced_input, ng_sliced_filter, ng_strides, ng_dilations,
        ng_padding_below, ng_padding_above);
    ng_args.push_back(ng_conv);
  }

  size_t ng_concatenation_axis = 1;  // channel axis
  std::shared_ptr<ng::Node> ng_concat = ConstructNgNode<ng::op::Concat>(
      op->name(), ng_args, ng_concatenation_axis);

  BatchToTensorflow(op->name(), is_nhwc, ng_concat);
  SaveNgOp(ng_op_map, op->name(), ng_concat);
  return Status::OK();
}

static Status TranslateExpandDimsOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_dim;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_dim));

  std::vector<int64> dim_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &dim_vec));

  if (dim_vec.size() != 1) {
    return errors::InvalidArgument(
        "The size of argument dim is not 1 for ExpandDims");
  }

  auto& shape = ng_input->get_shape();
  if (dim_vec[0] < 0) {
    // allow range [-rank(input) - 1, rank(input)]
    // where -1 append new axis at the end
    dim_vec[0] = shape.size() + dim_vec[0] + 1;
  }
  auto out_shape = shape;
  out_shape.insert(out_shape.begin() + size_t(dim_vec[0]), 1);

  auto ng_shape = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::u64, ng::Shape{out_shape.size()}, out_shape);

  std::shared_ptr<ng::Node> ng_expand_dim =
      ConstructNgNode<ng::opset3::Reshape>(op->name(), ng_input, ng_shape,
                                           false);

  SaveNgOp(ng_op_map, op->name(), ng_expand_dim);
  return Status::OK();
}

static Status TranslateFillOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_value;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, nullptr, &ng_value));

  std::vector<int64> dims_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 0, static_input_map, &dims_vec));

  auto ng_output_shape = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::i64, ng::Shape{dims_vec.size()}, dims_vec);

  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::opset3::Broadcast>(
                                      op->name(), ng_value, ng_output_shape));
  return Status::OK();
}

static Status TranslateFloorDivOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "T", &dtype));
  auto int_types = NGraphIntDTypes();
  std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>,
                                          std::shared_ptr<ng::Node>)>
      ng_bin_fn;
  if (std::find(int_types.begin(), int_types.end(), dtype) != int_types.end()) {
    ng_bin_fn = [&op](std::shared_ptr<ng::Node> ng_input1,
                      std::shared_ptr<ng::Node> ng_input2) {
      return ConstructNgNode<ng::opset3::Divide>(op->name(), ng_input1,
                                                 ng_input2);
    };
  } else {
    ng_bin_fn = [&op](std::shared_ptr<ng::Node> ng_input1,
                      std::shared_ptr<ng::Node> ng_input2) {
      return ConstructNgNode<ng::opset3::Floor>(
          op->name(), ConstructNgNode<ng::opset3::Divide>(op->name(), ng_input1,
                                                          ng_input2));
    };
  }
  return TranslateBinaryOp(op, static_input_map, ng_op_map, ng_bin_fn);
}

// static Status TranslateFloorModOp(
//     const Node* op, const std::vector<const Tensor*>& static_input_map,
//     Builder::OpMap& ng_op_map) {
//   auto ng_floormod = [&op](std::shared_ptr<ng::Node> ng_input1,
//                            std::shared_ptr<ng::Node> ng_input2) {
//     auto floordiv = ConstructNgNode<ng::opset3::Floor>(
//         op->name(),
//         ConstructNgNode<ng::opset3::Divide>(op->name(), ng_input1,
//         ng_input2));
//     return ConstructNgNode<ng::opset3::Subtract>(
//         op->name(), ng_input1,
//         ConstructNgNode<ng::opset3::Multiply>(op->name(), floordiv,
//         ng_input2));
//   };
//   return TranslateBinaryOp(op, static_input_map, ng_op_map, ng_floormod);
// }

static Status TranslateFusedBatchNormOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_scale, ng_offset, ng_mean, ng_variance;
  bool is_v3 = op->type_string() == "FusedBatchNormV3";

  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_scale,
                                   &ng_offset, &ng_mean, &ng_variance));

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2D data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << "data_format: " << tf_data_format;

  float tf_epsilon;
  if (GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon) != Status::OK()) {
    NGRAPH_VLOG(3) << "epsilon attribute not present, setting to 0.0001";
    // TensorFlow default
    tf_epsilon = 0.0001;
  }

  NGRAPH_VLOG(3) << "epsilon: " << tf_epsilon;

  BatchToNGraph(op->name(), is_nhwc, ng_input);

  std::shared_ptr<ng::Node> ng_batch_norm;

  ng_batch_norm = ConstructNgNode<ng::opset3::BatchNormInference>(
      op->name(), tf_epsilon, ng_scale, ng_offset, ng_input, ng_mean,
      ng_variance);
  BatchToTensorflow(op->name(), is_nhwc, ng_batch_norm);
  SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
  if (is_v3) {
    SaveNgOp(ng_op_map, op->name(), ng_mean);
    SaveNgOp(ng_op_map, op->name(), ng_variance);
    SaveNgOp(ng_op_map, op->name(), ng_mean);
    SaveNgOp(ng_op_map, op->name(), ng_variance);
    // FusedBatchNormV3 has 6 outputs (reserve_space_3)
    shared_ptr<ng::Node> ng_reserved_3 =
        ConstructNgNode<ngraph::opset3::Constant>(
            op->name(), ng_mean->get_element_type(), ng::Shape{},
            std::vector<std::string>{""});
    SaveNgOp(ng_op_map, op->name(), ng_reserved_3);
  }
  return Status::OK();
}

static Status TranslateGatherNdOp(const Node* op,
                                  const std::vector<const Tensor*>&,
                                  Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_params;
  shared_ptr<ng::Node> ng_indices;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_params, &ng_indices));

  auto ng_params_shape = ng_params->get_shape();
  size_t ng_params_rank = ng_params_shape.size();
  auto ng_indices_shape = ng_indices->get_shape();
  size_t ng_indices_rank = ng_indices_shape.size();

  for (size_t i = 0; i < ng_params_rank; i++) {
    if (ng_params_shape[i] == 0) {
      return errors::InvalidArgument(
          "Requested more than 0 entries, but params is empty.  Params shape: "
          "[",
          ng::join(ng_params_shape, ","), "]");
    }
  }

  if ((ng_indices_shape[ng_indices_rank - 1]) > ng_params_rank) {
    return errors::InvalidArgument(
        "The last dimension of indices can be at most the rank of params");
  }

  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::op::GatherND>(
                                      op->name(), ng_params, ng_indices));

  return Status::OK();
}

static Status TranslateFusedMatMulOp(const Node* op,
                                     const std::vector<const Tensor*>&,
                                     Builder::OpMap& ng_op_map) {
  int num_args;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_args", &num_args));

  std::vector<string> fused_ops;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "fused_ops", &fused_ops));

  // Transpose arguments if requested.
  bool transpose_a = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_a", &transpose_a));

  bool transpose_b = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_b", &transpose_b));

  shared_ptr<ng::Node> ng_lhs, ng_rhs, ng_bias, ng_matmul;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs, &ng_bias));
  ng_matmul = ConstructNgNode<ngraph::opset3::MatMul>(
      op->name(), ng_lhs, ng_rhs, transpose_a, transpose_b);

  auto ng_matmul_shape = ng_matmul->get_shape();
  auto ng_bias_shape = ng_bias->get_shape();

  if (ng_bias_shape.size() != 1) {
    return errors::InvalidArgument(
        "Bias argument to BiasAdd does not have one dimension");
  }

  auto ng_add =
      ConstructNgNode<ng::opset3::Add>(op->name(), ng_matmul, ng_bias);
  if (fused_ops.size() == 1) {  // Only fusing BiasAdd
    SaveNgOp(ng_op_map, op->name(), ng_add);
  } else if (fused_ops.size() == 2) {  // Also has activation
    if (fused_ops[1] == "Relu") {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<ng::opset3::Relu>(op->name(), ng_add));
    } else if (fused_ops[1] == "Relu6") {
      // TODO fill
      auto constant_6 = ConstructNgNode<ng::opset3::Constant>(
          op->name(), ng_add->get_element_type(), ng_add->get_shape(),
          std::vector<std::string>(ng::shape_size(ng_add->get_shape()), "6"));
      auto relu6_op = ConstructNgNode<ng::opset3::Minimum>(
          op->name(), ConstructNgNode<ng::opset3::Relu>(op->name(), ng_add),
          constant_6);
      SaveNgOp(ng_op_map, op->name(), relu6_op);
    } else {
      return errors::Internal(
          "Expected activation to be Relu or Relu6 but got ", fused_ops[1]);
    }
  } else {
    // Adding this here to catch future changes in _FusedMatMul
    return errors::Internal("Unsupported combination");
  }

  return Status::OK();
}

static Status TranslateGatherV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_input_coords;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input, &ng_input_coords, nullptr));

  std::vector<int64> tf_axis;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &tf_axis));

  if (tf_axis.size() > 1) {
    return errors::Internal("Found axis in GatherV2 op (", op->name(),
                            ") translation to be non scalar, of size ",
                            tf_axis.size());
  }

  std::string backend_name;
  TF_RETURN_IF_ERROR(ngraph_bridge::GetNodeBackend(op, &backend_name));

  // Negative axis is supported. Accounting for that
  auto ng_input_shape = ng_input->get_shape();
  size_t ng_input_rank = ng_input_shape.size();
  int axis;
  if (tf_axis[0] >= 0) {
    axis = tf_axis[0];
  } else {
    axis = tf_axis[0] + ng_input_rank;
  }
  if (axis < 0 || axis >= ng_input_rank) {
    return errors::InvalidArgument("Expected axis in the range [-",
                                   ng_input_rank, ", ", ng_input_rank,
                                   "), but got ", tf_axis[0]);
  }

  auto gather_op = ConstructNgNode<ng::op::Gather>(op->name(), ng_input,
                                                   ng_input_coords, tf_axis[0]);

  SaveNgOp(ng_op_map, op->name(), gather_op);
  return Status::OK();
}

static Status TranslateFusedConv2DOp(const Node* op,
                                     const std::vector<const Tensor*>&,
                                     Builder::OpMap& ng_op_map) {
  int num_args;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_args", &num_args));

  std::vector<string> fused_ops;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "fused_ops", &fused_ops));

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));
  bool is_nhwc = (tf_data_format == "NHWC");

  auto CreateNgConv = [&](shared_ptr<ng::Node>& ng_input,
                          shared_ptr<ng::Node>& ng_filter,
                          shared_ptr<ng::Node>& ng_conv) {
    std::vector<int32> tf_strides;
    std::vector<int32> tf_dilations;
    std::string tf_padding_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
      return errors::InvalidArgument(
          "Conv2D data format is neither NHWC nor NCHW");
    }

    // TF Kernel Test Checks
    // Strides in the batch and depth dimension is not supported
    if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
      return errors::InvalidArgument(
          "Strides in batch and depth dimensions is not supported: ",
          op->type_string());
    }

    NGRAPH_VLOG(3) << ng::join(tf_strides);
    NGRAPH_VLOG(3) << ng::join(tf_dilations);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    ng::Strides ng_strides(2);
    ng::Strides ng_dilations(2);
    ng::Shape ng_image_shape(2);
    ng::Shape ng_kernel_shape(2);

    BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
    BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
    BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
    BatchToNGraph(op->name(), is_nhwc, ng_input);

    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter->get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    Transpose<3, 2, 0, 1>(ng_filter);
    Builder::SetTracingInfo(op->name(), ng_filter);

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff ng_padding_below{0, 0};
    ng::CoordinateDiff ng_padding_above{0, 0};

    ng::op::PadType ng_pad_type = ng::op::PadType::EXPLICIT;
    if (tf_padding_type == "VALID") {
      ng_pad_type = ng::op::PadType::VALID;
    } else {
      Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                           ng_strides, ng_dilations, ng_padding_below,
                           ng_padding_above);
    }

    ng_conv = ConstructNgNode<ng::opset3::Convolution>(
        op->name() + "_FusedConv2D_Conv", ng_input, ng_filter, ng_strides,
        ng_padding_below, ng_padding_above, ng_dilations, ng_pad_type);

    return Status::OK();
  };

  auto create_relu6 = [](const string& op_name,
                         const shared_ptr<ng::Node>& ng_node) {
    auto constant_6 = ConstructNgNode<ng::opset3::Constant>(
        op_name, ng_node->get_element_type(), ng_node->get_shape(),
        std::vector<std::string>(ng::shape_size(ng_node->get_shape()), "6"));
    auto relu6_op = ConstructNgNode<ng::opset3::Minimum>(
        op_name, ConstructNgNode<ng::opset3::Relu>(
                     op_name + "_FusedConv2D_Relu", ng_node),
        constant_6);
    return relu6_op;
  };

  if (VecStrCmp(fused_ops, {"BiasAdd"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
    if (num_args != 1) {
      return errors::InvalidArgument(
          "FusedConv2DBiasAdd has incompatible num_args");
    }

    shared_ptr<ng::Node> ng_input, ng_filter, ng_bias, ng_conv;
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, &ng_input, &ng_filter, &ng_bias));

    TF_RETURN_IF_ERROR(CreateNgConv(ng_input, ng_filter, ng_conv));

    BatchToTensorflow(op->name(), is_nhwc, ng_conv);

    auto ng_conv_shape = ng_conv->get_shape();
    auto ng_bias_shape = ng_bias->get_shape();
    if (ng_bias_shape.size() != 1) {
      return errors::InvalidArgument(
          "Bias argument to BiasAdd does not have one dimension");
    }

    auto ng_add = ConstructNgNode<ng::opset3::Add>(
        op->name() + "_FusedConv2D_BiasAdd", ng_conv, ng_bias);

    if (VecStrCmp(fused_ops, {"BiasAdd", "Relu"})) {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<ng::opset3::Relu>(
                   op->name() + "_FusedConv2D_Relu", ng_add));
    } else if (VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
      SaveNgOp(ng_op_map, op->name(), create_relu6(op->name(), ng_add));
    } else {
      SaveNgOp(ng_op_map, op->name(), ng_add);
    }
  } else if (VecStrCmp(fused_ops, {"FusedBatchNorm"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
    if (num_args != 4) {
      return errors::InvalidArgument(
          "FusedConv2D with FusedBatchNorm has incompatible num_args");
    }

    shared_ptr<ng::Node> ng_input, ng_filter, ng_conv, ng_scale, ng_offset,
        ng_mean, ng_variance;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_filter,
                                     &ng_scale, &ng_offset, &ng_mean,
                                     &ng_variance));
    TF_RETURN_IF_ERROR(CreateNgConv(ng_input, ng_filter, ng_conv));

    float tf_epsilon;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon));

    std::shared_ptr<ng::Node> ng_batch_norm =
        ConstructNgNode<ng::opset3::BatchNormInference>(
            op->name() + "_FusedConv2D_BatchNorm", tf_epsilon, ng_scale,
            ng_offset, ng_conv, ng_mean, ng_variance);

    BatchToTensorflow(op->name(), is_nhwc, ng_batch_norm);

    if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"})) {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<ng::opset3::Relu>(
                   op->name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm));
    } else if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
      SaveNgOp(ng_op_map, op->name(), create_relu6(op->name(), ng_batch_norm));
    } else {
      SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
    }
  } else {
    return errors::Unimplemented("Unsupported _FusedConv2D " +
                                 absl::StrJoin(fused_ops, ","));
  }
  return Status::OK();
}

static Status TranslateIdentityOp(const Node* op,
                                  const std::vector<const Tensor*>&,
                                  Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_arg;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_arg));
  SaveNgOp(ng_op_map, op->name(), ng_arg);
  return Status::OK();
}

static Status TranslateIsFiniteOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  // Implemented tf.is_finite by checking:
  // (in != inf) && (in != -inf) && (in == in)
  //                                 ^^^^^^^^ checks for NaN's
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto const_inf = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_input->get_element_type(), ng::Shape{},
      std::vector<float>{std::numeric_limits<float>::infinity()});

  auto const_neg_inf = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_input->get_element_type(), ng::Shape{},
      std::vector<float>{-std::numeric_limits<float>::infinity()});

  auto neq_inf = ConstructNgNode<ng::op::NotEqual>(
      op->name(), ng_input, const_inf,
      ng::op::AutoBroadcastSpec(ng::op::AutoBroadcastType::NUMPY));
  auto neq_neg_inf = ConstructNgNode<ng::op::NotEqual>(
      op->name(), ng_input, const_neg_inf,
      ng::op::AutoBroadcastSpec(ng::op::AutoBroadcastType::NUMPY));
  auto eq_nan = ConstructNgNode<ng::op::Equal>(op->name(), ng_input, ng_input);

  auto neq_inf_and_neq_neg_inf =
      ConstructNgNode<ng::op::And>(op->name(), neq_inf, neq_neg_inf);
  auto is_finite =
      ConstructNgNode<ng::op::And>(op->name(), neq_inf_and_neq_neg_inf, eq_nan);

  SaveNgOp(ng_op_map, op->name(), is_finite);
  return Status::OK();
}

static Status TranslateL2LossOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto const_2 = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_input->get_element_type(), ng::Shape{},
      std::vector<std::string>{"2"});

  std::shared_ptr<ng::Node> ng_pow =
      ConstructNgNode<ng::opset3::Multiply>(op->name(), ng_input, ng_input);

  size_t input_rank = ng_input->get_shape().size();
  ng::AxisSet axes;
  for (size_t i = 0; i < input_rank; ++i) {
    axes.insert(i);
  }

  std::shared_ptr<ng::Node> ng_sum =
      ConstructNgNode<ng::op::Sum>(op->name(), ng_pow, axes);
  std::shared_ptr<ng::Node> ng_l2loss =
      ConstructNgNode<ng::opset3::Divide>(op->name(), ng_sum, const_2);
  SaveNgOp(ng_op_map, op->name(), ng_l2loss);
  return Status::OK();
}

static Status TranslateLog1pOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](std::shared_ptr<ng::Node> n) {
        auto et = n->get_element_type();
        auto shape = n->get_shape();
        std::vector<std::string> val_1(ng::shape_size(shape), "1");
        auto ng_const1 =
            ConstructNgNode<ng::opset3::Constant>(op->name(), et, shape, val_1);
        auto ng_add =
            ConstructNgNode<ng::opset3::Add>(op->name(), ng_const1, n);
        return ConstructNgNode<ng::opset3::Log>(op->name(), ng_add);
      });
}

static Status TranslateLogSoftmaxOp(const Node* op,
                                    const std::vector<const Tensor*>&,
                                    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_inp));
  auto inp_shape = ng_inp->get_shape();
  size_t rank = inp_shape.size();
  auto ng_axis = ng::AxisSet{rank - 1};
  // Batch i, class j
  // logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
  // Actually implementing: logsoftmax[i, j] = logits[i, j] - max(logits[i]) -
  // log(sum(exp(logits[i] - max(logits[i]))))
  auto ng_max = ConstructNgNode<ng::op::Broadcast>(
      op->name(), ConstructNgNode<ng::op::Max>(op->name(), ng_inp, ng_axis),
      inp_shape, ng_axis);
  auto ng_inp_minus_max =
      ConstructNgNode<ng::opset3::Subtract>(op->name(), ng_inp, ng_max);
  auto ng_exp = ConstructNgNode<ng::op::Exp>(op->name(), ng_inp_minus_max);
  auto ng_log_sum = ConstructNgNode<ng::op::Log>(
      op->name(), ConstructNgNode<ng::op::Sum>(op->name(), ng_exp, ng_axis));
  auto ng_broadcast = ConstructNgNode<ng::op::Broadcast>(
      op->name(), ng_log_sum, ng_inp->get_shape(), ng_axis);
  auto ng_output = ConstructNgNode<ng::opset3::Subtract>(
      op->name(), ng_inp_minus_max, ng_broadcast);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateSoftplusOp(const Node* op,
                                  const std::vector<const Tensor*>&,
                                  Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_inp));
  auto ng_exp = ConstructNgNode<ng::opset3::Exp>(op->name(), ng_inp);
  auto constant_1 = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng_inp->get_element_type(), ng_inp->get_shape(),
      std::vector<std::string>(ng::shape_size(ng_inp->get_shape()), "1"));
  auto ng_output = ConstructNgNode<ng::opset3::Log>(
      op->name(),
      ConstructNgNode<ng::opset3::Add>(op->name(), ng_exp, constant_1));
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateMatMulOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));

  // Transpose arguments if requested.
  bool transpose_a = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_a", &transpose_a));

  bool transpose_b = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_b", &transpose_b));

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ngraph::opset3::MatMul>(op->name(), ng_lhs, ng_rhs,
                                                   transpose_a, transpose_b));
  return Status::OK();
}

static Status TranslateMaxPoolOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "MaxPool data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);

  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(op->name(), is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for MaxPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  ng::op::PadType ng_pad_type = ng::op::PadType::EXPLICIT;
  if (tf_padding_type == "VALID") {
    ng_pad_type = ng::op::PadType::VALID;
  } else {
    Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_padding_below, ng_padding_above);
  }

  std::shared_ptr<ng::Node> ng_maxpool = ConstructNgNode<ng::opset3::MaxPool>(
      op->name(), ng_input, ng_strides, ng_padding_below, ng_padding_above,
      ng_kernel_shape, ng::op::RoundingType::FLOOR, ng_pad_type);

  BatchToTensorflow(op->name(), is_nhwc, ng_maxpool);

  NGRAPH_VLOG(3) << "maxpool outshape: {" << ng::join(ng_maxpool->get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_maxpool);
  return Status::OK();
}

static Status TranslateMaxPool3DOp(const Node* op,
                                   const std::vector<const Tensor*>&,
                                   Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NDHWC" && tf_data_format != "NCDHW") {
    return errors::InvalidArgument(
        "MaxPool3D data format is neither NDHWC nor NCDHW");
  }

  bool is_ndhwc = (tf_data_format == "NDHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(3);
  ng::Shape ng_image_shape(3);
  ng::Shape ng_kernel_shape(3);

  BatchedOpParam3DToNGraph(is_ndhwc, tf_strides, ng_strides);
  BatchedOpParam3DToNGraph(is_ndhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParam3DToNGraph(is_ndhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph3D(op->name(), is_ndhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for MaxPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0, 0};
  ng::Shape ng_padding_above{0, 0, 0};

  ng::op::PadType ng_pad_type = ng::op::PadType::EXPLICIT;
  if (tf_padding_type == "VALID") {
    ng_pad_type = ng::op::PadType::VALID;
  } else {
    Builder::MakePadding3D(tf_padding_type, ng_image_shape, ng_kernel_shape,
                           ng_strides, ng_padding_below, ng_padding_above);
  }

  std::shared_ptr<ng::Node> ng_maxpool = ConstructNgNode<ng::opset3::MaxPool>(
      op->name(), ng_input, ng_strides, ng_padding_below, ng_padding_above,
      ng_kernel_shape, ng::op::RoundingType::FLOOR, ng_pad_type);

  BatchToTensorflow3D(op->name(), is_ndhwc, ng_maxpool);

  NGRAPH_VLOG(3) << "maxpool outshape: {" << ng::join(ng_maxpool->get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_maxpool);
  return Status::OK();
}

static Status TranslateNonMaxSuppressionV4Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_boxes, ng_scores;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_boxes, &ng_scores,
                                   nullptr, nullptr, nullptr));

  std::vector<int> max_output_size;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 2, static_input_map, &max_output_size));
  std::vector<float> iou_threshold;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 3, static_input_map, &iou_threshold));

  std::vector<float> score_threshold;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 4, static_input_map, &score_threshold));

  bool pad_to_max_output_size;
  if (GetNodeAttr(op->attrs(), "pad_to_max_output_size",
                  &pad_to_max_output_size) != Status::OK()) {
    pad_to_max_output_size = false;
  }
  // max_output_size must be scalar
  if (max_output_size.size() != 1) {
    return errors::InvalidArgument(
        "NonMaxSuppressionV4 Op: max_output_size of nms must be scalar ",
        max_output_size.size());
  }
  // iou_threshold must be scalar
  if (iou_threshold.size() != 1) {
    return errors::InvalidArgument(
        "NonMaxSuppressionV4 Op: iou_threshold of nms must be scalar ",
        iou_threshold.size());
  }

  // score_threshold must be scalar
  if (score_threshold.size() != 1) {
    return errors::InvalidArgument(
        "NonMaxSuppressionV4 Op: score_threshold of nms must be scalar ",
        score_threshold.size());
  }

  auto ng_max_output_size = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::i64, ng::Shape{}, max_output_size[0]);
  auto ng_iou_threshold = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::f32, ng::Shape{}, iou_threshold[0]);
  auto ng_score_threshold = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::f32, ng::Shape{}, score_threshold[0]);

  shared_ptr<ng::Node> ng_nmsv4 =
      ConstructNgNode<ng::opset3::NonMaxSuppression>(
          op->name(), ng_boxes, ng_scores, ng_max_output_size, ng_iou_threshold,
          ng_score_threshold);

  // ADK: (FIXME) are the outputs set correctly here?
  Builder::SetTracingInfo(op->name(), ng_nmsv4);
  shared_ptr<ngraph::Node> ng_selected_indices =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_nmsv4, 0);
  shared_ptr<ngraph::Node> ng_valid_output =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_nmsv4, 1);

  SaveNgOp(ng_op_map, op->name(), ng_selected_indices);
  SaveNgOp(ng_op_map, op->name(), ng_valid_output);

  return Status::OK();
}

static Status TranslateReduceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(
        std::shared_ptr<ng::Node>, std::shared_ptr<ng::Node>, const bool)>
        create_ng_node) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input));
  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &axes));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(axes.size());
  std::transform(
      axes.begin(), axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  auto ng_reduction_axes = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::i64, ng::Shape{ng_reduction_axes_vect.size()},
      ng_reduction_axes_vect);

  std::shared_ptr<ng::Node> ng_node =
      create_ng_node(ng_input, ng_reduction_axes, tf_keep_dims);
  Builder::SetTracingInfo(op->name(), ng_node);

  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

template <typename T>
static Status TranslateDirectReduceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  // ensure its either an arithmetic or a logical reduction
  if (!(std::is_base_of<ngraph::op::util::ArithmeticReduction, T>::value ||
        std::is_base_of<ngraph::op::util::LogicalReduction, T>::value)) {
    return errors::InvalidArgument(
        "Expected node to be either a valid logical or arithmetic reduction "
        "type");
  }
  return TranslateReduceOp(
      op, static_input_map, ng_op_map,
      [&op](std::shared_ptr<ng::Node> ng_input,
            std::shared_ptr<ng::Node> ng_reduction_axes, const bool keep_dims) {
        return ConstructNgNode<T>(op->name(), ng_input, ng_reduction_axes,
                                  keep_dims);
      });
}

static Status TranslateOneHotOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_features, ng_on, ng_off, ng_depth;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_features, nullptr, &ng_on, &ng_off));

  auto ng_features_shape = ng_features->get_shape();
  std::vector<int> depth;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &depth));

  // Depth must be scalar
  if (depth.size() != 1) {
    return errors::InvalidArgument(
        "OneHot Op: depth of one hot dimension must be scalar ", depth.size());
  }

  auto const_depth = ConstructNgNode<ng::op::Constant>(
      op->name(), ng::element::i64, ng::Shape{}, depth);

  int one_hot_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &one_hot_axis));

  shared_ptr<ng::Node> ng_onehot = ConstructNgNode<ng::opset3::OneHot>(
      op->name(), ng_features, const_depth, ng_on, ng_off, one_hot_axis);
  SaveNgOp(ng_op_map, op->name(), ng_onehot);
  return Status::OK();
}

static Status TranslatePackOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 1));

  ng::NodeVector ng_concat_inputs;

  for (tensorflow::int32 i = 0; i < op->num_inputs(); ++i) {
    shared_ptr<ng::Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, &ng_input));
    ng_concat_inputs.push_back(ng_input);
  }

  int32 tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  size_t input_rank = ng_concat_inputs[0]->get_shape().size();

  auto concat_axis = tf_axis;
  if (concat_axis == -1) {
    concat_axis = input_rank;
  }

  ng::Shape input_shape = ng_concat_inputs[0]->get_shape();
  ng::Shape output_shape(input_rank + 1);

  // if inputs shape is (2, 3, 4), and axis is 1, then we want
  // to create output_shape (2, num_inputs, 3, 4)
  for (size_t i = 0; i < input_rank; ++i) {
    output_shape[((int)i < concat_axis) ? i : i + 1] = input_shape[i];
  }
  output_shape[concat_axis] = op->num_inputs();

  ng::AxisVector ng_axis_order(input_rank);
  std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

  if ((size_t)concat_axis == input_rank) {
    // need to add extra dimension before we concatenate
    // along it
    ng::Shape extended_shape = input_shape;
    extended_shape.push_back(1);
    for (size_t i = 0; i < ng_concat_inputs.size(); ++i) {
      ng_concat_inputs[i] = ConstructNgNode<ng::op::Reshape>(
          op->name(), ng_concat_inputs[i], ng_axis_order, extended_shape);
    }
    ng_axis_order.push_back(input_rank);
  }

  auto concat = ConstructNgNode<ng::op::Concat>(op->name(), ng_concat_inputs,
                                                concat_axis);
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::Reshape>(op->name(), concat, ng_axis_order,
                                            output_shape));
  return Status::OK();
}

// 3 different Pad Ops: Pad, PadV2, MirrorPad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad-v2
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mirror-pad
static Status TranslatePadOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_paddings_op, pad_val_op, result_pad_op;

  // Set inputs and pad_val_op
  if (op->type_string() == "Pad" || op->type_string() == "MirrorPad") {
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, &ng_input, &ng_paddings_op));
    pad_val_op = ConstructNgNode<ng::opset3::Constant>(
        op->name(), ng_input->get_element_type(), ng::Shape(),
        std::vector<int>({0}));
  } else if (op->type_string() == "PadV2") {
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, &ng_input, &ng_paddings_op, &pad_val_op));
  } else {
    return errors::InvalidArgument("Incorrect TF Pad OpType: " +
                                   op->type_string());
  }

  // Set pad_mode
  auto pad_mode = ng::op::PadMode::CONSTANT;
  if (op->type_string() == "MirrorPad") {
    std::string pad_mode_str;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "mode", &pad_mode_str));
    if (pad_mode_str == "REFLECT") {
      pad_mode = ng::op::PadMode::REFLECT;
    } else if (pad_mode_str == "SYMMETRIC") {
      pad_mode = ng::op::PadMode::SYMMETRIC;
    } else {
      return errors::InvalidArgument(pad_mode_str,
                                     " is not an allowed padding mode.");
    }
  }

  // Set pads_begin & pads_end (from the pad_val_op)
  shared_ptr<ng::opset3::Constant> pads_begin_node, pads_end_node;
  std::vector<int64> paddings;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &paddings));
  NGRAPH_VLOG(6) << op->name() << " pads {" << ng::join(paddings) << "}";
  if (paddings.size() % 2 != 0) {
    return errors::InvalidArgument(
        "Constant node for paddings does not have an even number of "
        "elements");
  }
  ng::CoordinateDiff pad_begin(paddings.size() / 2);
  ng::CoordinateDiff pad_end(paddings.size() / 2);
  for (size_t i = 0; i < paddings.size() / 2; i++) {
    pad_begin[i] = paddings[2 * i];
    pad_end[i] = paddings[2 * i + 1];
  }
  pads_begin_node = make_shared<ng::opset3::Constant>(
      ng::element::i64, ng::Shape{pad_begin.size()}, pad_begin);
  pads_end_node = make_shared<ng::opset3::Constant>(
      ng::element::i64, ng::Shape{pad_end.size()}, pad_end);

  // Create final Op
  result_pad_op =
      ConstructNgNode<ng::opset3::Pad>(op->name(), ng_input, pads_begin_node,
                                       pads_end_node, pad_val_op, pad_mode);

  SaveNgOp(ng_op_map, op->name(), result_pad_op);
  return Status::OK();
}

static Status TranslateRankOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;

  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  ng::Shape input_shape = ng_input->get_shape();
  auto input_rank = static_cast<int>(input_shape.size());

  auto ng_rank = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::i32, ng::Shape(),
      std::vector<int>({input_rank}));

  SaveNgOp(ng_op_map, op->name(), ng_rank);
  return Status::OK();
}

static Status TranslateReciprocalOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](std::shared_ptr<ng::Node> n) {
        // Create a constant tensor populated with the value -1.
        // (1/x = x^(-1))
        auto et = n->get_element_type();
        auto shape = n->get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-1");
        auto ng_exponent = ConstructNgNode<ng::opset3::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -1.
        return ConstructNgNode<ng::opset3::Power>(op->name(), n, ng_exponent);
      });
}

template <typename T>
Status QuantizeAndDequantizeV2Helper(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    const bool& range_given, const bool& signed_input, const int& num_bits,
    float* scale_out) {
  // TODO: currently handling only float, generalize later?
  T min_range = 0, max_range = 0;
  if (range_given) {
    std::vector<T> input_min, input_max;
    TF_RETURN_IF_ERROR(
        GetStaticInputVector(op, 1, static_input_map, &input_min));
    TF_RETURN_IF_ERROR(
        GetStaticInputVector(op, 2, static_input_map, &input_max));
    if (input_min.size() != 1) {
      return errors::InvalidArgument(
          "QuantizeAndDequantizeV2 Op: input_min must be scalar. Got a "
          "vector "
          "of size, ",
          input_min.size());
    }
    if (input_max.size() != 1) {
      return errors::InvalidArgument(
          "QuantizeAndDequantizeV2 Op: input_max must be scalar. Got a "
          "vector "
          "of size, ",
          input_max.size());
    }
    min_range = input_min[0];
    max_range = input_max[0];
    if (min_range > max_range) {
      return errors::InvalidArgument(
          "Expected QuantizeAndDequantizeV2's input_min <= input_max but "
          "got, "
          "input_min = ",
          min_range, " and input_max = ", max_range);
    }
    // m = max(abs(input_min), abs(input_max));
  } else {
    // m = max(abs(min_elem(input)), abs(max_elem(input)));
    // TODO implement this.
    // Note to implement this we need:
    // min = ng_min(inp_tensor); max = mg_max(inp_tensor).
    // which means, unless we support pattern matching that accepts the ng min
    // and max nodes, we have to declare inp_data tensor to be static
  }
  const int64 min_quantized = signed_input ? -(1ULL << (num_bits - 1)) : 0;
  const int64 max_quantized = min_quantized + ((1ULL << num_bits) - 1);
  const T scale_from_min_side = (min_quantized * min_range > 0)
                                    ? min_quantized / min_range
                                    : std::numeric_limits<T>::max();
  const T scale_from_max_side = (max_quantized * max_range > 0)
                                    ? max_quantized / max_range
                                    : std::numeric_limits<T>::max();
  T inverse_scale;
  if (scale_from_min_side < scale_from_max_side && min_quantized != 0) {
    // min_quantized != 0 is not really necessary but klocwork complains
    // T scale = scale_from_min_side;
    inverse_scale = min_range / min_quantized;
    // max_range = max_quantized * inverse_scale;
  } else {
    // T scale = scale_from_max_side;
    inverse_scale = max_range / max_quantized;
    // min_range = min_quantized * inverse_scale;
  }
  *scale_out = inverse_scale;

  return Status::OK();
}

static Status TranslateQuantizeAndDequantizeV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, nullptr, nullptr));
  bool range_given;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "range_given", &range_given));

  bool signed_input;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "signed_input", &signed_input));

  int num_bits;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_bits", &num_bits));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "T", &dtype));
  // T: float, double....not supported: bfloat16, half
  float scale;
  ng::element::Type ng_r_et;
  switch (dtype) {
    case DT_FLOAT:
      TF_RETURN_IF_ERROR(QuantizeAndDequantizeV2Helper<float>(
          op, static_input_map, range_given, signed_input, num_bits, &scale));
      ng_r_et = ng::element::f32;
      break;
    case DT_DOUBLE:
      TF_RETURN_IF_ERROR(QuantizeAndDequantizeV2Helper<double>(
          op, static_input_map, range_given, signed_input, num_bits, &scale));
      ng_r_et = ng::element::f64;
      break;
    default:
      return errors::InvalidArgument(
          "Expected QuantizeAndDequantizeV2's datatype to be of DT_FLOAT or "
          "DT_DOUBLE but got ",
          DataTypeString(dtype));
  }
  // The quantized data type
  ng::element::Type ng_q_et;
  switch (num_bits) {
    case 8:
      ng_q_et = signed_input ? ng::element::i8 : ng::element::u8;
      break;
    default:
      return errors::InvalidArgument(
          "Expected QuantizeAndDequantizeV2's num_bits to be 8, but got ",
          num_bits);
  }
  auto ng_scale = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_r_et, ng::Shape(), std::vector<float>({scale}));
  auto ng_offset = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_q_et, ng::Shape(), std::vector<int>({0}));
  ng::op::Quantize::RoundMode ng_round_mode;
  string round_mode_string;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "round_mode", &round_mode_string));
  if (round_mode_string == "HALF_UP") {
    ng_round_mode = ng::op::Quantize::RoundMode::ROUND_NEAREST_UPWARD;
  } else if (round_mode_string == "HALF_TO_EVEN") {
    ng_round_mode = ng::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
  } else {
    return errors::Internal("Tensorflow Rounding Mode not supported by Ngraph");
  }
  auto ng_quant = ConstructNgNode<ng::op::Quantize>(
      op->name(), ng_input, ng_scale, ng_offset, ng_q_et, ng::AxisSet(),
      ng_round_mode);
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::op::Dequantize>(
                                      op->name(), ng_quant, ng_scale, ng_offset,
                                      ng_r_et, ng::AxisSet()));

  // TODO: what of clamping?
  return Status::OK();
}

static Status TranslateQuantizedAvgPoolOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateQuantizedPoolOp(op, static_input_map, ng_op_map,
                                  "QuantizedAvgPool");
}

// Helper function to translate QuantizedConcat and QuantizedConcatV2
static Status TranslateQuantizedConcatOpHelper(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map, std::string op_name) {
  int axis_index;         // index for concat_axis input
  int value_start_index;  // start index for N tensor inputs
  auto num_of_tensors_to_concat = (op->num_inputs() - 1) / 3;

  if (op_name == "QuantizedConcat") {
    axis_index = 0;
    value_start_index = 1;
  } else if (op_name == "QuantizedConcatV2") {
    axis_index = num_of_tensors_to_concat;
    value_start_index = 0;
  } else {
    return errors::InvalidArgument(
        "This helper function is only used for QuantizedConcat and "
        "QuantizedConcatV2 ops");
  }

  auto collect_nodes = [&op, &ng_op_map](int start, int end,
                                         ng::NodeVector* p_ng_args) {
    for (int i = start; i < end; i++) {
      shared_ptr<ng::Node> ng_arg;
      TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, &ng_arg));
      (*p_ng_args).push_back(ng_arg);
    }
    return Status::OK();
  };

  // Collect the N input tensors to concat with
  ng::NodeVector ng_args;
  if (op_name == "QuantizedConcat") {
    TF_RETURN_IF_ERROR(collect_nodes(value_start_index,
                                     num_of_tensors_to_concat + 1, &ng_args));
  } else if (op_name == "QuantizedConcatV2") {
    TF_RETURN_IF_ERROR(
        collect_nodes(value_start_index, num_of_tensors_to_concat, &ng_args));
  }

  // Get the input concat_axis
  std::vector<int64> tf_concat_axis_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, axis_index, static_input_map,
                                          &tf_concat_axis_vec));

  // QuantizedConcat doesn't have negative concat_axis
  int64 concat_axis = tf_concat_axis_vec[0];

  // Get input_mins and input_maxs
  std::vector<float> all_mins(num_of_tensors_to_concat),
      all_maxs(num_of_tensors_to_concat);

  // Construct input parameters to ScaledQuantizedConcat op
  ng::NodeVector ng_all_mins, ng_all_maxs;
  std::vector<float> min_tmp, max_tmp;

  // Collect the N input mins and input maxes
  for (int idx = 0; idx < num_of_tensors_to_concat; idx++) {
    TF_RETURN_IF_ERROR(GetStaticInputVector(
        op, num_of_tensors_to_concat + 1 + idx, static_input_map, &min_tmp));
    TF_RETURN_IF_ERROR(
        GetStaticInputVector(op, 2 * num_of_tensors_to_concat + 1 + idx,
                             static_input_map, &max_tmp));

    all_mins[idx] = min_tmp[0];
    all_maxs[idx] = max_tmp[0];

    auto min_node = ConstructNgNode<ng::op::Constant>(
        op->name(), ng::element::f32, ng::Shape{}, min_tmp);
    auto max_node = ConstructNgNode<ng::op::Constant>(
        op->name(), ng::element::f32, ng::Shape{}, max_tmp);

    ng_all_mins.push_back(ConstructNgNode<ngraph::op::Reshape>(
        op->name(), min_node, ngraph::AxisVector{}, ngraph::Shape{1}));
    ng_all_maxs.push_back(ConstructNgNode<ngraph::op::Reshape>(
        op->name(), max_node, ngraph::AxisVector{}, ngraph::Shape{1}));
  }

  // return the min among the input_mins, and the max among the input_maxs
  // TODO: TF has a different way of determine the output_min and output_max
  // TF reference:
  // https://github.com/tensorflow/tensorflow/blob/86950c2c440be956a9fcb3a25868a1df15444467/tensorflow/core/kernels/quantized_concat_op.cc#L78
  std::vector<float> min_of_mins(
      1, *std::min_element(all_mins.begin(), all_mins.end()));
  std::vector<float> max_of_maxs(
      1, *std::max_element(all_maxs.begin(), all_maxs.end()));

  // construct output_min and output_max
  shared_ptr<ng::Node> ng_min_of_mins = ConstructNgNode<ng::op::Constant>(
      op->name(), ng::element::f32, ng::Shape{}, min_of_mins);
  shared_ptr<ng::Node> ng_max_of_maxs = ConstructNgNode<ng::op::Constant>(
      op->name(), ng::element::f32, ng::Shape{}, max_of_maxs);

  auto ng_qconcat = ng::builder::QuantizedConcatBuilder(
      ng_args, size_t(concat_axis), ng_all_mins, ng_all_maxs);
  Builder::SetTracingInfo(op->name(), ng_qconcat);

  SaveNgOp(ng_op_map, op->name(), ng_qconcat);
  SaveNgOp(ng_op_map, op->name(), ng_min_of_mins);
  SaveNgOp(ng_op_map, op->name(), ng_max_of_maxs);
  return Status::OK();
}

static Status TranslateQuantizedConcatOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateQuantizedConcatOpHelper(op, static_input_map, ng_op_map,
                                          "QuantizedConcat");
}

static Status TranslateQuantizedConcatV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateQuantizedConcatOpHelper(op, static_input_map, ng_op_map,
                                          "QuantizedConcatV2");
}

static Status TranslateQuantizedConv(
    const Node* op, Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(
        std::vector<std::shared_ptr<ng::Node>>, ng::Strides, ng::Strides,
        ng::CoordinateDiff, ng::CoordinateDiff, ng::Strides)>
        create_quantized_conv_node) {
  size_t num_tf_op_inputs = op->num_inputs();
  size_t num_node_inputs = num_tf_op_inputs;
  std::vector<std::shared_ptr<ng::Node>> node_inps(num_node_inputs);
  for (size_t inp_idx = 0; inp_idx < num_tf_op_inputs; inp_idx++) {
    TF_RETURN_IF_ERROR(
        GetInputNode(ng_op_map, op, inp_idx, &(node_inps[inp_idx])));
  }

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::string tf_padding_type;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  bool is_nhwc = true;  // TODO: Assuming this data format for now
  ng::Strides ng_strides(2);
  ng::Strides ng_dilations(2);
  ng::Strides ng_data_dilations({1, 1});
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, node_inps[0]->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations, ng_dilations);
  // Generally, the mapping is: 0->input, 1->filter, 2->bias, 3->sum input
  BatchToNGraph(op->name(), is_nhwc, node_inps[0]);
  // QconvBiasAdd variants
  if (num_node_inputs == 12) {
    BatchToNGraph(op->name(), is_nhwc, node_inps[9]);
  }
  auto& ng_filter_shape = node_inps[1]->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(node_inps[1]);
  Builder::SetTracingInfo(op->name(), node_inps[1]);
  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  // It is expected by QuantizedConvolutionBiasBuilder (and other builder
  // functions) that the min max inputs be constant nodes
  // Hence declaring them static, reading their values and converting to
  // constant nodes
  std::shared_ptr<ng::Node> ng_quant_conv_bias = create_quantized_conv_node(
      node_inps, ng_strides, ng_dilations, ng_padding_below, ng_padding_above,
      ng_data_dilations);
  Builder::SetTracingInfo(op->name(), ng_quant_conv_bias);

  BatchToTensorflow(op->name(), is_nhwc, ng_quant_conv_bias);
  SaveNgOp(ng_op_map, op->name(), ng_quant_conv_bias);
  // QconvBiasAdd variants have summand and its min/max as the last input
  // nodes
  auto adjust_idx = num_node_inputs == 12 ? 3 : 0;
  // Forward the min_freezed_output input to output min
  SaveNgOp(ng_op_map, op->name(), node_inps[num_node_inputs - 2 - adjust_idx]);
  // Forward the max_freezed_output input to output max
  SaveNgOp(ng_op_map, op->name(), node_inps[num_node_inputs - 1 - adjust_idx]);
  return Status::OK();
}

template <bool IsRelu>
static Status TranslateQuantizedConv2DWithBiasMaybeReluAndRequantizeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  string op_name = op->name();
  auto create_quantized_conv_node = [&op_name](
      std::vector<std::shared_ptr<ng::Node>> node_inps, ng::Strides ng_strides,
      ng::Strides ng_dilations, ng::CoordinateDiff ng_padding_below,
      ng::CoordinateDiff ng_padding_above, ng::Strides ng_data_dilations) {
    auto ng_node = ng::builder::QuantizedConvolutionBiasBuilder(
        node_inps[0], node_inps[1], node_inps[2], ng_strides, ng_dilations,
        ng_padding_below, ng_padding_above, ng_data_dilations, node_inps[3],
        node_inps[4], node_inps[5], node_inps[6], node_inps[7], node_inps[8],
        IsRelu);
    Builder::SetTracingInfo(op_name, ng_node);
    return ng_node;
  };
  return TranslateQuantizedConv(op, ng_op_map, create_quantized_conv_node);
}

static Status TranslateQuantizedConv2DWithBiasSumAndReluAndRequantizeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  string op_name = op->name();
  auto create_quantized_conv_node = [&op_name](
      std::vector<std::shared_ptr<ng::Node>> node_inps, ng::Strides ng_strides,
      ng::Strides ng_dilations, ng::CoordinateDiff ng_padding_below,
      ng::CoordinateDiff ng_padding_above, ng::Strides ng_data_dilations) {
    auto ng_node = ng::builder::QuantizedConvolutionBiasAddBuilder(
        node_inps[0], node_inps[1], node_inps[2], node_inps[9], ng_strides,
        ng_dilations, ng_padding_below, ng_padding_above, ng_data_dilations,
        node_inps[3], node_inps[4], node_inps[5], node_inps[6], node_inps[7],
        node_inps[8], node_inps[10], node_inps[11], true);
    Builder::SetTracingInfo(op_name, ng_node);
    return ng_node;
  };
  return TranslateQuantizedConv(op, ng_op_map, create_quantized_conv_node);
}

static Status TranslateQuantizedConv2DWithBiasSignedSumAndReluAndRequantizeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  string op_name = op->name();
  auto create_quantized_conv_node = [&op_name](
      std::vector<std::shared_ptr<ng::Node>> node_inps, ng::Strides ng_strides,
      ng::Strides ng_dilations, ng::CoordinateDiff ng_padding_below,
      ng::CoordinateDiff ng_padding_above, ng::Strides ng_data_dilations) {
    auto ng_node = ng::builder::QuantizedConvolutionBiasSignedAddBuilder(
        node_inps[0], node_inps[1], node_inps[2], node_inps[9], ng_strides,
        ng_dilations, ng_padding_below, ng_padding_above, ng_data_dilations,
        node_inps[3], node_inps[4], node_inps[5], node_inps[6], node_inps[7],
        node_inps[8], node_inps[10], node_inps[11], true);
    Builder::SetTracingInfo(op_name, ng_node);
    return ng_node;
  };
  return TranslateQuantizedConv(op, ng_op_map, create_quantized_conv_node);
}

static Status TranslateQuantizedMaxPoolOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateQuantizedPoolOp(op, static_input_map, ng_op_map,
                                  "QuantizedMaxPool");
}

static Status TranslateQuantizeV2Op(const Node* op,
                                    const std::vector<const Tensor*>&,
                                    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_min, &ng_max));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "T", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  ng::op::Quantize::RoundMode ng_round_mode;
  string round_mode_string;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "round_mode", &round_mode_string));
  if (round_mode_string == "HALF_UP") {
    ng_round_mode = ng::op::Quantize::RoundMode::ROUND_NEAREST_UPWARD;
  } else if (round_mode_string == "HALF_TO_EVEN") {
    ng_round_mode = ng::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
  } else {
    return errors::Internal("Tensorflow Rounding Mode not supported by Ngraph");
  }

  auto ng_node = ng::builder::QuantizeBuilder(ng_input, ng_min, ng_max, ng_et,
                                              ng::AxisSet(), ng_round_mode);
  Builder::SetTracingInfo(op->name(), ng_node);
  SaveNgOp(ng_op_map, op->name(), ng_node);
  SaveNgOp(ng_op_map, op->name(), ng_min);
  SaveNgOp(ng_op_map, op->name(), ng_max);

  return Status::OK();
}

static Status TranslateDequantizeOp(const Node* op,
                                    const std::vector<const Tensor*>&,
                                    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_min, &ng_max));

  // TF only dequantizes to fp32
  auto ng_node = ng::builder::DequantizeBuilder(
      ng_input, ng_min, ng_max, ng::element::f32, ng::AxisSet());
  Builder::SetTracingInfo(op->name(), ng_node);
  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

static Status TranslateRelu6Op(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto constant_6 = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng_input->get_element_type(), ng_input->get_shape(),
      std::vector<std::string>(ng::shape_size(ng_input->get_shape()), "6"));
  auto relu6_op = ConstructNgNode<ng::opset3::Minimum>(
      op->name(), ConstructNgNode<ng::opset3::Relu>(op->name(), ng_input),
      constant_6);

  SaveNgOp(ng_op_map, op->name(), relu6_op);
  return Status::OK();
}

static Status TranslateReshapeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_shape_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_shape_op));

  NGRAPH_VLOG(3) << "Input shape: " << ng::join(ng_input->get_shape());

  std::vector<int64> shape;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &shape));

  NGRAPH_VLOG(3) << "Requested result shape: " << ng::join(shape);

  auto ng_shape = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::u64, ng::Shape{shape.size()}, shape);

  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::opset3::Reshape>(
                                      op->name(), ng_input, ng_shape, false));
  return Status::OK();
}

static Status TranslateRsqrtOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](std::shared_ptr<ng::Node> n) {
        // Create a constant tensor populated with the value -1/2.
        // (1/sqrt(x) = x^(-1/2))
        auto et = n->get_element_type();
        auto shape = n->get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-0.5");
        auto ng_exponent = ConstructNgNode<ng::opset3::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -0.5.
        return ConstructNgNode<ng::opset3::Power>(op->name(), n, ng_exponent);
      });
}

static Status TranslateScatterNdOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_indices;
  shared_ptr<ng::Node> ng_updates;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_indices, &ng_updates, nullptr));

  std::vector<int> ng_shape;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &ng_shape));
  // Copy the int vector to a size_t vector, because that is what ng::Shape
  // accepts
  std::vector<size_t> ng_shape_size_t(ng_shape.begin(), ng_shape.end());

  // Create a tensor and populate the tensor with "0" to Add to ScatterNd
  auto et = ng_updates->get_element_type();
  std::vector<std::string> constant_values(ng::shape_size(ng_shape_size_t),
                                           "0");
  auto ng_inputs = ConstructNgNode<ng::op::Constant>(
      op->name(), et, ng::Shape(ng_shape_size_t), constant_values);

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::ScatterNDAdd>(op->name(), ng_inputs,
                                                 ng_indices, ng_updates));

  return Status::OK();
}

static Status TranslateShapeOp(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  // the shape of the input tensor which will be the value to the Constant Op
  auto input_shape = ng_input->get_shape();

  // the rank of the input tensor which will be the shape to the Constant Op
  size_t rank = input_shape.size();

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  // the inputs to the Constant Op
  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

  auto shape = ng::Shape(1, rank);

  std::vector<int> values(rank);
  for (size_t i = 0; i < rank; i++) {
    values[i] = input_shape[i];
  }
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ng::op::Constant>(op->name(), type, shape, values));
  return Status::OK();
}

static Status TranslateSizeOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  // Size has an attribute to specify output, int32 or int64
  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

  auto ng_input_shape = ng_input->get_shape();

  int64 result = 1;
  for (auto dim : ng_input_shape) {
    result *= dim;
  }

  // make a scalar with value equals to result
  auto ng_result = ConstructNgNode<ng::opset3::Constant>(
      op->name(), type, ng::Shape(0), std::vector<int64>({result}));

  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

static Status TranslateSliceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_begin, ng_size;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input, &ng_begin, &ng_size));

  std::vector<int64> lower_vec;
  std::vector<int64> size_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &lower_vec));
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &size_vec));

  if (lower_vec.size() != size_vec.size())
    return errors::InvalidArgument(
        "Cannot translate sliceop: Size of lower = ", lower_vec.size(),
        ", size of size_vec = ", size_vec.size(), ". Expected them to match.");

  NGRAPH_VLOG(3) << "Begin input for Slice: " << ng::join(lower_vec);
  NGRAPH_VLOG(3) << "Size input for Slice: " << ng::join(size_vec);

  std::vector<int> upper_vec(lower_vec.size());
  const auto ng_input_shape = ng_input->get_shape();
  stringstream err_stream;
  string err_msg;
  for (size_t i = 0; i < size_vec.size(); i++) {
    if (size_vec[i] != -1) {
      upper_vec[i] = lower_vec[i] + size_vec[i];
    } else {
      // support -1 for size_vec, to the end of the tensor
      upper_vec[i] = ng_input_shape[i];
    }

    // check for this condition: 0 <= begin[i] <= begin[i] + size[i] <= Di
    if (0 > lower_vec[i])
      err_stream << "lower < 0: " << lower_vec[i]
                 << ". It should have been positive.\n";
    if (lower_vec[i] > upper_vec[i])
      err_stream << "upper < lower: upper = " << upper_vec[i]
                 << ", lower = " << lower_vec[i] << "\n";
    if (upper_vec[i] > ng_input_shape[i])
      err_stream << "dim < upper: dim = " << ng_input_shape[i]
                 << ", upper = " << upper_vec[i] << "\n";

    err_msg = err_stream.str();
    if (!err_msg.empty())
      return errors::InvalidArgument("Cannot translate sliceop at position ", i,
                                     " of ", size_vec.size(),
                                     ". The reasons are:\n", err_msg);
  }

  std::vector<size_t> l(lower_vec.begin(), lower_vec.end());
  std::vector<size_t> u(upper_vec.begin(), upper_vec.end());
  auto ng_slice = ConstructNgNode<ng::op::Slice>(op->name(), ng_input, l, u);
  SaveNgOp(ng_op_map, op->name(), ng_slice);
  return Status::OK();
}

static Status TranslateSoftmaxOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto ng_input_shape = ng_input->get_shape();
  auto rank = ng_input_shape.size();
  if (rank < 1) {
    return errors::InvalidArgument("TF Softmax logits must be >=1 dimension");
  }

  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::opset3::Softmax>(
                                      op->name(), ng_input, rank - 1));
  return Status::OK();
}

// Translate SpaceToDepthOp
static Status TranslateSpaceToDepthOp(const Node* op,
                                      const std::vector<const Tensor*>&,
                                      Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  // Get the attributes
  int64 block_size;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "block_size", &block_size));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  ng::Shape input_shape = ng_input->get_shape();
  std::map<std::string, int> format_to_int_map = {
      {"NHWC", 0}, {"NCHW", 1}, {"NCHW_VECT_C", 1}};

  int height_index;
  int width_index;
  int channel_index;

  switch (format_to_int_map[tf_data_format]) {
    // NHWC
    case 0:
      height_index = 1;
      width_index = 2;
      channel_index = 3;
      break;
    // NCHW
    case 1:
      height_index = 2;
      width_index = 3;
      channel_index = 1;
      break;
    // NCHW_VEC_C
    case 2:
      return errors::InvalidArgument(
          "NCHW_VECT_C is not supported in SpaceToDepth for now");
    default:
      return errors::InvalidArgument(
          "SpaceToDepth supported data format is NCHW, NHWC, or NCHW_VECT_C");
  }

  // Error checking: width and height must be divisible by block_size
  if (input_shape[height_index] % block_size != 0) {
    return errors::InvalidArgument(
        "Input tensor's height ,", input_shape[height_index],
        " is not divisible by block_size ", block_size);
  }

  if (input_shape[width_index] % block_size != 0) {
    return errors::InvalidArgument(
        "Input tensor's width ,", input_shape[width_index],
        " is not divisible by block_size ", block_size);
  }

  // Upper indexes will be the same for all strided slices
  std::vector<size_t> upper = {input_shape[0], input_shape[1], input_shape[2],
                               input_shape[3]};
  // Store the strided_slice result for concat
  std::vector<std::shared_ptr<ng::Node>> strided_slice_result;

  for (int counter_height = 0; counter_height < block_size; counter_height++) {
    for (int counter_width = 0; counter_width < block_size; counter_width++) {
      std::vector<size_t> begin = {0, 0, 0, 0};
      begin[width_index] = counter_width;
      begin[height_index] = counter_height;
      std::vector<size_t> strides = {1, 1, 1, 1};
      strides[width_index] = size_t(block_size);
      strides[height_index] = size_t(block_size);
      strided_slice_result.push_back(ConstructNgNode<ng::op::Slice>(
          op->name(), ng_input, begin, upper, strides));
    }
  }

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ngraph::op::Concat>(op->name(), strided_slice_result,
                                               channel_index));
  return Status::OK();
}

static Status TranslateSplitOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, nullptr, &ng_input));
  // num_split : The number of ways to split. Must evenly divide
  // value.shape[split_dim]
  int32 num_split;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_split", &num_split));

  ng::Shape shape = ng_input->get_shape();
  int rank = shape.size();
  std::vector<size_t> lower;
  std::vector<size_t> upper;
  for (int i = 0; i < rank; ++i) {
    lower.push_back(0);
    upper.push_back(shape[i]);
  }
  std::vector<int> split_dim_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 0, static_input_map, &split_dim_vec));
  int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64)rank : 0);

  int size = shape[split_dim] / num_split;
  int cursor = 0;

  for (int i = 0; i < num_split; ++i) {
    lower[split_dim] = cursor;
    cursor += size;
    upper[split_dim] = cursor;

    std::string output_name = op->name();
    SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::op::Slice>(
                                        op->name(), ng_input, lower, upper));
  }
  return Status::OK();
}

static Status TranslateSplitVOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_length, ng_split_dim;

  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input));

  std::vector<int> lengths;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &lengths));

  ng::Shape shape = ng_input->get_shape();
  int rank = shape.size();
  std::vector<size_t> lower(rank, 0);
  std::vector<size_t> upper(shape);

  std::vector<int64> split_dim_vec;

  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 2, static_input_map, &split_dim_vec));

  // there should be at least one element specified as axis and not more than
  // one
  // as axis is 0-D
  if (split_dim_vec.size() != 1) {
    return errors::InvalidArgument(
        "split_dim_tensor must have "
        "exactly one element.");
  }

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(split_dim_vec, rank));

  int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64)rank : 0);

  // length: Length of size_splits
  int length = 0;
  int idx = -1;

  // Find out the total length of the splits and locate -1 's index, if any
  bool has_one_neg = false;
  for (size_t i = 0; i < lengths.size(); ++i) {
    if (lengths[i] != -1) {
      length += lengths[i];
    } else {
      if (has_one_neg) {
        return errors::InvalidArgument("size_splits can only have one -1");
      } else {
        idx = i;
        has_one_neg = true;
      }
    }
  }

  // Size splits must sum to the dimension of value along split_dim
  if (idx > 0) {
    lengths[idx] = shape[split_dim] - length;
  }

  if ((!has_one_neg && length != shape[split_dim]) ||
      (has_one_neg && lengths[idx] < 0)) {
    return errors::InvalidArgument(
        "The length of size_splits must sum to the value of the dimension "
        "along split_dim");
  }

  int cursor = 0;

  if (lengths.size() != 1) {
    for (size_t i = 0; i < lengths.size(); ++i) {
      lower[split_dim] = cursor;
      cursor += lengths[i];
      upper[split_dim] = cursor;
      SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::op::Slice>(
                                          op->name(), ng_input, lower, upper));
    }
  } else {
    SaveNgOp(ng_op_map, op->name(), ng_input);
  }

  return Status::OK();
}

static Status TranslateSquareOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](std::shared_ptr<ng::Node> n) {
        return ConstructNgNode<ng::opset3::Multiply>(op->name(), n, n);
      });
}

static Status TranslateSqueezeOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));
  size_t input_dims = ng_input->get_shape().size();

  std::vector<int32> tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "squeeze_dims", &tf_axis));

  // If input dimension is negative, make it positive
  for (size_t i = 0; i < tf_axis.size(); i++) {
    tf_axis[i] = tf_axis[i] < 0 ? (int32)(input_dims) + tf_axis[i] : tf_axis[i];
  }

  auto ng_const = ConstructNgNode<ng::opset3::Constant>(
      op->name(), ng::element::u32, ng::Shape{tf_axis.size()}, tf_axis);

  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<ng::opset3::Squeeze>(
                                      op->name(), ng_input, ng_const));

  return Status::OK();
}

static Status TranslateStridedSliceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input));

  strided_slice_mask_attrs mask_attrs;
  TF_RETURN_IF_ERROR(GetStridedSliceAttrs(op, mask_attrs));

  std::vector<int64> begin_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &begin_vec));

  std::vector<int64> end_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &end_vec));

  std::vector<int64> stride_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 3, static_input_map, &stride_vec));

  // Desired implementation ==>
  // SaveNgOp(ng_op_map, op->name(),
  //          ConstructNgNode<ng::op::StridedSlice>(op->name(), begin_vec,
  //          end_vec, stride_vec,
  //                             tf_begin_mask, tf_end_mask,
  //                             tf_new_axis_mask, tf_shrink_axis_mask,
  //                             tf_ellipsis_mask));

  // Temporarily we are borrowing this implementation from nGraph-core until
  // ng::op::StridedSlice is released for use in ngraph-bridge

  ng::Shape input_shape = ng_input->get_shape();

  std::vector<int64_t> begin_vec_longint(begin_vec.begin(), begin_vec.end());
  std::vector<int64_t> end_vec_longint(end_vec.begin(), end_vec.end());
  std::vector<int64_t> stride_vec_longint(stride_vec.begin(), stride_vec.end());

  NGRAPH_VLOG(5) << "Arguments to make_slice_plan: Input shape: " << input_shape
                 << ", begin vector: " << ng::join(begin_vec_longint)
                 << ", end vector: " << ng::join(end_vec_longint)
                 << ", stride vector: " << ng::join(stride_vec_longint);

  auto in_rank = ng_input->get_shape().size();
  if (mask_attrs.new_axis == 0) {
    if (begin_vec_longint.size() > in_rank) {
      return errors::InvalidArgument("Index out of range using input dim ",
                                     begin_vec_longint.size(),
                                     "; input has only ", in_rank, " dims");
    }
  }

  ng::SlicePlan sp =
      GetSlicePlan(input_shape, begin_vec_longint, end_vec_longint,
                   stride_vec_longint, mask_attrs);

  // Need to convert int64_t to size_t
  std::vector<size_t> sp_begins(sp.begins.begin(), sp.begins.end());
  std::vector<size_t> sp_ends(sp.ends.begin(), sp.ends.end());
  std::vector<size_t> sp_strides(sp.strides.begin(), sp.strides.end());

  shared_ptr<ng::Node> ng_result = ConstructNgNode<ng::op::Slice>(
      op->name(), ng_input, sp_begins, sp_ends, sp_strides);

  if (sp.reshape_in_shape != sp.reshape_out_shape) {
    ng::Shape ng_out_shape(sp.reshape_out_shape);
    ng::AxisVector ng_axis_order(sp.reshape_in_shape.size());
    // std::iota Fills the range [first, last) with sequentially increasing
    // values,
    // starting with value and repetitively evaluating ++value
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

    NGRAPH_VLOG(3) << " Output  shape " << ng::join(ng_out_shape);
    NGRAPH_VLOG(3) << " NG  axis order " << ng::join(ng_axis_order);

    ng_result = ConstructNgNode<ng::op::Reshape>(op->name(), ng_result,
                                                 ng_axis_order, ng_out_shape);
  }

  if (!sp.reverse_axes.empty()) {
    ng_result = ConstructNgNode<ng::op::Reverse>(op->name(), ng_result,
                                                 sp.reverse_axes);
  }

  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

static Status TranslateTileOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_multiples;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_multiples));

  std::vector<int64> multiples;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &multiples));

  auto ng_input_shape = ng_input->get_shape();
  if (ng_input_shape.size() != multiples.size()) {
    return errors::InvalidArgument(
        "dimension of input does not match length of multiples");
  }
  std::shared_ptr<ng::Node> ng_output = ng_input;
  ng::Shape output_shape = ng_input_shape;
  bool is_empty = false;
  for (size_t i = 0; i < ng_input_shape.size(); i++) {
    if (multiples[i] == 0) {
      is_empty = true;
    }
    output_shape[i] = ng_input_shape[i] * multiples[i];
  }
  if (is_empty) {
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<ngraph::op::Constant>(
                 op->name(), ng_input->get_element_type(), output_shape,
                 std::vector<std::string>(ng::shape_size(output_shape), "0")));
  } else {
    for (size_t i = 0; i < ng_input_shape.size(); i++) {
      if (multiples[i] < 0) {
        return errors::InvalidArgument("Expected multiples[", i,
                                       "] >= 0, but got ", multiples[i]);
      }
      vector<shared_ptr<ng::Node>> tmp_tensors;
      for (int k = 0; k < multiples[i]; k++) {
        tmp_tensors.push_back(ng_output);
      }
      auto ng_concat =
          ConstructNgNode<ngraph::op::Concat>(op->name(), tmp_tensors, i);
      ng_output = ng_concat;
    }
    SaveNgOp(ng_op_map, op->name(), ng_output);
  }
  return Status::OK();
}

// Translate TopKV2 Op using ngraph core op TopK
static Status TranslateTopKV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ngraph::Node> ng_input;
  TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input));

  size_t k_axis = ng_input->get_shape().size() - 1;

  std::vector<int32> ng_k;
  size_t k;
  bool sorted = true;

  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &ng_k));
  k = ng_k[0];

  // sorted = false is not supported right now, it falls back to TF if set to
  // false.
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "sorted", &sorted));

  // index element type - currently only int32 or int64 are supported by
  // ngraph

  shared_ptr<ngraph::Node> ng_result = ConstructNgNode<ngraph::op::TopK>(
      op->name(), ng_input, k_axis, ng::element::i32, k, sorted);

  shared_ptr<ngraph::Node> ng_values =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_result, 1);
  shared_ptr<ngraph::Node> ng_indices =
      ConstructNgNode<ngraph::op::GetOutputElement>(op->name(), ng_result, 0);

  SaveNgOp(ng_op_map, op->name(), ng_values);
  SaveNgOp(ng_op_map, op->name(), ng_indices);

  return Status::OK();
}

static Status TranslateTransposeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_permutation_op;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input, &ng_permutation_op));

  std::vector<int64> permutation;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 1, static_input_map, &permutation));

  // Check to make sure that the permutation requested for transpose
  // is valid for example:
  // - it should not have duplicates,
  // - it should have all the dimensions.

  int ng_input_rank = ng_input->get_shape().size();
  vector<bool> count(ng_input_rank, false);
  for (auto p : permutation) {
    if (0 <= p && p < ng_input_rank) {
      count[p] = true;
    }
  }
  for (int i = 0; i < ng_input_rank; i++) {
    if (!count[i]) {
      return errors::InvalidArgument(i, " is missing from {",
                                     ng::join(permutation), "}.");
    }
  }

  ng::AxisVector ng_axis_order;
  ng_axis_order.reserve(permutation.size());

  NGRAPH_VLOG(3) << ng::join(permutation);

  for (auto i : permutation) {
    ng_axis_order.push_back(i);
  }

  NGRAPH_VLOG(3) << ng::join(ng_axis_order);

  auto ng_node = ng::builder::numpy_transpose(ng_input, ng_axis_order);
  Builder::SetTracingInfo(op->name(), ng_node);
  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

static Status TranslateUnpackOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  int32 tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  auto unpack_axis = tf_axis;
  if (unpack_axis == -1) {
    unpack_axis = input_rank - 1;
  }

  int32 tf_num;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num", &tf_num));
  int num_outputs = tf_num;

  ng::Shape output_shape;
  for (size_t i = 0; i < input_rank; ++i) {
    if ((int)i != unpack_axis) {
      output_shape.push_back(input_shape[i]);
    }
  }

  ng::AxisVector ng_axis_order;
  for (size_t i = 0; i < input_rank; i++) {
    ng_axis_order.push_back(i);
  }

  std::vector<size_t> lower_bound(input_rank, 0);
  std::vector<size_t> upper_bound(input_rank);

  for (size_t i = 0; i < input_rank; i++) {
    upper_bound[i] = input_shape[i];
  }

  for (int i = 0; i < num_outputs; ++i) {
    lower_bound[unpack_axis] = i;
    upper_bound[unpack_axis] = i + 1;
    auto slice = ConstructNgNode<ngraph::op::Slice>(op->name(), ng_input,
                                                    lower_bound, upper_bound);
    auto reshaped = ConstructNgNode<ng::op::Reshape>(
        op->name(), slice, ng_axis_order, output_shape);
    SaveNgOp(ng_op_map, op->name(), reshaped);
  }
  return Status::OK();
}

static Status TranslateXdivyOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input1, ng_input2;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input1, &ng_input2));
  std::tie(ng_input1, ng_input2) =
      Builder::PerformNgBroadcast(op->name(), ng_input1, ng_input2);
  ng::Shape input1_shape = ng_input1->get_shape();
  std::vector<std::string> const_values(ng::shape_size(input1_shape), "0");
  auto const_zero = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_input1->get_element_type(), input1_shape, const_values);
  auto ng_is_zero =
      ConstructNgNode<ng::op::Equal>(op->name(), ng_input1, const_zero);
  auto ng_x_over_y =
      ConstructNgNode<ng::op::Divide>(op->name(), ng_input1, ng_input2);
  auto ng_xdivy = ConstructNgNode<ng::op::Select>(op->name(), ng_is_zero,
                                                  ng_input1, ng_x_over_y);
  SaveNgOp(ng_op_map, op->name(), ng_xdivy);
  return Status::OK();
}

static Status TranslateUnsortedSegmentSumOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_segment_ids;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input, &ng_segment_ids, nullptr));

  int num_segments;
  std::vector<int64> tmp_num_segments;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 2, static_input_map, &tmp_num_segments));

  if (tmp_num_segments.size() != 1) {
    return errors::InvalidArgument(
        "num_segments should be scalar, not tensor with ",
        tmp_num_segments.size(), " dimensions");
  }

  num_segments = tmp_num_segments[0];

  auto& input_shape = ng_input->get_shape();
  auto& segment_shape = ng_segment_ids->get_shape();

  ng::Shape output_shape;
  output_shape.push_back(num_segments);
  output_shape.insert(output_shape.end(),
                      input_shape.begin() + segment_shape.size(),
                      input_shape.end());

  auto result = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_input->get_element_type(), output_shape,
      std::vector<std::string>(ng::shape_size(output_shape), "0"));

  auto unsorted_segment_sum = ConstructNgNode<ng::op::ScatterAdd>(
      op->name(), result, ng_segment_ids, ng_input);

  SaveNgOp(ng_op_map, op->name(), unsorted_segment_sum);
  return Status::OK();
}

static Status TranslateSelectOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input1, ng_input2, ng_input3;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input1, &ng_input2, &ng_input3));
  shared_ptr<ng::Node> ng_select;
  ng_select = ConstructNgNode<ng::opset3::Select>(op->name(), ng_input1,
                                                  ng_input2, ng_input3);
  SaveNgOp(ng_op_map, op->name(), ng_select);
  return Status::OK();
}

static Status TranslateZerosLikeOp(const Node* op,
                                   const std::vector<const Tensor*>&,
                                   Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  ng::Shape input_shape = ng_input->get_shape();
  std::vector<std::string> const_values(ng::shape_size(input_shape), "0");
  auto ng_result = ConstructNgNode<ng::op::Constant>(
      op->name(), ng_input->get_element_type(), input_shape, const_values);

  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

const static std::map<
    const string,
    const function<Status(const Node*, const std::vector<const Tensor*>&,
                          Builder::OpMap&)>>
    TRANSLATE_OP_MAP{
        {"Abs", TranslateUnaryOp<ngraph::opset3::Abs>},
        {"Acos", TranslateUnaryOp<ngraph::opset3::Acos>},
        {"Add", TranslateBinaryOp<ngraph::opset3::Add>},
        {"AddN", TranslateAddNOp},
        {"AddV2", TranslateBinaryOp<ngraph::opset3::Add>},
        {"Any", TranslateDirectReduceOp<ng::opset3::ReduceLogicalOr>},
        {"All", TranslateDirectReduceOp<ng::opset3::ReduceLogicalAnd>},
        {"ArgMax", TranslateArgMinMaxOp<ng::op::ArgMax>},
        {"ArgMin", TranslateArgMinMaxOp<ng::op::ArgMin>},
        {"Asin", TranslateUnaryOp<ngraph::opset3::Asin>},
        {"Atan", TranslateUnaryOp<ngraph::opset3::Atan>},
        {"Atan2", TranslateBinaryOp<ngraph::op::Atan2>},
        {"AvgPool", TranslateAvgPoolOp},
        {"BatchMatMul", TranslateBatchMatMulOp},
        {"BatchMatMulV2", TranslateBatchMatMulV2Op},
        {"BiasAdd", TranslateBiasAddOp},
        {"Cast", TranslateCastOp},
        {"Ceil", TranslateUnaryOp<ngraph::opset3::Ceiling>},
        {"ConcatV2", TranslateConcatV2Op},
        {"Const", TranslateConstOp},
        {"Conv2D", TranslateConv2DOp},
        {"Conv2DBackpropInput", TranslateConv2DBackpropInputOp},
        {"Conv3D", TranslateConv3DOp},
        {"Cos", TranslateUnaryOp<ngraph::opset3::Cos>},
        {"Cosh", TranslateUnaryOp<ngraph::opset3::Cosh>},
        {"CropAndResize", TranslateCropAndResizeOp},
        {"Cumsum", TranslateCumsumOp},
        {"DepthToSpace", TranslateDepthToSpaceOp},
        {"DepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
        {"Dequantize", TranslateDequantizeOp},
        {"Equal", TranslateBinaryOp<ngraph::opset3::Equal>},
        {"Exp", TranslateUnaryOp<ngraph::opset3::Exp>},
        {"ExpandDims", TranslateExpandDimsOp},
        {"Fill", TranslateFillOp},
        {"Floor", TranslateUnaryOp<ngraph::opset3::Floor>},
        {"FloorDiv", TranslateFloorDivOp},
        {"FloorMod", TranslateBinaryOp<ngraph::opset3::FloorMod>},
        {"FusedBatchNorm", TranslateFusedBatchNormOp},
        {"FusedBatchNormV2", TranslateFusedBatchNormOp},
        {"FusedBatchNormV3", TranslateFusedBatchNormOp},
        {"GatherNd", TranslateGatherNdOp},
        {"GatherV2", TranslateGatherV2Op},
        {"_FusedConv2D", TranslateFusedConv2DOp},
        {"_FusedMatMul", TranslateFusedMatMulOp},
        {"Greater", TranslateBinaryOp<ngraph::opset3::Greater>},
        {"GreaterEqual", TranslateBinaryOp<ngraph::opset3::GreaterEqual>},
        {"Identity", TranslateIdentityOp},
        {"IsFinite", TranslateIsFiniteOp},
        {"L2Loss", TranslateL2LossOp},
        {"LogSoftmax", TranslateLogSoftmaxOp},
        {"Less", TranslateBinaryOp<ngraph::opset3::Less>},
        {"LessEqual", TranslateBinaryOp<ngraph::opset3::LessEqual>},
        {"Log", TranslateUnaryOp<ngraph::opset3::Log>},
        {"Log1p", TranslateLog1pOp},
        {"LogicalAnd", TranslateBinaryOp<ngraph::opset3::LogicalAnd>},
        {"LogicalNot", TranslateUnaryOp<ngraph::opset3::LogicalNot>},
        {"LogicalOr", TranslateBinaryOp<ngraph::opset3::LogicalOr>},
        {"MatMul", TranslateMatMulOp},
        {"Max", TranslateDirectReduceOp<ng::opset3::ReduceMax>},
        {"Maximum", TranslateBinaryOp<ngraph::opset3::Maximum>},
        {"MaxPool", TranslateMaxPoolOp},
        {"MaxPool3D", TranslateMaxPool3DOp},
        {"NonMaxSuppressionV4", TranslateNonMaxSuppressionV4Op},
        {"Mean", TranslateDirectReduceOp<ng::opset3::ReduceMean>},
        {"Min", TranslateDirectReduceOp<ng::opset3::ReduceMin>},
        {"Minimum", TranslateBinaryOp<ngraph::opset3::Minimum>},
        {"MirrorPad", TranslatePadOp},
        {"Mul", TranslateBinaryOp<ngraph::opset3::Multiply>},
        {"Neg", TranslateUnaryOp<ngraph::opset3::Negative>},
        // Do nothing! NoOps sometimes get placed on nGraph for bureaucratic
        // reasons, but they have no data flow inputs or outputs.
        {"NoOp", [](const Node*, const std::vector<const Tensor*>&,
                    Builder::OpMap&) { return Status::OK(); }},
        {"OneHot", TranslateOneHotOp},
        {"Pack", TranslatePackOp},
        {"Pad", TranslatePadOp},
        {"PadV2", TranslatePadOp},
        {"Pow", TranslateBinaryOp<ngraph::opset3::Power>},
        // PreventGradient is just Identity in data-flow terms, so reuse that.
        {"PreventGradient", TranslateIdentityOp},
        {"Prod", TranslateDirectReduceOp<ng::opset3::ReduceProd>},
        {"QuantizeAndDequantizeV2", TranslateQuantizeAndDequantizeV2Op},
        {"QuantizedAvgPool", TranslateQuantizedAvgPoolOp},
        {"QuantizedConcat", TranslateQuantizedConcatOp},
        {"QuantizedConcatV2", TranslateQuantizedConcatV2Op},
        {"QuantizedConv2DWithBiasAndReluAndRequantize",
         TranslateQuantizedConv2DWithBiasMaybeReluAndRequantizeOp<true>},
        {"QuantizedConv2DWithBiasAndRequantize",
         TranslateQuantizedConv2DWithBiasMaybeReluAndRequantizeOp<false>},
        {"QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
         TranslateQuantizedConv2DWithBiasSignedSumAndReluAndRequantizeOp},
        {"QuantizedConv2DWithBiasSumAndReluAndRequantize",
         TranslateQuantizedConv2DWithBiasSumAndReluAndRequantizeOp},
        {"QuantizedMaxPool", TranslateQuantizedMaxPoolOp},
        {"QuantizeV2", TranslateQuantizeV2Op},
        {"Rank", TranslateRankOp},
        {"RealDiv", TranslateBinaryOp<ngraph::opset3::Divide>},
        {"Reciprocal", TranslateReciprocalOp},
        {"Relu", TranslateUnaryOp<ngraph::opset3::Relu>},
        {"Relu6", TranslateRelu6Op},
        {"Reshape", TranslateReshapeOp},
        {"Rsqrt", TranslateRsqrtOp},
        {"ScatterNd", TranslateScatterNdOp},
        {"Select", TranslateSelectOp},
        {"Shape", TranslateShapeOp},
        {"Sigmoid", TranslateUnaryOp<ngraph::opset3::Sigmoid>},
        {"Sin", TranslateUnaryOp<ngraph::opset3::Sin>},
        {"Sinh", TranslateUnaryOp<ngraph::opset3::Sinh>},
        {"Size", TranslateSizeOp},
        {"Sign", TranslateUnaryOp<ngraph::opset3::Sign>},
        {"Slice", TranslateSliceOp},
        {"Snapshot", TranslateIdentityOp},
        {"Softmax", TranslateSoftmaxOp},
        {"Softplus", TranslateSoftplusOp},
        {"SpaceToDepth", TranslateSpaceToDepthOp},
        {"Split", TranslateSplitOp},
        {"SplitV", TranslateSplitVOp},
        {"Sqrt", TranslateUnaryOp<ngraph::opset3::Sqrt>},
        {"Square", TranslateSquareOp},
        {"SquaredDifference",
         TranslateBinaryOp<ngraph::opset3::SquaredDifference>},
        {"Squeeze", TranslateSqueezeOp},
        {"StridedSlice", TranslateStridedSliceOp},
        {"Sub", TranslateBinaryOp<ngraph::opset3::Subtract>},
        {"Sum", TranslateDirectReduceOp<ng::opset3::ReduceSum>},
        {"Tan", TranslateUnaryOp<ngraph::opset3::Tan>},
        {"Tanh", TranslateUnaryOp<ngraph::opset3::Tanh>},
        {"Tile", TranslateTileOp},
        {"TopKV2", TranslateTopKV2Op},
        {"Transpose", TranslateTransposeOp},
        {"UnsortedSegmentSum", TranslateUnsortedSegmentSumOp},
        {"Unpack", TranslateUnpackOp},
        {"Xdivy", TranslateXdivyOp},
        {"ZerosLike", TranslateZerosLikeOp}};

Status Builder::TranslateGraph(
    const std::vector<TensorShape>& inputs,
    const std::vector<const Tensor*>& static_input_map,
    const Graph* input_graph, shared_ptr<ng::Function>& ng_function) {
  //
  // We will visit ops in topological order.
  //
  // ought to be `const Node*`, but GetReversePostOrder doesn't use `const`

  vector<Node*> ordered;
  GetReversePostOrder(*input_graph, &ordered, NodeComparatorName());

  //
  // Split ops into params, retvals, and all others.
  //
  vector<const Node*> tf_params;
  vector<const Node*> tf_ret_vals;
  vector<const Node*> tf_ops;

  for (const auto n : ordered) {
    if (n->IsSink() || n->IsSource()) {
      continue;
    }

    if (n->IsControlFlow()) {
      return errors::Unimplemented(
          "Encountered a control flow op in the nGraph bridge: ",
          n->DebugString());
    }

    if (n->IsArg()) {
      tf_params.push_back(n);
    } else if (n->IsRetval()) {
      tf_ret_vals.push_back(n);
    } else {
      tf_ops.push_back(n);
    }
  }

  //
  // The op map holds a mapping from TensorFlow op names (strings) to
  // vector of generated nGraph nodes.
  //
  Builder::OpMap ng_op_map;

  //
  // Populate the parameter list, and also put parameters into the op map.
  //
  vector<shared_ptr<ng::op::Parameter>> ng_parameter_list(tf_params.size());

  for (auto parm : tf_params) {
    DataType dtype;
    if (GetNodeAttr(parm->attrs(), "T", &dtype) != Status::OK()) {
      return errors::InvalidArgument("No data type defined for _Arg");
    }
    int index;
    if (GetNodeAttr(parm->attrs(), "index", &index) != Status::OK()) {
      return errors::InvalidArgument("No index defined for _Arg");
    }

    ng::element::Type ng_et;
    TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

    ng::Shape ng_shape;
    TF_RETURN_IF_ERROR(TFTensorShapeToNGraphShape(inputs[index], &ng_shape));

    string prov_tag;
    GetNodeAttr(parm->attrs(), "_prov_tag", &prov_tag);
    auto ng_param =
        ConstructNgNode<ng::op::Parameter>(prov_tag, ng_et, ng_shape);
    SaveNgOp(ng_op_map, parm->name(), ng_param);
    ng_parameter_list[index] = ng_param;
  }

  //
  // Now create the nGraph ops from TensorFlow ops.
  //
  for (auto op : tf_ops) {
    NGRAPH_VLOG(2) << "Constructing op " << op->name() << " which is "
                   << op->type_string();

    const function<Status(const Node*, const std::vector<const Tensor*>&,
                          Builder::OpMap&)>* op_fun;

    try {
      op_fun = &(TRANSLATE_OP_MAP.at(op->type_string()));
    } catch (const std::out_of_range&) {
      // -----------------------------
      // Catch-all for unsupported ops
      // -----------------------------
      NGRAPH_VLOG(3) << "No translation handler registered for op: "
                     << op->name() << " (" << op->type_string() << ")";
      NGRAPH_VLOG(3) << op->def().DebugString();
      return errors::InvalidArgument(
          "No translation handler registered for op: ", op->name(), " (",
          op->type_string(), ")\n", op->def().DebugString());
    }

    try {
      TF_RETURN_IF_ERROR((*op_fun)(op, static_input_map, ng_op_map));
    } catch (const std::exception& e) {
      return errors::Internal("Unhandled exception in op handler: ", op->name(),
                              " (", op->type_string(), ")\n",
                              op->def().DebugString(), "\n", "what(): ",
                              e.what());
    }
  }

  //
  // Populate the result list.
  //
  vector<shared_ptr<ng::Node>> ng_result_list(tf_ret_vals.size());

  for (auto n : tf_ret_vals) {
    // Make sure that this _Retval only has one input node.
    if (n->num_inputs() != 1) {
      return errors::InvalidArgument("_Retval has ", n->num_inputs(),
                                     " inputs, should have 1");
    }

    int index;
    if (GetNodeAttr(n->attrs(), "index", &index) != Status::OK()) {
      return errors::InvalidArgument("No index defined for _Retval");
    }

    shared_ptr<ng::Node> result;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, n, 0, &result));

    ng_result_list[index] = result;
  }

  //
  // Create the nGraph function.
  //
  ng_function = make_shared<ng::Function>(ng_result_list, ng_parameter_list);

  //
  // Request row-major layout on results.
  //
  for (auto result : ng_function->get_results()) {
    result->set_needs_default_layout(true);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
