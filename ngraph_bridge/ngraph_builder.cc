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

#include "ngraph/op/util/logical_reduction.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/slice_plan.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/default_opset.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_conversions.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/pass/transpose_sinking.h"

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
//    ng::Output<ng::Node> output_node - ng::Node to store
//

static void SaveNgOp(Builder::OpMap& ng_op_map, const std::string& op_name,
                     ng::Output<ng::Node> output_node) {
  // no need to try-catch, map[key] will create vector object
  // if not exists
  ng_op_map[op_name].push_back(output_node);
}

void Builder::SetTracingInfo(const std::string& op_name,
                             const ng::Output<ng::Node> ng_node) {
  auto node = ng_node.get_node_shared_ptr();
  node->set_friendly_name(op_name + "/" + node->get_name());
  node->add_provenance_tag(op_name);
  if (config::IsLoggingPlacement()) {
    cout << "TF_to_NG: " << op_name << " --> " << node << "\n";
  }
}

template <class TOpType, class... TArg>
ng::Output<ng::Node> ConstructNgNode(const std::string& op_name,
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
//      ng::Output<ng::Node> ng_input;
//      try {
//        ng_input = ng_op_map.at(tf_input->name());
//      } catch (const std::out_of_range&) {
//        return errors::NotFound(tf_input->name(),
//                                    " is not found in the ng_op_map");
//      }
//
// Into 2 lines:
//
//      ng::Output<ng::node> ng_input;
//      TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input))
//
//
//
// Parameters:
//    Builder::OpMap& ng_op_map     - The TF-to-nGraph op map.
//    Node* op                  - TF op being translated.
//    input_idx                     - index of input
//
//    ng::Output<ng::Node> *result  - ng::Node pointer where result
//                                    will be written
//
//

static Status GetInputNode(const Builder::OpMap& ng_op_map, const Node* op,
                           size_t input_idx, ng::Output<ng::Node>& result) {
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
  std::vector<ng::Output<ng::Node>> ng_op;
  try {
    ng_op = ng_op_map.at(tf_input->name());
  } catch (const out_of_range&) {
    return Status(error::NOT_FOUND,
                  string("Ngraph op not found for ") + tf_input->name());
  }
  try {
    result = ng_op.at(src_output_idx);
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
                            size_t index, ng::Output<ng::Node>& result,
                            Arguments&... remaining) {
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, index, result));
  return GetInputNodes(ng_op_map, op, index + 1, remaining...);
}
}  // namespace detail

template <typename... Arguments>
static Status GetInputNodes(const Builder::OpMap& ng_op_map, const Node* op,
                            Arguments&... remaining) {
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
                          ng::Output<ng::Node>& ng_node) {
  vector<VecT> const_values;
  TensorShapeProto shape_proto;

  TF_RETURN_IF_ERROR(
      ValuesFromConstNode<T, VecT>(op->def(), &shape_proto, &const_values));

  TensorShape const_shape(shape_proto);

  ng::Shape ng_shape;
  TF_RETURN_IF_ERROR(TFTensorShapeToNGraphShape(const_shape, &ng_shape));

  ng_node =
      ConstructNgNode<opset::Constant>(op->name(), et, ng_shape, const_values);
  return Status::OK();
}

const Builder::ConstMap& Builder::TF_NGRAPH_CONST_MAP() {
  static const Builder::ConstMap the_map = {
      {DataType::DT_FLOAT, make_pair(MakeConstOp<float>, ng::element::f32)},
      {DataType::DT_DOUBLE, make_pair(MakeConstOp<double>, ng::element::f64)},
      {DataType::DT_INT8, make_pair(MakeConstOp<int8>, ng::element::i8)},
      {DataType::DT_INT16, make_pair(MakeConstOp<int16>, ng::element::i16)},
      {DataType::DT_QINT8, make_pair(MakeConstOp<qint8>, ng::element::i8)},
      {DataType::DT_QUINT8, make_pair(MakeConstOp<quint8>, ng::element::u8)},
      {DataType::DT_QUINT16, make_pair(MakeConstOp<quint16>, ng::element::u16)},
      {DataType::DT_INT32, make_pair(MakeConstOp<int32>, ng::element::i32)},
      {DataType::DT_INT64, make_pair(MakeConstOp<int64>, ng::element::i64)},
      {DataType::DT_UINT8, make_pair(MakeConstOp<uint8>, ng::element::u8)},
      {DataType::DT_UINT16, make_pair(MakeConstOp<uint16>, ng::element::u16)},
      {DataType::DT_BOOL,
       make_pair(MakeConstOp<bool, char>, ng::element::boolean)}};
  return the_map;
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
//    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>>
//      create_unary_op           - Function to construct the graph implementing
//                                 the unary op, given the input to the unop
//                                 as an argument.
//
// Example usage:
//
//  if (n->type_string == "Square") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp(n, static_input_map, ng_op_map,
//                       [] (ng::Output<ng::Node> n) {
//                           return
//                           (ng::Output<opset::Multiply>(n,n));
//                       });
//  }
static Status TranslateUnaryOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map,
    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>)> create_unary_op) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
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
                          [&op](ng::Output<ng::Node> n) {
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
//    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>,
//    ng::Output<ng::Node>)>
//    create_binary_op           - Function to construct the graph implementing
//                                 the binary op, given the 2 ng_inputs to the
//                                 binaryop
// Example Usage:
//
// if (op->type_string() == "SquaredDifference") {
//      TF_RETURN_IF_ERROR(TranslateBinaryOp(op, ng_op_map,
//         [](ng::Output<ng::Node> ng_input1, ng::Output<ng::Node>
//         ng_input2) {
//           auto ng_diff = ng::Output<opset::Subtract>(input1,
//           input2);
//           return ng::Output<opset::Multiply>(ng_diff,ng_diff);
//         }));
//    }
//

static Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map,
    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>&,
                                       ng::Output<ng::Node>&)>
        create_binary_op) {
  ng::Output<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs));
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
//    TF_RETURN_IF_ERROR(TranslateBinaryOp<opset::Add>(op,
//    static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
static Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateBinaryOp(
      op, static_input_map, ng_op_map,
      [&op](ng::Output<ng::Node>& ng_lhs, ng::Output<ng::Node>& ng_rhs) {
        return ConstructNgNode<T>(op->name(), ng_lhs, ng_rhs);
      });
}

static Status TranslateAddNOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  std::vector<ng::Output<ng::Node>> ng_arg_vec(op->num_inputs());

  for (int inp_idx = 0; inp_idx < op->num_inputs(); inp_idx++)
    TF_RETURN_IF_ERROR(
        GetInputNode(ng_op_map, op, inp_idx, ng_arg_vec[inp_idx]));
  auto ng_addn = std::accumulate(
      std::next(ng_arg_vec.begin()), ng_arg_vec.end(), ng_arg_vec.at(0),
      [&op](ng::Output<ng::Node> a, ng::Output<ng::Node> b) {
        return ConstructNgNode<opset::Add>(op->name(), a, b);
      });  // accumulation: start with
           // first element. default op is
           // addition
  SaveNgOp(ng_op_map, op->name(), ng_addn);
  return Status::OK();
}
static Status TranslateArgMinMax(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map, std::string mode) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  std::vector<int64> tf_dim;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &tf_dim));

  ng::Shape input_shape = ng_input.get_shape();
  size_t input_rank = input_shape.size();

  if (tf_dim.size() != 1) {
    return errors::InvalidArgument(
        "ArgMax Op: dimension must be scalar, operates on a single axis");
  }

  // If input dimension is negative, make it positive
  if (tf_dim[0] < 0) {
    NGRAPH_VLOG(3) << "Input dimension is negative, make it positive "
                   << tf_dim[0];
    tf_dim[0] = (int64)input_rank + tf_dim[0];
  }
  NGRAPH_VLOG(3) << "Axis along which to compute " << tf_dim[0];
  size_t k_axis = tf_dim[0];

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "output_type", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  auto ng_k = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{}, std::vector<int64>({1}));

  std::string sort = "none";
  auto ng_topk =
      std::make_shared<opset::TopK>(ng_input, ng_k, k_axis, mode, sort, ng_et);
  auto ng_indices = ng_topk->output(1);
  int axis = ng_topk->get_axis();
  auto axis_to_remove = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{1}, std::vector<int64>({axis}));
  auto reshaped_indices =
      ConstructNgNode<opset::Squeeze>(op->name(), ng_indices, axis_to_remove);
  Builder::SetTracingInfo(op->name(), reshaped_indices);
  SaveNgOp(ng_op_map, op->name(), reshaped_indices);
  return Status::OK();
}

static Status TranslateArgMaxOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return (TranslateArgMinMax(op, static_input_map, ng_op_map, "max"));
}

static Status TranslateArgMinOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return (TranslateArgMinMax(op, static_input_map, ng_op_map, "min"));
}

static Status TranslateAvgPoolOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

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
  NHWCtoHW(is_nhwc, tf_strides, ng_strides);
  NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
  NHWCtoHW(is_nhwc, tf_ksize, ng_kernel_shape);
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff padding_below;
  ng::CoordinateDiff padding_above;
  ng::Shape ng_dilations{1, 1};
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, padding_below, padding_above);

  // TODO: remove this once nGraph supports negative padding
  // (CoordinateDiff) for AvgPool
  ng::Shape ng_padding_below(padding_below.begin(), padding_below.end());
  ng::Shape ng_padding_above(padding_above.begin(), padding_above.end());

  ng::Output<ng::Node> ng_avgpool = ConstructNgNode<opset::AvgPool>(
      op->name(), ng_input, ng_strides, ng_padding_below, ng_padding_above,
      ng_kernel_shape, true, ng::op::RoundingType::FLOOR);

  NCHWtoNHWC(op->name(), is_nhwc, ng_avgpool);
  NGRAPH_VLOG(3) << "avgpool outshape: {" << ng::join(ng_avgpool.get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_avgpool);
  return Status::OK();
}

static Status TranslateBiasAddOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_bias;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_bias));

  std::string tf_data_format;
  if (GetNodeAttr(op->attrs(), "data_format", &tf_data_format) !=
      Status::OK()) {
    tf_data_format = "NHWC";
  }

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "BiasAdd data format is neither NHWC nor NCHW");
  }

  auto ng_input_shape = ng_input.get_shape();
  auto ng_bias_shape = ng_bias.get_shape();
  if (ng_bias_shape.size() != 1) {
    return errors::InvalidArgument(
        "Bias argument to BiasAdd does not have one dimension");
  }

  // We'll choose reshape over broadcast
  // Reshape the bias to (1, C, 1, ...) if input is channels-first.
  ng::Output<ng::Node> ng_bias_reshaped = ng_bias;
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
    auto target_shape_node = make_shared<opset::Constant>(
        ng::element::i64, ng::Shape{ng_input_shape.size()}, target_shape);
    ng_bias_reshaped = ConstructNgNode<opset::Reshape>(
        op->name(), ng_bias, target_shape_node, false);
  }

  ng::Output<ng::Node> ng_add =
      ConstructNgNode<opset::Add>(op->name(), ng_input, ng_bias_reshaped);

  SaveNgOp(ng_op_map, op->name(), ng_add);
  return Status::OK();
}

static Status TranslateCastOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "DstT", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  try {
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<opset::Convert>(op->name(), ng_input, ng_et));
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
    ng::Output<ng::Node> ng_first_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_first_arg));

    concat_axis += int64(ng_first_arg.get_shape().size());
  }

  ng::OutputVector ng_args;

  for (int i = 0; i < op->num_inputs() - 1; i++) {
    ng::Output<ng::Node> ng_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, ng_arg));
    ng_args.push_back(ng_arg);
  }

  SaveNgOp(
      ng_op_map, op->name(),
      ConstructNgNode<opset::Concat>(op->name(), ng_args, size_t(concat_axis)));
  return Status::OK();
}

static Status TranslateConstOp(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dtype", &dtype));

  ng::Output<ng::Node> ng_node;

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
    TF_RETURN_IF_ERROR(func_param.first(op, func_param.second, ng_node));
  } catch (const std::out_of_range&) {
    return errors::Unimplemented("Failed to translate Constant with TF type:",
                                 DataType_Name(dtype));
  }

  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

static Status TranslateConv2DOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter));

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

  NHWCtoHW(is_nhwc, tf_strides, ng_strides);
  NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
  NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below;
  ng::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  ng::Output<ng::Node> ng_conv = ConstructNgNode<opset::Convolution>(
      op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
      ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_nhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateConv2DBackpropInputOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_filter, ng_out_backprop, ng_unused;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_unused, ng_filter, ng_out_backprop));

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

  NHWCtoHW(is_nhwc, tf_strides, ng_strides);
  NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
  NHWCtoHW(is_nhwc, tf_input_sizes, ng_image_shape);
  NHWCtoNCHW(op->name(), is_nhwc, ng_out_backprop);
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

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below;
  ng::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  auto ng_output_shape = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{ng_batch_shape.size() - 2},
      vector<size_t>(ng_batch_shape.begin() + 2, ng_batch_shape.end()));

  auto ng_data = ConstructNgNode<opset::ConvolutionBackpropData>(
      op->name(), ng_out_backprop, ng_filter, ng_output_shape, ng_strides,
      ng_padding_below, ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_nhwc, ng_data);
  SaveNgOp(ng_op_map, op->name(), ng_data);
  return Status::OK();
}

// Translate Conv3D Op
static Status TranslateConv3DOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter));

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

  NHWCtoHW(is_ndhwc, tf_strides, ng_strides);
  NHWCtoHW(is_ndhwc, ng_input.get_shape(), ng_image_shape);
  NHWCtoHW(is_ndhwc, tf_dilations, ng_dilations);
  NHWCtoNCHW(op->name(), is_ndhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  ng_kernel_shape[2] = ng_filter_shape[2];
  Transpose3D<4, 3, 0, 1, 2>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below;
  ng::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  ng::Output<ng::Node> ng_conv = ConstructNgNode<opset::Convolution>(
      op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
      ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_ndhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateCumsumOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_x, ng_axis;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_x, ng_axis));
  bool exclusive, reverse;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "exclusive", &exclusive));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "reverse", &reverse));

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::CumSum>(op->name(), ng_x, ng_axis, exclusive,
                                          reverse));
  return Status::OK();
}

// Translate DepthToSpace op
static Status TranslateDepthToSpaceOp(const Node* op,
                                      const std::vector<const Tensor*>&,
                                      Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  // Get the attributes
  int64 block_size;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "block_size", &block_size));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "DepthToSpace data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  auto ng_mode = opset::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
  ng::Output<ng::Node> depth_to_space = ConstructNgNode<opset::DepthToSpace>(
      op->name(), ng_input, ng_mode, block_size);
  NCHWtoNHWC(op->name(), is_nhwc, depth_to_space);
  SaveNgOp(ng_op_map, op->name(), depth_to_space);
  return Status::OK();
}

static Status TranslateDepthwiseConv2dNativeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter));

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

  NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
  NHWCtoHW(is_nhwc, tf_strides, ng_strides);
  NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below;
  ng::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  // ng input shape is NCHW
  auto& input_shape = ng_input.get_shape();
  // ng filter shape is OIHW
  auto& filter_shape = ng_filter.get_shape();
  ng::OutputVector ng_args;

  for (size_t i = 0; i < input_shape[1]; i++) {
    const std::vector<size_t> lower_bound_vec{0, i, 0, 0};
    const std::vector<size_t> upper_bound_vec{input_shape[0], i + 1,
                                              input_shape[2], input_shape[3]};
    auto lower_bound = ConstructNgNode<opset::Constant>(
        op->name(), ng::element::i64, ng::Shape{lower_bound_vec.size()},
        lower_bound_vec);
    auto upper_bound = ConstructNgNode<opset::Constant>(
        op->name(), ng::element::i64, ng::Shape{upper_bound_vec.size()},
        upper_bound_vec);
    auto ng_sliced_input = ConstructNgNode<opset::StridedSlice>(
        op->name(), ng_input, lower_bound, upper_bound, std::vector<int64_t>{},
        std::vector<int64_t>{});

    const std::vector<size_t> f_lower_bound_vec{0, i, 0, 0};
    const std::vector<size_t> f_upper_bound_vec{
        filter_shape[0], i + 1, filter_shape[2], filter_shape[3]};
    auto f_lower_bound = ConstructNgNode<opset::Constant>(
        op->name(), ng::element::i64, ng::Shape{f_lower_bound_vec.size()},
        f_lower_bound_vec);
    auto f_upper_bound = ConstructNgNode<opset::Constant>(
        op->name(), ng::element::i64, ng::Shape{f_upper_bound_vec.size()},
        f_upper_bound_vec);
    auto ng_sliced_filter = ConstructNgNode<opset::StridedSlice>(
        op->name(), ng_filter, f_lower_bound, f_upper_bound,
        std::vector<int64_t>{}, std::vector<int64_t>{});

    NGRAPH_VLOG(3) << "depthwise conv 2d.";
    NGRAPH_VLOG(3) << "sliced shape " << ng::join(ng_sliced_input.get_shape());
    NGRAPH_VLOG(3) << "filter shape " << ng::join(ng_sliced_filter.get_shape());
    auto ng_conv = ConstructNgNode<opset::Convolution>(
        op->name(), ng_sliced_input, ng_sliced_filter, ng_strides,
        ng_padding_below, ng_padding_above, ng_dilations);

    ng_args.push_back(ng_conv);
  }

  size_t ng_concatenation_axis = 1;  // channel axis
  auto ng_concat = ConstructNgNode<opset::Concat>(op->name(), ng_args,
                                                  ng_concatenation_axis);

  NCHWtoNHWC(op->name(), is_nhwc, ng_concat);
  SaveNgOp(ng_op_map, op->name(), ng_concat);
  return Status::OK();
}

static Status TranslateExpandDimsOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_dim;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_dim));

  std::vector<int64> dim_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &dim_vec));

  if (dim_vec.size() != 1) {
    return errors::InvalidArgument(
        "The size of argument dim is not 1 for ExpandDims");
  }

  auto& shape = ng_input.get_shape();
  if (dim_vec[0] < 0) {
    // allow range [-rank(input) - 1, rank(input)]
    // where -1 append new axis at the end
    dim_vec[0] = shape.size() + dim_vec[0] + 1;
  }
  auto out_shape = shape;
  out_shape.insert(out_shape.begin() + size_t(dim_vec[0]), 1);

  auto ng_shape = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::u64, ng::Shape{out_shape.size()}, out_shape);

  ng::Output<ng::Node> ng_expand_dim =
      ConstructNgNode<opset::Reshape>(op->name(), ng_input, ng_shape, false);

  SaveNgOp(ng_op_map, op->name(), ng_expand_dim);
  return Status::OK();
}

static Status TranslateFillOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_value, ng_unused;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_unused, ng_value));

  std::vector<int64> dims_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 0, static_input_map, &dims_vec));

  auto ng_output_shape = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{dims_vec.size()}, dims_vec);

  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Broadcast>(
                                      op->name(), ng_value, ng_output_shape));
  return Status::OK();
}

static Status TranslateFloorDivOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "T", &dtype));
  auto int_types = NGraphIntDTypes();
  std::function<ng::Output<ng::Node>(ng::Output<ng::Node>,
                                     ng::Output<ng::Node>)>
      ng_bin_fn;
  if (std::find(int_types.begin(), int_types.end(), dtype) != int_types.end()) {
    ng_bin_fn = [&op](ng::Output<ng::Node> ng_input1,
                      ng::Output<ng::Node> ng_input2) {
      return ConstructNgNode<opset::Divide>(op->name(), ng_input1, ng_input2);
    };
  } else {
    ng_bin_fn = [&op](ng::Output<ng::Node> ng_input1,
                      ng::Output<ng::Node> ng_input2) {
      return ConstructNgNode<opset::Floor>(
          op->name(),
          ConstructNgNode<opset::Divide>(op->name(), ng_input1, ng_input2));
    };
  }
  return TranslateBinaryOp(op, static_input_map, ng_op_map, ng_bin_fn);
}

static Status TranslateFusedBatchNormOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_scale, ng_offset, ng_mean, ng_variance;
  bool is_v3 = op->type_string() == "FusedBatchNormV3";

  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_scale, ng_offset,
                                   ng_mean, ng_variance));

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

  NHWCtoNCHW(op->name(), is_nhwc, ng_input);

  auto ng_batch_norm = ConstructNgNode<opset::BatchNormInference>(
      op->name(), ng_input, ng_scale, ng_offset, ng_mean, ng_variance,
      tf_epsilon);
  NCHWtoNHWC(op->name(), is_nhwc, ng_batch_norm);
  SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
  SaveNgOp(ng_op_map, op->name(), ng_mean);
  SaveNgOp(ng_op_map, op->name(), ng_variance);
  SaveNgOp(ng_op_map, op->name(), ng_mean);      // reserve_space_1
  SaveNgOp(ng_op_map, op->name(), ng_variance);  // reserve_space_2
  if (is_v3) {
    // FusedBatchNormV3 has 6 outputs
    SaveNgOp(ng_op_map, op->name(), ng_mean);  // reserve_space_3
  }
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

  ng::Output<ng::Node> ng_lhs, ng_rhs, ng_bias, ng_matmul;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs, ng_bias));
  ng_matmul = ConstructNgNode<opset::MatMul>(op->name(), ng_lhs, ng_rhs,
                                             transpose_a, transpose_b);

  auto ng_matmul_shape = ng_matmul.get_shape();
  auto ng_bias_shape = ng_bias.get_shape();

  if (ng_bias_shape.size() != 1) {
    return errors::InvalidArgument(
        "Bias argument to BiasAdd does not have one dimension");
  }

  auto ng_add = ConstructNgNode<opset::Add>(op->name(), ng_matmul, ng_bias);
  if (fused_ops.size() == 1) {  // Only fusing BiasAdd
    SaveNgOp(ng_op_map, op->name(), ng_add);
  } else if (fused_ops.size() == 2) {  // Also has activation
    if (fused_ops[1] == "Relu") {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<opset::Relu>(op->name(), ng_add));
    } else if (fused_ops[1] == "Relu6") {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<opset::Clamp>(op->name(), ng_add, 0, 6));
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
  ng::Output<ng::Node> ng_input, ng_input_coords, ng_unused;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_input, ng_input_coords, ng_unused));

  std::vector<int64> tf_axis;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &tf_axis));

  if (tf_axis.size() > 1) {
    return errors::Internal("Found axis in GatherV2 op (", op->name(),
                            ") translation to be non scalar, of size ",
                            tf_axis.size());
  }

  // Negative axis is supported. Accounting for that
  auto ng_input_shape = ng_input.get_shape();
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

  auto ng_axis = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{tf_axis.size()}, tf_axis);

  auto gather_op = ConstructNgNode<opset::Gather>(op->name(), ng_input,
                                                  ng_input_coords, ng_axis);

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

  auto CreateNgConv = [&](ng::Output<ng::Node>& ng_input,
                          ng::Output<ng::Node>& ng_filter,
                          ng::Output<ng::Node>& ng_conv) {
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

    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(op->name(), is_nhwc, ng_input);

    NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    Transpose<3, 2, 0, 1>(ng_filter);
    Builder::SetTracingInfo(op->name(), ng_filter);

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

    ng::CoordinateDiff ng_padding_below;
    ng::CoordinateDiff ng_padding_above;
    Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_dilations, ng_padding_below,
                         ng_padding_above);

    ng_conv = ConstructNgNode<opset::Convolution>(
        op->name() + "_FusedConv2D_Conv", ng_input, ng_filter, ng_strides,
        ng_padding_below, ng_padding_above, ng_dilations);

    return Status::OK();
  };

  if (VecStrCmp(fused_ops, {"BiasAdd"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
    if (num_args != 1) {
      return errors::InvalidArgument(
          "FusedConv2DBiasAdd has incompatible num_args");
    }

    ng::Output<ng::Node> ng_input, ng_filter, ng_bias, ng_conv;
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, ng_input, ng_filter, ng_bias));

    TF_RETURN_IF_ERROR(CreateNgConv(ng_input, ng_filter, ng_conv));

    auto ng_conv_shape = ng_conv.get_shape();
    auto ng_bias_shape = ng_bias.get_shape();
    if (ng_bias_shape.size() != 1) {
      return errors::InvalidArgument(
          "Bias argument to BiasAdd does not have one dimension");
    }

    std::vector<size_t> reshape_pattern_values(ng_conv_shape.size(), 1U);
    reshape_pattern_values[1] = ng_bias.get_shape().front();
    auto reshape_pattern = make_shared<opset::Constant>(
        ng::element::u64, ng::Shape{reshape_pattern_values.size()},
        reshape_pattern_values);
    auto ng_bias_reshaped = ConstructNgNode<opset::Reshape>(
        op->name(), ng_bias, reshape_pattern, false);

    auto ng_add = ConstructNgNode<opset::Add>(
        op->name() + "_FusedConv2D_BiasAdd", ng_conv, ng_bias_reshaped);

    if (VecStrCmp(fused_ops, {"BiasAdd", "Relu"})) {
      auto ng_relu = ConstructNgNode<opset::Relu>(
          op->name() + "_FusedConv2D_Relu", ng_add);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu);
      SaveNgOp(ng_op_map, op->name(), ng_relu);
    } else if (VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
      auto ng_relu6 = ConstructNgNode<opset::Clamp>(
          op->name() + "_FusedConv2D_Relu6", ng_add, 0, 6);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu6);
      SaveNgOp(ng_op_map, op->name(), ng_relu6);
    } else {
      NCHWtoNHWC(op->name(), is_nhwc, ng_add);
      SaveNgOp(ng_op_map, op->name(), ng_add);
    }
  } else if (VecStrCmp(fused_ops, {"FusedBatchNorm"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
    if (num_args != 4) {
      return errors::InvalidArgument(
          "FusedConv2D with FusedBatchNorm has incompatible num_args");
    }

    ng::Output<ng::Node> ng_input, ng_filter, ng_conv, ng_scale, ng_offset,
        ng_mean, ng_variance;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter,
                                     ng_scale, ng_offset, ng_mean,
                                     ng_variance));
    TF_RETURN_IF_ERROR(CreateNgConv(ng_input, ng_filter, ng_conv));

    float tf_epsilon;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon));

    auto ng_batch_norm = ConstructNgNode<opset::BatchNormInference>(
        op->name() + "_FusedConv2D_BatchNorm", ng_conv, ng_scale, ng_offset,
        ng_mean, ng_variance, tf_epsilon);

    if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"})) {
      auto ng_relu = ConstructNgNode<opset::Relu>(
          op->name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu);
      SaveNgOp(ng_op_map, op->name(), ng_relu);
    } else if (VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
      auto ng_relu6 = ConstructNgNode<opset::Clamp>(
          op->name() + "_FusedConv2D_BatchNormRelu", ng_batch_norm, 0, 6);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu6);
      SaveNgOp(ng_op_map, op->name(), ng_relu6);
    } else {
      NCHWtoNHWC(op->name(), is_nhwc, ng_batch_norm);
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
  ng::Output<ng::Node> ng_arg;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_arg));
  SaveNgOp(ng_op_map, op->name(), ng_arg);
  return Status::OK();
}

static Status TranslateIsFiniteOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  // Implemented tf.is_finite by checking:
  // (in != inf) && (in != -inf) && (in == in)
  //                                 ^^^^^^^^ checks for NaN's
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  auto const_inf = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), ng::Shape{},
      std::vector<float>{std::numeric_limits<float>::infinity()});

  auto const_neg_inf = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), ng::Shape{},
      std::vector<float>{-std::numeric_limits<float>::infinity()});

  auto neq_inf =
      ConstructNgNode<opset::NotEqual>(op->name(), ng_input, const_inf);
  auto neq_neg_inf =
      ConstructNgNode<opset::NotEqual>(op->name(), ng_input, const_neg_inf);
  auto eq_nan = ConstructNgNode<opset::Equal>(op->name(), ng_input, ng_input);

  auto neq_inf_and_neq_neg_inf =
      ConstructNgNode<opset::LogicalAnd>(op->name(), neq_inf, neq_neg_inf);
  auto is_finite = ConstructNgNode<opset::LogicalAnd>(
      op->name(), neq_inf_and_neq_neg_inf, eq_nan);

  SaveNgOp(ng_op_map, op->name(), is_finite);
  return Status::OK();
}

static Status TranslateL2LossOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  std::vector<float> val;
  val.push_back(2.0);
  auto const_2 = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), ng::Shape{}, val[0]);

  auto ng_pow =
      ConstructNgNode<opset::Multiply>(op->name(), ng_input, ng_input);

  size_t input_rank = ng_input.get_shape().size();
  std::vector<int64> axes;
  for (size_t i = 0; i < input_rank; ++i) {
    axes.push_back(i);
  }

  auto ng_reduction_axes = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{axes.size()}, axes);
  auto ng_sum =
      ConstructNgNode<opset::ReduceSum>(op->name(), ng_pow, ng_reduction_axes);
  auto ng_l2loss = ConstructNgNode<opset::Divide>(op->name(), ng_sum, const_2);
  SaveNgOp(ng_op_map, op->name(), ng_l2loss);
  return Status::OK();
}

static Status TranslateLog1pOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> val_1(ng::shape_size(shape), "1");
        auto ng_const1 =
            ConstructNgNode<opset::Constant>(op->name(), et, shape, val_1);
        auto ng_add = ConstructNgNode<opset::Add>(op->name(), ng_const1, n);
        return ConstructNgNode<opset::Log>(op->name(), ng_add);
      });
}

static Status TranslateLRNOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp));

  float alpha;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "alpha", &alpha));
  float beta;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "beta", &beta));
  float bias;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "bias", &bias));
  int64 depth_radius;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "depth_radius", &depth_radius));

  // OV: Each input value is divided by (bias+(alpha/size)*sum(xi^2 for every xi
  // in the local region))^beta
  // TF: sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d +
  // depth_radius + 1] ** 2)
  //     output = input / (bias + alpha * sqr_sum) ** beta
  int64 size = depth_radius * 2 + 1;
  alpha = alpha * size;
  // nGraph expects the input to be in NCHW format
  NHWCtoNCHW(op->name(), true, ng_inp);
  auto ng_output = ConstructNgNode<opset::LRN>(op->name(), ng_inp, alpha, beta,
                                               bias, (size_t)size);
  NCHWtoNHWC(op->name(), true, ng_output);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateLogSoftmaxOp(const Node* op,
                                    const std::vector<const Tensor*>&,
                                    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp));
  auto inp_shape = ng_inp.get_shape();
  size_t rank = inp_shape.size();
  // Batch i, class j
  // logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
  // Actually implementing: logsoftmax[i, j] = logits[i, j] - max(logits[i]) -
  // log(sum(exp(logits[i] - max(logits[i]))))
  std::vector<int64> axes;
  axes.push_back(rank - 1);
  auto ng_axis = ConstructNgNode<opset::Constant>(op->name(), ng::element::i64,
                                                  ng::Shape{axes.size()}, axes);
  auto ng_max =
      ConstructNgNode<opset::ReduceMax>(op->name(), ng_inp, ng_axis, true);
  auto ng_inp_minus_max =
      ConstructNgNode<opset::Subtract>(op->name(), ng_inp, ng_max);
  auto ng_exp = ConstructNgNode<opset::Exp>(op->name(), ng_inp_minus_max);
  auto ng_log_sum = ConstructNgNode<opset::Log>(
      op->name(),
      ConstructNgNode<opset::ReduceSum>(op->name(), ng_exp, ng_axis, true));
  auto ng_output = ConstructNgNode<opset::Subtract>(
      op->name(), ng_inp_minus_max, ng_log_sum);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateMatMulOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs));

  // Transpose arguments if requested.
  bool transpose_a = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_a", &transpose_a));

  bool transpose_b = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "transpose_b", &transpose_b));

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::MatMul>(op->name(), ng_lhs, ng_rhs,
                                          transpose_a, transpose_b));
  return Status::OK();
}

template <unsigned int N>
static Status TranslateMaxPoolOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  bool is_nhwc = (tf_data_format == "NHWC") || (tf_data_format == "NDHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(N);
  ng::Shape ng_image_shape(N);
  ng::Shape ng_kernel_shape(N);
  ng::Shape ng_dilations(N, 1);

  NHWCtoHW(is_nhwc, tf_strides, ng_strides);
  NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
  NHWCtoHW(is_nhwc, tf_ksize, ng_kernel_shape);
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff padding_below;
  ng::CoordinateDiff padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, padding_below, padding_above);

  // TODO: remove this once nGraph supports negative padding
  // (CoordinateDiff) for MaxPool
  ng::Shape ng_padding_below(padding_below.begin(), padding_below.end());
  ng::Shape ng_padding_above(padding_above.begin(), padding_above.end());

  auto ng_maxpool = ConstructNgNode<opset::MaxPool>(
      op->name(), ng_input, ng_strides, ng_padding_below, ng_padding_above,
      ng_kernel_shape, ng::op::RoundingType::FLOOR);

  NCHWtoNHWC(op->name(), is_nhwc, ng_maxpool);

  NGRAPH_VLOG(3) << "maxpool outshape: {" << ng::join(ng_maxpool.get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_maxpool);
  return Status::OK();
}

static Status TranslateNonMaxSuppressionV4Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_boxes, ng_scores;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_boxes, ng_scores));

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

  auto ng_max_output_size = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{}, max_output_size[0]);
  auto ng_iou_threshold = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::f32, ng::Shape{}, iou_threshold[0]);
  auto ng_score_threshold = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::f32, ng::Shape{}, score_threshold[0]);

  auto ng_nmsv4 = ConstructNgNode<opset::NonMaxSuppression>(
      op->name(), ng_boxes, ng_scores, ng_max_output_size, ng_iou_threshold,
      ng_score_threshold);

  Builder::SetTracingInfo(op->name(), ng_nmsv4);
  auto ng_selected_indices = ng_nmsv4.get_node_shared_ptr()->output(0);
  auto ng_valid_output = ng_nmsv4.get_node_shared_ptr()->output(1);
  SaveNgOp(ng_op_map, op->name(), ng_selected_indices);
  SaveNgOp(ng_op_map, op->name(), ng_valid_output);
  return Status::OK();
}

static Status TranslateReduceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map,
    std::function<ng::Output<ng::Node>(ng::Output<ng::Node>,
                                       ng::Output<ng::Node>, const bool)>
        create_ng_node) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &axes));

  ng::Shape input_shape = ng_input.get_shape();
  size_t input_rank = input_shape.size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(axes.size());
  std::transform(
      axes.begin(), axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  auto ng_reduction_axes = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{ng_reduction_axes_vect.size()},
      ng_reduction_axes_vect);

  ng::Output<ng::Node> ng_node =
      create_ng_node(ng_input, ng_reduction_axes, tf_keep_dims);

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
      [&op](ng::Output<ng::Node> ng_input,
            ng::Output<ng::Node> ng_reduction_axes, const bool keep_dims) {
        return ConstructNgNode<T>(op->name(), ng_input, ng_reduction_axes,
                                  keep_dims);
      });
}

static Status TranslateOneHotOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_features, ng_unused, ng_on, ng_off, ng_depth;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_features, ng_unused, ng_on, ng_off));

  auto ng_features_shape = ng_features.get_shape();
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

  auto ng_onehot = ConstructNgNode<opset::OneHot>(
      op->name(), ng_features, const_depth, ng_on, ng_off, one_hot_axis);
  SaveNgOp(ng_op_map, op->name(), ng_onehot);
  return Status::OK();
}

static Status TranslatePackOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 1));

  ng::OutputVector ng_concat_inputs;
  for (tensorflow::int32 i = 0; i < op->num_inputs(); ++i) {
    ng::Output<ng::Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, ng_input));
    ng_concat_inputs.push_back(ng_input);
  }

  int32 tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  size_t input_rank = ng_concat_inputs[0].get_shape().size();

  auto concat_axis = tf_axis;
  if (concat_axis == -1) {
    concat_axis = input_rank;
  }

  ng::Shape input_shape = ng_concat_inputs[0].get_shape();
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
    auto ng_shape = ConstructNgNode<opset::Constant>(
        op->name(), ng::element::u64, ng::Shape{extended_shape.size()},
        extended_shape);

    for (size_t i = 0; i < ng_concat_inputs.size(); ++i) {
      ng_concat_inputs[i] = ConstructNgNode<opset::Reshape>(
          op->name(), ng_concat_inputs[i], ng_shape, false);
    }
    ng_axis_order.push_back(input_rank);
  }

  auto concat = ConstructNgNode<opset::Concat>(op->name(), ng_concat_inputs,
                                               (size_t)concat_axis);

  auto ng_output_shape = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::u64, ng::Shape{output_shape.size()},
      output_shape);
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Reshape>(op->name(), concat, ng_output_shape,
                                           false));
  return Status::OK();
}

// 3 different Pad Ops: Pad, PadV2, MirrorPad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad-v2
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mirror-pad
static Status TranslatePadOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_paddings_op, pad_val_op, result_pad_op;

  // Set inputs and pad_val_op
  if (op->type_string() == "Pad" || op->type_string() == "MirrorPad") {
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_paddings_op));
    pad_val_op = ConstructNgNode<opset::Constant>(
        op->name(), ng_input.get_element_type(), ng::Shape(),
        std::vector<int>({0}));
  } else if (op->type_string() == "PadV2") {
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, ng_input, ng_paddings_op, pad_val_op));
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
  std::vector<int64> paddings;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &paddings));
  NGRAPH_VLOG(3) << op->name() << " pads {" << ng::join(paddings) << "}";
  if (paddings.size() % 2 != 0) {
    return errors::InvalidArgument(
        "Constant node for paddings does not have an even number of "
        "elements");
  }
  std::vector<int64> pad_begin(paddings.size() / 2);
  std::vector<int64> pad_end(paddings.size() / 2);
  for (size_t i = 0; i < paddings.size() / 2; i++) {
    pad_begin[i] = paddings[2 * i];
    pad_end[i] = paddings[2 * i + 1];
  }
  auto pads_begin_node = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{pad_begin.size()}, pad_begin);
  auto pads_end_node = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{pad_end.size()}, pad_end);

  // Create final Op
  result_pad_op =
      ConstructNgNode<opset::Pad>(op->name(), ng_input, pads_begin_node,
                                  pads_end_node, pad_val_op, pad_mode);

  SaveNgOp(ng_op_map, op->name(), result_pad_op);
  return Status::OK();
}

static Status TranslateRankOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  ng::Shape input_shape = ng_input.get_shape();
  auto input_rank = static_cast<int>(input_shape.size());

  auto ng_rank = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i32, ng::Shape(),
      std::vector<int>({input_rank}));

  SaveNgOp(ng_op_map, op->name(), ng_rank);
  return Status::OK();
}

static Status TranslateReciprocalOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        // Create a constant tensor populated with the value -1.
        // (1/x = x^(-1))
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-1");
        auto ng_exponent = ConstructNgNode<opset::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -1.
        return ConstructNgNode<opset::Power>(op->name(), n, ng_exponent);
      });
}

static Status TranslateRelu6Op(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Clamp>(op->name(), ng_input, 0, 6));
  return Status::OK();
}

static Status TranslateReshapeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_shape_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_shape_op));

  NGRAPH_VLOG(3) << "Input shape: " << ng::join(ng_input.get_shape());

  std::vector<int64> shape;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &shape));

  NGRAPH_VLOG(3) << "Requested result shape: " << ng::join(shape);

  auto ng_shape = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{shape.size()}, shape);
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Reshape>(
                                      op->name(), ng_input, ng_shape, false));
  return Status::OK();
}

static Status TranslateRsqrtOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        // Create a constant tensor populated with the value -1/2.
        // (1/sqrt(x) = x^(-1/2))
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-0.5");
        auto ng_exponent = ConstructNgNode<opset::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -0.5.
        return ConstructNgNode<opset::Power>(op->name(), n, ng_exponent);
      });
}

static Status TranslateShapeOp(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

  // default output_type = element::i64
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::ShapeOf>(op->name(), ng_input, type));
  return Status::OK();
}

static Status TranslateSizeOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  // Size has an attribute to specify output, int32 or int64
  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

  auto ng_input_shape = ng_input.get_shape();
  int64 result = 1;
  for (auto dim : ng_input_shape) {
    result *= dim;
  }

  // make a scalar with value equals to result
  auto ng_result = ConstructNgNode<opset::Constant>(
      op->name(), type, ng::Shape(0), std::vector<int64>({result}));

  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

static Status TranslateSliceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_begin, ng_size;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_begin, ng_size));

  std::vector<int64> begin_vec;
  std::vector<int64> size_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &begin_vec));
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &size_vec));

  if (begin_vec.size() != size_vec.size())
    return errors::InvalidArgument(
        "Cannot translate slice op: size of begin = ", begin_vec.size(),
        ", size of size_vec = ", size_vec.size(), ". Expected them to match.");

  NGRAPH_VLOG(3) << "Begin input for Slice: " << ng::join(begin_vec);
  NGRAPH_VLOG(3) << "Size input for Slice: " << ng::join(size_vec);

  std::vector<int64> end_vec(begin_vec.size());
  const auto ng_input_shape = ng_input.get_shape();
  stringstream err_stream;
  string err_msg;
  for (size_t i = 0; i < size_vec.size(); i++) {
    if (size_vec[i] != -1) {
      end_vec[i] = begin_vec[i] + size_vec[i];
    } else {
      // support -1 for size_vec, to the end of the tensor
      end_vec[i] = ng_input_shape[i];
    }

    // check for this condition: 0 <= begin[i] <= begin[i] + size[i] <= Di
    if (0 > begin_vec[i])
      err_stream << "lower < 0: " << begin_vec[i]
                 << ". It should have been positive.\n";
    if (begin_vec[i] > end_vec[i])
      err_stream << "upper < lower: upper = " << end_vec[i]
                 << ", lower = " << begin_vec[i] << "\n";
    if (begin_vec[i] > ng_input_shape[i])
      err_stream << "dim < upper: dim = " << ng_input_shape[i]
                 << ", upper = " << end_vec[i] << "\n";

    err_msg = err_stream.str();
    if (!err_msg.empty())
      return errors::InvalidArgument("Cannot translate slice op at position ",
                                     i, " of ", size_vec.size(),
                                     ". The reasons are:\n", err_msg);
  }

  auto begin = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{begin_vec.size()}, begin_vec);
  auto end = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{end_vec.size()}, end_vec);

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::StridedSlice>(op->name(), ng_input, begin,
                                                end, std::vector<int64_t>{},
                                                std::vector<int64_t>{}));
  return Status::OK();
}

static Status TranslateSoftmaxOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  auto input_shape = ng_input.get_shape();
  auto rank = input_shape.size();
  if (rank < 1) {
    return errors::InvalidArgument("TF Softmax logits must be >=1 dimension");
  }

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Softmax>(op->name(), ng_input, rank - 1));
  return Status::OK();
}

// Translate SpaceToDepthOp
static Status TranslateSpaceToDepthOp(const Node* op,
                                      const std::vector<const Tensor*>&,
                                      Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  // Get the attributes
  int64 block_size;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "block_size", &block_size));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "DepthToSpace data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  auto ng_mode = opset::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
  auto space_to_depth = ConstructNgNode<opset::SpaceToDepth>(
      op->name(), ng_input, ng_mode, block_size);
  NCHWtoNHWC(op->name(), is_nhwc, space_to_depth);
  SaveNgOp(ng_op_map, op->name(), space_to_depth);
  return Status::OK();
}

static Status TranslateSplitOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, ng_input));
  // num_split : The number of ways to split. Must evenly divide
  // value.shape[split_dim]
  int32 num_split;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_split", &num_split));

  ng::Shape shape = ng_input.get_shape();
  int rank = shape.size();

  std::vector<int> split_dim_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 0, static_input_map, &split_dim_vec));
  int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64)rank : 0);
  auto ng_split_dim = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::u64, ng::Shape{}, split_dim);
  auto ng_split = make_shared<opset::Split>(ng_input, ng_split_dim, num_split);

  for (int i = 0; i < num_split; ++i) {
    auto out = ng_split->output(i);
    Builder::SetTracingInfo(op->name(), out);
    SaveNgOp(ng_op_map, op->name(), out);
  }
  return Status::OK();
}

static Status TranslateSplitVOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_split_length, ng_split_dim;

  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  ng::Shape shape = ng_input.get_shape();
  int rank = shape.size();

  std::vector<int64> split_dim_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 2, static_input_map, &split_dim_vec));
  // there should be at least one element specified as axis and not more than
  // one as axis is 0-D
  if (split_dim_vec.size() != 1) {
    return errors::InvalidArgument(
        "split_dim_tensor must have "
        "exactly one element.");
  }
  TF_RETURN_IF_ERROR(CheckAxisDimInRange(split_dim_vec, rank));
  int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64)rank : 0);
  ng_split_dim = ConstructNgNode<opset::Constant>(op->name(), ng::element::i32,
                                                  ng::Shape{}, split_dim);

  std::vector<int> split_lengths_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 1, static_input_map, &split_lengths_vec));

  // length: Length of size_splits
  int length = 0;
  int idx = -1;

  // Find out the total length of the splits and locate -1 's index, if any
  bool has_one_neg = false;
  for (size_t i = 0; i < split_lengths_vec.size(); ++i) {
    if (split_lengths_vec[i] != -1) {
      length += split_lengths_vec[i];
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
    split_lengths_vec[idx] = shape[split_dim] - length;
  }

  if ((!has_one_neg && length != shape[split_dim]) ||
      (has_one_neg && split_lengths_vec[idx] < 0)) {
    return errors::InvalidArgument(
        "The length of size_splits must sum to the value of the dimension "
        "along split_dim");
  }

  ng_split_length = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i32, ng::Shape{split_lengths_vec.size()},
      split_lengths_vec);

  if (split_lengths_vec.size() != 1) {
    auto ng_split = make_shared<opset::VariadicSplit>(ng_input, ng_split_dim,
                                                      ng_split_length);
    for (size_t i = 0; i < split_lengths_vec.size(); ++i) {
      auto out = ng_split->output(i);
      Builder::SetTracingInfo(op->name(), out);
      SaveNgOp(ng_op_map, op->name(), out);
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
      op, static_input_map, ng_op_map, [&op](ng::Output<ng::Node> n) {
        return ConstructNgNode<opset::Multiply>(op->name(), n, n);
      });
}

static Status TranslateSqueezeOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
  size_t input_dims = ng_input.get_shape().size();

  std::vector<int32> tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "squeeze_dims", &tf_axis));

  // If input dimension is negative, make it positive
  for (size_t i = 0; i < tf_axis.size(); i++) {
    tf_axis[i] = tf_axis[i] < 0 ? (int32)(input_dims) + tf_axis[i] : tf_axis[i];
  }

  auto ng_const = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i32, ng::Shape{tf_axis.size()}, tf_axis);

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Squeeze>(op->name(), ng_input, ng_const));
  return Status::OK();
}

static Status TranslateStridedSliceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  int32 begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "begin_mask", &begin_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "end_mask", &end_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "new_axis_mask", &new_axis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "shrink_axis_mask", &shrink_axis_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ellipsis_mask", &ellipsis_mask));

  NGRAPH_VLOG(5) << "strided slice attributes: "
                 << "  begin mask: " << begin_mask << "  end mask: " << end_mask
                 << "  new axis mask: " << new_axis_mask
                 << "  shrink axis mask: " << shrink_axis_mask
                 << "  ellipsis mask: " << ellipsis_mask;

  std::vector<int64> begin_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &begin_vec));
  std::vector<int64> end_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &end_vec));
  std::vector<int64> stride_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 3, static_input_map, &stride_vec));

  auto begin = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{begin_vec.size()}, begin_vec);
  auto end = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{end_vec.size()}, end_vec);
  auto strides = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{stride_vec.size()}, stride_vec);

  auto mask_to_vec = [](int32 mask) {
    auto length = sizeof(mask) * CHAR_BIT;
    std::vector<int64_t> vec(length, 0);
    if (mask == 0) {
      return vec;
    }
    for (auto i = 0; i < length; ++i) {
      if ((unsigned char)(mask >> i & 0x01) == 1) {
        vec[i] = 1;
      }
    }
    return vec;
  };

  SaveNgOp(
      ng_op_map, op->name(),
      ConstructNgNode<opset::StridedSlice>(
          op->name(), ng_input, begin, end, strides, mask_to_vec(begin_mask),
          mask_to_vec(end_mask), mask_to_vec(new_axis_mask),
          mask_to_vec(shrink_axis_mask), mask_to_vec(ellipsis_mask)));
  return Status::OK();
}

static Status TranslateTileOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_multiples;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_multiples));

  std::vector<int64> multiples;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &multiples));

  auto ng_repeats = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::i64, ng::Shape{multiples.size()}, multiples);
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Tile>(op->name(), ng_input, ng_repeats));
  return Status::OK();
}

// Translate TopKV2 Op using ngraph core op TopK
static Status TranslateTopKV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ngraph::Node> ng_input;

  TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  // axis along which to compute top k indices
  int64 k_axis = ng_input.get_shape().size() - 1;

  // scalar input tensor specifying how many max/min elts should be computed
  // CPU backend only supports element type i64
  std::vector<int64> ng_k_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &ng_k_vec));
  auto ng_k = ConstructNgNode<opset::Constant>(op->name(), ng::element::i64,
                                               ng::Shape{}, ng_k_vec[0]);

  std::string mode = "max";

  std::string sort = "value";
  bool sorted = true;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "sorted", &sorted));
  if (!sorted) {
    sort = "index";
  }

  auto ng_result =
      std::make_shared<opset::TopK>(ng_input, ng_k, k_axis, mode, sort);

  ng::Output<ng::Node> ng_values = ng_result->output(0);
  Builder::SetTracingInfo(op->name(), ng_values);
  ng::Output<ng::Node> ng_indices = ng_result->output(1);
  Builder::SetTracingInfo(op->name(), ng_indices);

  SaveNgOp(ng_op_map, op->name(), ng_values);
  SaveNgOp(ng_op_map, op->name(), ng_indices);

  return Status::OK();
}

static Status TranslateTransposeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input, ng_permutation;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_permutation));

  std::vector<int64> permutation;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 1, static_input_map, &permutation));

  // Check to make sure that the permutation requested for transpose
  // is valid for example:
  // - it should not have duplicates,
  // - it should have all the dimensions.

  int ng_input_rank = ng_input.get_shape().size();
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

  NGRAPH_VLOG(3) << ng::join(permutation);

  auto input_order = ConstructNgNode<opset::Constant>(
      op->name(), ng::element::u64, ng::Shape{permutation.size()}, permutation);
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Transpose>(
                                      op->name(), ng_input, input_order));
  return Status::OK();
}

static Status TranslateUnpackOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  ng::Shape input_shape = ng_input.get_shape();
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

  std::vector<int64> output_shape;
  for (size_t i = 0; i < input_rank; ++i) {
    if ((int)i != unpack_axis) {
      output_shape.push_back(input_shape[i]);
    }
  }

  ng::AxisVector ng_axis_order;
  for (size_t i = 0; i < input_rank; i++) {
    ng_axis_order.push_back(i);
  }

  std::vector<size_t> lower_bound_vec(input_rank, 0);
  std::vector<size_t> upper_bound_vec(input_rank);

  for (size_t i = 0; i < input_rank; i++) {
    upper_bound_vec[i] = input_shape[i];
  }

  for (int i = 0; i < num_outputs; ++i) {
    lower_bound_vec[unpack_axis] = i;
    upper_bound_vec[unpack_axis] = i + 1;
    auto lower_bound = ConstructNgNode<opset::Constant>(
        op->name(), ng::element::i64, ng::Shape{lower_bound_vec.size()},
        lower_bound_vec);
    auto upper_bound = ConstructNgNode<opset::Constant>(
        op->name(), ng::element::i64, ng::Shape{upper_bound_vec.size()},
        upper_bound_vec);
    auto slice = ConstructNgNode<opset::StridedSlice>(
        op->name(), ng_input, lower_bound, upper_bound, std::vector<int64_t>{},
        std::vector<int64_t>{});
    auto ng_shape = ConstructNgNode<opset::Constant>(
        op->name(), ng::element::u64, ng::Shape{output_shape.size()},
        output_shape);
    auto reshaped =
        ConstructNgNode<opset::Reshape>(op->name(), slice, ng_shape, false);
    SaveNgOp(ng_op_map, op->name(), reshaped);
  }
  return Status::OK();
}

static Status TranslateXdivyOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ng::Output<ngraph::Node> ng_x, ng_y;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_x, ng_y));
  auto zero =
      ConstructNgNode<opset::Constant>(op->name(), ng_x.get_element_type(),
                                       ngraph::Shape{}, std::vector<int>({0}));
  auto x_is_zero = ConstructNgNode<opset::Equal>(op->name(), ng_x, zero);
  auto ng_xdivy = ConstructNgNode<opset::Divide>(op->name(), ng_x, ng_y);
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Select>(
                                      op->name(), x_is_zero, ng_x, ng_xdivy));
  return Status::OK();
}

static Status TranslateSelectOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input1, ng_input2, ng_input3;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_input1, ng_input2, ng_input3));
  auto ng_select = ConstructNgNode<opset::Select>(op->name(), ng_input1,
                                                  ng_input2, ng_input3);
  SaveNgOp(ng_op_map, op->name(), ng_select);
  return Status::OK();
}

static Status TranslateZerosLikeOp(const Node* op,
                                   const std::vector<const Tensor*>&,
                                   Builder::OpMap& ng_op_map) {
  ng::Output<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  ng::Shape input_shape = ng_input.get_shape();
  std::vector<std::string> const_values(ng::shape_size(input_shape), "0");
  auto ng_result = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), input_shape, const_values);
  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

const static std::map<
    const string,
    const function<Status(const Node*, const std::vector<const Tensor*>&,
                          Builder::OpMap&)>>
    TRANSLATE_OP_MAP{
        {"Abs", TranslateUnaryOp<opset::Abs>},
        {"Acos", TranslateUnaryOp<opset::Acos>},
        {"Acosh", TranslateUnaryOp<opset::Acosh>},
        {"Add", TranslateBinaryOp<opset::Add>},
        {"AddN", TranslateAddNOp},
        {"AddV2", TranslateBinaryOp<opset::Add>},
        {"Any", TranslateDirectReduceOp<opset::ReduceLogicalOr>},
        {"All", TranslateDirectReduceOp<opset::ReduceLogicalAnd>},
        {"ArgMax", TranslateArgMaxOp},
        {"ArgMin", TranslateArgMinOp},
        {"Asin", TranslateUnaryOp<opset::Asin>},
        {"Asinh", TranslateUnaryOp<opset::Asinh>},
        {"Atan", TranslateUnaryOp<opset::Atan>},
        {"Atanh", TranslateUnaryOp<opset::Atanh>},
        {"AvgPool", TranslateAvgPoolOp},
        {"BiasAdd", TranslateBiasAddOp},
        {"Cast", TranslateCastOp},
        {"Ceil", TranslateUnaryOp<opset::Ceiling>},
        {"ConcatV2", TranslateConcatV2Op},
        {"Const", TranslateConstOp},
        {"Conv2D", TranslateConv2DOp},
        {"Conv2DBackpropInput", TranslateConv2DBackpropInputOp},
        {"Conv3D", TranslateConv3DOp},
        {"Cos", TranslateUnaryOp<opset::Cos>},
        {"Cosh", TranslateUnaryOp<opset::Cosh>},
        {"Cumsum", TranslateCumsumOp},
        {"DepthToSpace", TranslateDepthToSpaceOp},
        {"DepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
        {"Equal", TranslateBinaryOp<opset::Equal>},
        {"Exp", TranslateUnaryOp<opset::Exp>},
        {"ExpandDims", TranslateExpandDimsOp},
        {"Fill", TranslateFillOp},
        {"Floor", TranslateUnaryOp<opset::Floor>},
        {"FloorDiv", TranslateFloorDivOp},
        {"FloorMod", TranslateBinaryOp<opset::FloorMod>},
        {"FusedBatchNorm", TranslateFusedBatchNormOp},
        {"FusedBatchNormV2", TranslateFusedBatchNormOp},
        {"FusedBatchNormV3", TranslateFusedBatchNormOp},
        {"GatherV2", TranslateGatherV2Op},
        {"_FusedConv2D", TranslateFusedConv2DOp},
        {"_FusedMatMul", TranslateFusedMatMulOp},
        {"Greater", TranslateBinaryOp<opset::Greater>},
        {"GreaterEqual", TranslateBinaryOp<opset::GreaterEqual>},
        {"Identity", TranslateIdentityOp},
        {"IsFinite", TranslateIsFiniteOp},
        {"L2Loss", TranslateL2LossOp},
        {"LogSoftmax", TranslateLogSoftmaxOp},
        {"Less", TranslateBinaryOp<opset::Less>},
        {"LessEqual", TranslateBinaryOp<opset::LessEqual>},
        {"Log", TranslateUnaryOp<opset::Log>},
        {"Log1p", TranslateLog1pOp},
        {"LogicalAnd", TranslateBinaryOp<opset::LogicalAnd>},
        {"LogicalNot", TranslateUnaryOp<opset::LogicalNot>},
        {"LogicalOr", TranslateBinaryOp<opset::LogicalOr>},
        {"LRN", TranslateLRNOp},
        {"MatMul", TranslateMatMulOp},
        {"Max", TranslateDirectReduceOp<opset::ReduceMax>},
        {"Maximum", TranslateBinaryOp<opset::Maximum>},
        {"MaxPool", TranslateMaxPoolOp<2>},
        {"MaxPool3D", TranslateMaxPoolOp<3>},
        {"NonMaxSuppressionV4", TranslateNonMaxSuppressionV4Op},
        {"Mean", TranslateDirectReduceOp<opset::ReduceMean>},
        {"Min", TranslateDirectReduceOp<opset::ReduceMin>},
        {"Minimum", TranslateBinaryOp<opset::Minimum>},
        {"MirrorPad", TranslatePadOp},
        {"Mul", TranslateBinaryOp<opset::Multiply>},
        {"Mod", TranslateBinaryOp<opset::Mod>},
        {"Neg", TranslateUnaryOp<opset::Negative>},
        {"NotEqual", TranslateBinaryOp<opset::NotEqual>},
        // Do nothing! NoOps sometimes get placed on nGraph for bureaucratic
        // reasons, but they have no data flow inputs or outputs.
        {"NoOp", [](const Node*, const std::vector<const Tensor*>&,
                    Builder::OpMap&) { return Status::OK(); }},
        {"OneHot", TranslateOneHotOp},
        {"Pack", TranslatePackOp},
        {"Pad", TranslatePadOp},
        {"PadV2", TranslatePadOp},
        {"Pow", TranslateBinaryOp<opset::Power>},
        // PreventGradient is just Identity in dataflow terms, so reuse that.
        {"PreventGradient", TranslateIdentityOp},
        {"Prod", TranslateDirectReduceOp<opset::ReduceProd>},
        {"Rank", TranslateRankOp},
        {"RealDiv", TranslateBinaryOp<opset::Divide>},
        {"Reciprocal", TranslateReciprocalOp},
        {"Relu", TranslateUnaryOp<opset::Relu>},
        {"Relu6", TranslateRelu6Op},
        {"Reshape", TranslateReshapeOp},
        {"Rsqrt", TranslateRsqrtOp},
        {"Select", TranslateSelectOp},
        {"SelectV2", TranslateSelectOp},
        {"Shape", TranslateShapeOp},
        {"Sigmoid", TranslateUnaryOp<opset::Sigmoid>},
        {"Sin", TranslateUnaryOp<opset::Sin>},
        {"Sinh", TranslateUnaryOp<opset::Sinh>},
        {"Size", TranslateSizeOp},
        {"Sign", TranslateUnaryOp<opset::Sign>},
        {"Slice", TranslateSliceOp},
        {"Snapshot", TranslateIdentityOp},
        {"Softmax", TranslateSoftmaxOp},
        {"Softplus", TranslateUnaryOp<opset::SoftPlus>},
        {"SpaceToDepth", TranslateSpaceToDepthOp},
        {"Split", TranslateSplitOp},
        {"SplitV", TranslateSplitVOp},
        {"Sqrt", TranslateUnaryOp<opset::Sqrt>},
        {"Square", TranslateSquareOp},
        {"SquaredDifference", TranslateBinaryOp<opset::SquaredDifference>},
        {"Squeeze", TranslateSqueezeOp},
        {"StridedSlice", TranslateStridedSliceOp},
        {"Sub", TranslateBinaryOp<opset::Subtract>},
        {"Sum", TranslateDirectReduceOp<opset::ReduceSum>},
        {"Tan", TranslateUnaryOp<opset::Tan>},
        {"Tanh", TranslateUnaryOp<opset::Tanh>},
        {"Tile", TranslateTileOp},
        {"TopKV2", TranslateTopKV2Op},
        {"Transpose", TranslateTransposeOp},
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
  // vector of generated nGraph Output<Node>.
  //
  Builder::OpMap ng_op_map;

  //
  // Populate the parameter list, and also put parameters into the op map.
  //
  ng::ParameterVector ng_parameter_list(tf_params.size());

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
        ConstructNgNode<opset::Parameter>(prov_tag, ng_et, ng_shape);
    SaveNgOp(ng_op_map, parm->name(), ng_param);
    ng_parameter_list[index] =
        ngraph::as_type_ptr<opset::Parameter>(ng_param.get_node_shared_ptr());
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
  ng::ResultVector ng_result_list(tf_ret_vals.size());

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

    ng::Output<ng::Node> result;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, n, 0, result));
    auto ng_result = ConstructNgNode<opset::Result>(n->name(), result);
    ng_result_list[index] =
        ngraph::as_type_ptr<opset::Result>(ng_result.get_node_shared_ptr());
  }

  //
  // Create the nGraph function.
  //
  ng_function = make_shared<ng::Function>(ng_result_list, ng_parameter_list);

  //
  // Apply additional passes on the nGraph function here.
  //
  {
    ngraph::pass::Manager passes;
    ngraph::pass::PassConfig pass_config;
    // set/honor the defaults, unless specified via env var
    auto set_default = [&pass_config](std::string pass, bool enable) {
      auto enables_map = pass_config.get_enables();
      if (enables_map.find(pass) == enables_map.end())
        pass_config.set_pass_enable(pass, enable);
    };
    set_default("ConstantFolding", false);
    set_default("TransposeSinking", true);
    if (pass_config.get_pass_enable("ConstantFolding"))
      passes.register_pass<ngraph::pass::ConstantFolding>();
    if (pass_config.get_pass_enable("TransposeSinking"))
      passes.register_pass<pass::TransposeSinking>();
    passes.run_passes(ng_function);
  }
  NGRAPH_VLOG(5) << "Done with passes";
  //
  // Request row-major layout on results.
  //
  for (auto result : ng_function->get_results()) {
    result->set_needs_default_layout(true);
  }
  NGRAPH_VLOG(5) << "Done with translations";
  return Status::OK();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
