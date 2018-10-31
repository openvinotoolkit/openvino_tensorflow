/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

#include "ngraph_builder.h"
#include "ngraph_conversions.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/numpy_transpose.hpp"
#include "ngraph/builder/quantization.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb_text.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

static Status ValidateInputCount(const Node* op, size_t count) {
  if (op->num_inputs() != count) {
    return errors::InvalidArgument("\"", op->name(), "\" requires ", count,
                                   " input(s), got ", op->num_inputs(),
                                   " instead");
  }
  return Status::OK();
}

static Status ValidateInputCountMin(const Node* op, size_t count) {
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
static Status GetInputNodes(const Builder::OpMap& ng_op_map, const Node* op,
                            size_t index) {
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
  if (node->type_string() == "_Arg") {
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

  *ng_node = make_shared<ng::op::Constant>(et, ng_shape, const_values);
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
//                           return (std::make_shared<ng::op::Multiply>(n,n));
//                       });
//  }
static Status TranslateUnaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>)>
        create_unary_op) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));
  SaveNgOp(ng_op_map, op->name(), create_unary_op(ng_input));

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
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map,
      [](std::shared_ptr<ng::Node> n) { return make_shared<T>(n); });
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
//           auto ng_diff = std::make_shared<ng::op::Subtract>(input1, input2);
//           return std::make_shared<ng::op::Multiply>(ng_diff,ng_diff);
//         }));
//    }
//

static Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>,
                                            std::shared_ptr<ng::Node>)>
        create_binary_op) {
  std::shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));

  std::tie(ng_lhs, ng_rhs) =
      ng::builder::numpy_broadcast(std::make_pair(ng_lhs, ng_rhs));

  SaveNgOp(ng_op_map, op->name(), create_binary_op(ng_lhs, ng_rhs));

  return Status::OK();
}

// Helper function to translate a binary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Add") {
//    TF_RETURN_IF_ERROR(TranslateBinaryOp<ng::op::Add>(op, static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
static Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateBinaryOp(
      op, static_input_map, ng_op_map,
      [](std::shared_ptr<ng::Node> ng_lhs, std::shared_ptr<ng::Node> ng_rhs) {
        return make_shared<T>(ng_lhs, ng_rhs);
      });
}

static Status TranslateAllreduceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  SaveNgOp(ng_op_map, op->name(), make_shared<ng::op::AllReduce>(ng_input));
  return Status::OK();
}

static Status TranslateAddNOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  std::vector<shared_ptr<ng::Node>> ng_arg_vec(op->num_inputs());

  for (int inp_idx = 0; inp_idx < op->num_inputs(); inp_idx++)
    TF_RETURN_IF_ERROR(
        GetInputNode(ng_op_map, op, inp_idx, &ng_arg_vec[inp_idx]));

  SaveNgOp(ng_op_map, op->name(),
           std::accumulate(std::next(ng_arg_vec.begin()), ng_arg_vec.end(),
                           ng_arg_vec.at(0)));  // accumulation: start with
                                                // first element. default op is
                                                // addition
  return Status::OK();
}

static Status TranslateAnyOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_axes_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_axes_op));
  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &axes));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = ng_input->get_shape().size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(axes.size());
  std::transform(
      axes.begin(), axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  ng::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

  std::vector<bool> init_val = {false};
  auto arg_init = make_shared<ng::op::Constant>(ng_input->get_element_type(),
                                                ng::Shape(0), init_val);
  auto f_A = make_shared<ng::op::Parameter>(ng::element::boolean, ng::Shape{});
  auto f_B = make_shared<ng::op::Parameter>(ng::element::boolean, ng::Shape{});
  auto ng_or = make_shared<ng::Function>(make_shared<ng::op::Or>(f_A, f_B),
                                         ng::op::ParameterVector{f_A, f_B});

  shared_ptr<ng::Node> ng_any =
      make_shared<ng::op::Reduce>(ng_input, arg_init, ng_or, ng_reduction_axes);

  // If keep_dims is specified we need to reshape to put back the reduced
  // axes, with length 1.
  if (tf_keep_dims) {
    ng::Shape ng_result_shape_with_keep(input_rank);
    for (size_t i = 0; i < input_rank; i++) {
      ng_result_shape_with_keep[i] =
          ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
    }
    ng::AxisVector ng_axis_order(ng_any->get_shape().size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
    ng_any = make_shared<ng::op::Reshape>(ng_any, ng_axis_order,
                                          ng_result_shape_with_keep);
  }

  SaveNgOp(ng_op_map, op->name(), ng_any);
  return Status::OK();
}

static Status TranslateAllOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_axes_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_axes_op));

  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> all_axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &all_axes));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = ng_input->get_shape().size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(all_axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(all_axes.size());
  std::transform(
      all_axes.begin(), all_axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  ng::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

  std::vector<bool> init_val = {true};
  auto arg_init = make_shared<ng::op::Constant>(ng_input->get_element_type(),
                                                ng::Shape(0), init_val);
  auto f_A = make_shared<ng::op::Parameter>(ng::element::boolean, ng::Shape{});
  auto f_B = make_shared<ng::op::Parameter>(ng::element::boolean, ng::Shape{});
  auto ng_and = make_shared<ng::Function>(make_shared<ng::op::And>(f_A, f_B),
                                          ng::op::ParameterVector{f_A, f_B});

  shared_ptr<ng::Node> ng_all = make_shared<ng::op::Reduce>(
      ng_input, arg_init, ng_and, ng_reduction_axes);

  // If keep_dims is specified we need to reshape to put back the reduced
  // axes, with length 1.
  if (tf_keep_dims) {
    ng::Shape ng_result_shape_with_keep(input_rank);

    for (size_t i = 0; i < input_rank; i++) {
      ng_result_shape_with_keep[i] =
          ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
    }

    ng::AxisVector ng_axis_order(ng_all->get_shape().size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
    ng_all = make_shared<ng::op::Reshape>(ng_all, ng_axis_order,
                                          ng_result_shape_with_keep);
  }

  SaveNgOp(ng_op_map, op->name(), ng_all);
  return Status::OK();
}

static Status TranslateArgMaxOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_dim;

  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_dim));

  std::vector<int64> tf_dim;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &tf_dim));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  if (tf_dim.size() != 1) {
    return errors::InvalidArgument(
        "ArgMax Op: dimension must be scalar, operates on a single axis");
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

  auto ng_argmax = make_shared<ng::op::ArgMax>(ng_input, input_dims, ng_et);

  SaveNgOp(ng_op_map, op->name(), ng_argmax);
  return Status::OK();
}

static Status TranslateArgMinOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_dim;

  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_dim));

  std::vector<int64> tf_dim;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &tf_dim));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  if (tf_dim.size() != 1) {
    return errors::InvalidArgument(
        "ArgMin Op:dimension must be scalar, operates on a single axis");
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

  auto ng_argmin = make_shared<ng::op::ArgMin>(ng_input, input_dims, ng_et);

  SaveNgOp(ng_op_map, op->name(), ng_argmin);
  return Status::OK();
}

static Status TranslateAvgPoolOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
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
  BatchToNGraph(is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for AvgPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_avgpool =
      make_shared<ng::op::AvgPool>(ng_input, ng_kernel_shape, ng_strides,
                                   ng_padding_below, ng_padding_above, false);

  BatchToTensorflow(is_nhwc, ng_avgpool);
  NGRAPH_VLOG(3) << "avgpool outshape: {" << ng::join(ng_avgpool->get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_avgpool);
  return Status::OK();
}

static Status TranslateAvgPoolGradOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_grad;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, nullptr, &ng_grad));

  std::vector<int32> tf_orig_input_shape_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 0, static_input_map, &tf_orig_input_shape_vec));

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
        "AvgPoolGrad data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Shape ng_orig_input_shape;
  for (int i = 0; i < tf_orig_input_shape_vec.size(); i++) {
    ng_orig_input_shape.push_back(tf_orig_input_shape_vec[i]);
  }

  ng::Shape ng_forward_arg_shape(4);
  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_window_shape(2);

  BatchedOpParamReshape(is_nhwc, ng_orig_input_shape, ng_forward_arg_shape);
  BatchToNGraph(is_nhwc, ng_grad);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_orig_input_shape, ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_window_shape);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_window_shape: " << ng::join(ng_window_shape);
  NGRAPH_VLOG(3) << "ng_forward_arg_shape: " << ng::join(ng_forward_arg_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for AvgPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_window_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_avgpool_backprop =
      make_shared<ng::op::AvgPoolBackprop>(
          ng_forward_arg_shape, ng_grad, ng_window_shape, ng_strides,
          ng_padding_below, ng_padding_above, false);

  BatchToTensorflow(is_nhwc, ng_avgpool_backprop);

  NGRAPH_VLOG(3) << "avgpoolbackprop outshape: {"
                 << ng::join(ng_avgpool_backprop->get_shape()) << "}";

  SaveNgOp(ng_op_map, op->name(), ng_avgpool_backprop);

  return Status::OK();
}

static Status TranslateBatchMatMulOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));

  auto ng_lhs_shape = ng_lhs->get_shape();
  auto ng_rhs_shape = ng_rhs->get_shape();

  if (ng_lhs_shape.size() != ng_rhs_shape.size()) {
    return errors::InvalidArgument(
        "Dimensions of two input args are not the same for BatchMatMul");
  }
  size_t n_dims = ng_lhs_shape.size();
  if (n_dims < 2) {
    return errors::InvalidArgument(
        "Dimensions of input args for BatchMatMul must be >=2", n_dims);
  }

  ng::AxisVector out_axes;
  for (size_t i = 0; i < n_dims - 2; ++i) {
    if (ng_lhs_shape[i] != ng_rhs_shape[i]) {
      return errors::InvalidArgument(
          "ng_lhs_shape and ng_rhs_shape must be the same for BatchMatMul "
          "for each dimension",
          i);
    }
    out_axes.push_back(i);
  }

  bool tf_adj_x = false;
  bool tf_adj_y = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "adj_x", &tf_adj_x));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "adj_y", &tf_adj_y));

  auto ng_lhs_axes = out_axes;
  auto ng_rhs_axes = out_axes;
  if (tf_adj_x) {
    ng_lhs_axes.push_back(n_dims - 1);
    ng_lhs_axes.push_back(n_dims - 2);
    ng_lhs = ng::builder::numpy_transpose(ng_lhs, ng_lhs_axes);
  }
  if (tf_adj_y) {
    ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims - 2);
    ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims - 1);
    ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng_rhs_axes);
  } else {
    ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims - 1);
    ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims - 2);
    ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng_rhs_axes);
  }

  ng_lhs_shape = ng_lhs->get_shape();
  ng_rhs_shape = ng_rhs->get_shape();

  if (ng_lhs_shape[n_dims - 1] != ng_rhs_shape[0]) {
    return errors::InvalidArgument(
        "The last dimension of ng_lhs and the first dimension of ng_rhs "
        "should have the same size");
  }
  if (n_dims == 2) {
    SaveNgOp(ng_op_map, op->name(),
             make_shared<ngraph::op::Dot>(ng_lhs, ng_rhs));
  } else {
    auto output_shape = ng_lhs_shape;
    output_shape[n_dims - 1] = ng_rhs_shape[1];
    auto dot_output = make_shared<ngraph::op::Dot>(ng_lhs, ng_rhs);
    size_t compound_size = 1;
    for (int i = 0; i < out_axes.size(); i++) {
      compound_size *= output_shape[i];
    }
    auto dot_axes = out_axes;
    dot_axes.push_back(n_dims - 2);
    dot_axes.push_back(n_dims - 1);
    for (int i = 0; i < out_axes.size(); i++) {
      dot_axes.push_back(n_dims + i);
    }
    ng::Shape dot_shape = {compound_size, ng_lhs_shape[n_dims - 2],
                           ng_rhs_shape[1], compound_size};
    std::shared_ptr<ng::Node> dot_reshape;
    if (n_dims == 3) {
      dot_reshape = dot_output;
    } else {
      dot_reshape =
          make_shared<ngraph::op::Reshape>(dot_output, dot_axes, dot_shape);
    }
    ng::Shape tmp_shape = {1, ng_lhs_shape[n_dims - 2], ng_rhs_shape[1]};
    vector<shared_ptr<ngraph::Node>> tmp_tensors;
    for (size_t i = 0; i < dot_shape[0]; i++) {
      const std::vector<size_t> lower_bound{i, 0, 0, i};
      const std::vector<size_t> upper_bound{i + 1, dot_shape[1], dot_shape[2],
                                            i + 1};
      auto slice_out =
          make_shared<ngraph::op::Slice>(dot_reshape, lower_bound, upper_bound);
      auto reshape_out = make_shared<ngraph::op::Reshape>(
          slice_out, ng::AxisVector{0, 1, 2, 3}, tmp_shape);
      tmp_tensors.push_back(reshape_out);
    }
    auto concat_op = make_shared<ngraph::op::Concat>(tmp_tensors, 0);
    if (n_dims == 3) {
      SaveNgOp(ng_op_map, op->name(), concat_op);
    } else {
      SaveNgOp(ng_op_map, op->name(),
               make_shared<ngraph::op::Reshape>(
                   concat_op, ng::AxisVector{0, 1, 2}, output_shape));
    }
  }
  return Status::OK();
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

  bool is_nhwc = (tf_data_format == "NHWC");

  ng::AxisSet ng_broadcast_axes;

  if (is_nhwc) {
    for (size_t i = 0; i < ng_input_shape.size() - 1; i++) {
      ng_broadcast_axes.insert(i);
    }
  } else {
    for (size_t i = 0; i < ng_input_shape.size(); i++) {
      if (i != 1) {
        ng_broadcast_axes.insert(i);
      }
    }
  }

  auto ng_bias_broadcasted = make_shared<ng::op::Broadcast>(
      ng_bias, ng_input_shape, ng_broadcast_axes);
  auto ng_add = ng_input + ng_bias_broadcasted;

  SaveNgOp(ng_op_map, op->name(), ng_add);
  return Status::OK();
}

static Status TranslateBiasAddGradOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  std::string tf_data_format;
  if (GetNodeAttr(op->attrs(), "data_format", &tf_data_format) !=
      Status::OK()) {
    tf_data_format = "NHWC";
  }

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "BiasAddGrad data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  ng::AxisSet reduction_axes;
  shared_ptr<ng::Node> ng_biasadd_backprop;
  auto ng_input_shape = ng_input->get_shape();

  if (is_nhwc) {
    if (ng_input_shape.size() < 2) {
      return errors::InvalidArgument(
          "BiasAddGrad argument needs to have at least 2 dimensions for NHWC "
          "data format");
    }
    for (size_t i = 0; i < ng_input_shape.size() - 1; i++) {
      reduction_axes.insert(i);
    }
  } else {
    // Tensorflow NCHW format supports only 4D input/output tensor
    if (ng_input_shape.size() != 4) {
      return errors::InvalidArgument(
          "BiasAddGrad only support 4d input/output for NCHW data format");
    }
    for (size_t i = 0; i < ng_input_shape.size(); i++) {
      if (i != ng_input_shape.size() - 3) reduction_axes.insert(i);
    }
  }

  ng_biasadd_backprop = make_shared<ng::op::Sum>(ng_input, reduction_axes);

  SaveNgOp(ng_op_map, op->name(), ng_biasadd_backprop);
  return Status::OK();
}

static Status TranslateCastOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "DstT", &dtype));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  try {
    SaveNgOp(ng_op_map, op->name(),
             make_shared<ng::op::Convert>(ng_input, ng_et));
  } catch (const std::out_of_range&) {
    return errors::Unimplemented("Unsupported TensorFlow data type: ",
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
           make_shared<ng::op::Concat>(ng_args, size_t(concat_axis)));
  return Status::OK();
}

static Status TranslateConstOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
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
    return errors::Unimplemented("Unsupported TensorFlow data type: ",
                                 DataType_Name(dtype));
  }

  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

static Status TranslateConv2DOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
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
  BatchToNGraph(is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Reshape<3, 2, 0, 1>(ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  std::shared_ptr<ng::Node> ng_conv = make_shared<ng::op::Convolution>(
      ng_input, ng_filter, ng_strides, ng_dilations, ng_padding_below,
      ng_padding_above);

  BatchToTensorflow(is_nhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateConv2DBackpropFilterOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_data_batch, ng_output_delta;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_data_batch, nullptr, &ng_output_delta));

  std::vector<int32> tf_strides;
  std::string tf_padding_type;
  std::vector<int32> tf_dilations;
  std::string tf_data_format;

  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument("Data format is neither NHWC nor NCHW: ",
                                   op->type_string());
  }

  NGRAPH_VLOG(3) << "tf data format" << tf_data_format;
  bool is_nhwc = (tf_data_format == "NHWC");

  // Dilations in batch and depth dimensions must be 1
  if (is_nhwc) {
    if (tf_dilations[0] != 1 || tf_dilations[3] != 1) {
      return errors::InvalidArgument(
          "Dilations in batch and depth dimensions must be 1: ",
          op->type_string());
    }
  } else {
    if (tf_dilations[0] != 1 || tf_dilations[1] != 1) {
      return errors::InvalidArgument(
          "Dilations in batch and depth dimensions must be 1: ",
          op->type_string());
    }
  }

  std::vector<int64> tf_filter_sizes;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 1, static_input_map, &tf_filter_sizes));

  if (std::any_of(tf_filter_sizes.begin(), tf_filter_sizes.end(),
                  [](int32 size) { return size <= 0; })) {
    return errors::InvalidArgument("Filter sizes must be positive integers :",
                                   op->type_string());
  }

  NGRAPH_VLOG(3) << "tf filter size" << ng::join(tf_filter_sizes);
  NGRAPH_VLOG(3) << "tf filter size" << ng::join(tf_filter_sizes);
  NGRAPH_VLOG(3) << "tf strides" << ng::join(tf_strides);
  NGRAPH_VLOG(3) << "tf dilations" << ng::join(tf_dilations);
  NGRAPH_VLOG(3) << "tf padding type" << tf_padding_type;

  ng::Shape ng_filters_shape(4);
  ng::Strides ng_window_movement_strides_forward(2);
  ng::Strides ng_window_dilation_strides_forward(2);
  ng::CoordinateDiff ng_padding_below_forward{0, 0};
  ng::CoordinateDiff ng_padding_above_forward{0, 0};
  // H,W data_dilation is set to 1 , TF does not have this attribute
  ng::Strides ng_data_dilation_strides_forward(2, 1);

  // Convert inputs, args to nGraph Format
  // nGraph Data Format:
  //    nGraph Tensor           [N, C_IN, D1, ... Df]
  //    nGraph Filter           [C_OUT, C_IN, F1, ... Ff]
  //    nGraph Output Delta     [N, C_OUT, F1, ... Ff]
  //    nGraph Window Strides   [f]
  //    nGraph Window Dilations [f]
  //    nGraph Padding Below    [f]
  //    nGraph Padding Above    [f]
  //    nGraph Dilation Stride  [f]
  BatchToNGraph(is_nhwc, ng_data_batch);
  // tf_filter shape :
  // [filter_height, filter_width, in_channels, out_channels]
  // reshape for nGraph
  ng_filters_shape = {static_cast<unsigned int>(tf_filter_sizes[3]),
                      static_cast<unsigned int>(tf_filter_sizes[2]),
                      static_cast<unsigned int>(tf_filter_sizes[0]),
                      static_cast<unsigned int>(tf_filter_sizes[1])};
  BatchToNGraph(is_nhwc, ng_output_delta);
  BatchedOpParamToNGraph(is_nhwc, tf_strides,
                         ng_window_movement_strides_forward);
  BatchedOpParamToNGraph(is_nhwc, tf_dilations,
                         ng_window_dilation_strides_forward);
  // H, W of image/input and filter are required to figure out padding
  // arguments
  ng::Shape ng_filter_HW(2);
  ng::Shape ng_input_data_HW(2);

  auto& ng_data_batch_shape = ng_data_batch->get_shape();
  ng_input_data_HW[0] = ng_data_batch_shape[2];
  ng_input_data_HW[1] = ng_data_batch_shape[3];

  ng_filter_HW[0] = ng_filters_shape[2];
  ng_filter_HW[1] = ng_filters_shape[3];

  Builder::MakePadding(tf_padding_type, ng_input_data_HW, ng_filter_HW,
                       ng_window_movement_strides_forward,
                       ng_window_dilation_strides_forward,
                       ng_padding_below_forward, ng_padding_above_forward);

  NGRAPH_VLOG(3) << "ng input data shape" << ng::join(ng_data_batch_shape);
  NGRAPH_VLOG(3) << "ng filter shape" << ng::join(ng_filters_shape);
  NGRAPH_VLOG(3) << "ng output delta shape"
                 << ng::join(ng_output_delta->get_shape());
  NGRAPH_VLOG(3) << "ng strides"
                 << ng::join(ng_window_movement_strides_forward);
  NGRAPH_VLOG(3) << "ng dilations"
                 << ng::join(ng_window_dilation_strides_forward);
  NGRAPH_VLOG(3) << "ng padding type" << tf_padding_type;

  std::shared_ptr<ng::Node> ng_back_prop_filter =
      make_shared<ng::op::ConvolutionBackpropFilters>(
          ng_data_batch, ng_filters_shape, ng_output_delta,
          ng_window_movement_strides_forward,
          ng_window_dilation_strides_forward, ng_padding_below_forward,
          ng_padding_above_forward, ng_data_dilation_strides_forward);

  // Reshape the output to tf format : [filter_height, filter_width,
  // in_channels, out_channels]
  Reshape<2, 3, 1, 0>(ng_back_prop_filter);

  SaveNgOp(ng_op_map, op->name(), ng_back_prop_filter);
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
  BatchToNGraph(is_nhwc, ng_out_backprop);
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
  Reshape<3, 2, 0, 1>(ng_filter);

  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::CoordinateDiff ng_padding_below{0, 0};
  ng::CoordinateDiff ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  std::shared_ptr<ng::Node> ng_data =
      make_shared<ng::op::ConvolutionBackpropData>(
          ng_batch_shape, ng_filter, ng_out_backprop, ng_strides, ng_dilations,
          ng_padding_below, ng_padding_above,
          ng::Strides(ng_batch_shape.size() - 2, 1));

  BatchToTensorflow(is_nhwc, ng_data);

  SaveNgOp(ng_op_map, op->name(), ng_data);
  return Status::OK();
}

// Translate DepthToSpace op
static Status TranslateDepthToSpaceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
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
    // NCHW or NCHW_VEC_C
    case 1:
      channel_dimension = 1;
      break;
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
  ng::AxisVector ng_transpose_shape;
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
      ng_transpose_shape.push_back(0);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_transpose_shape.push_back(i + 1);
        ng_transpose_shape.push_back(i + 1 + num_spatial_dimensions);
      }
      ng_transpose_shape.push_back(channel_dimension + num_spatial_dimensions);

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
      ng_transpose_shape.push_back(0);
      ng_transpose_shape.push_back(1 + num_spatial_dimensions);
      for (int i = 0; i < num_spatial_dimensions; i++) {
        ng_transpose_shape.push_back(i + 2 + num_spatial_dimensions);
        ng_transpose_shape.push_back(i + 1);
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
  auto reshaped =
      make_shared<ng::op::Reshape>(ng_input, ng_axis_order, ng_reshape_shape);

  auto transposed = ng::builder::numpy_transpose(reshaped, ng_transpose_shape);

  ng::AxisVector ng_axis_order_second_reshape(transposed->get_shape().size());
  std::iota(ng_axis_order_second_reshape.begin(),
            ng_axis_order_second_reshape.end(), 0);
  auto final_reshape = make_shared<ng::op::Reshape>(
      transposed, ng_axis_order_second_reshape, ng_output_shape);
  SaveNgOp(ng_op_map, op->name(), final_reshape);

  return Status::OK();
}

static Status TranslateDepthwiseConv2dNativeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
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
  BatchToNGraph(is_nhwc, ng_input);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter->get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Reshape<3, 2, 0, 1>(ng_filter);

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
    auto ng_sliced_input =
        make_shared<ng::op::Slice>(ng_input, lower_bound, upper_bound);

    const std::vector<size_t> f_lower_bound{0, i, 0, 0};
    const std::vector<size_t> f_upper_bound{filter_shape[0], i + 1,
                                            filter_shape[2], filter_shape[3]};
    auto ng_sliced_filter =
        make_shared<ng::op::Slice>(ng_filter, f_lower_bound, f_upper_bound);

    NGRAPH_VLOG(3) << "depthwise conv 2d.";
    NGRAPH_VLOG(3) << "sliced shape " << ng::join(ng_sliced_input->get_shape());
    NGRAPH_VLOG(3) << "filter shape "
                   << ng::join(ng_sliced_filter->get_shape());

    auto ng_conv = make_shared<ng::op::Convolution>(
        ng_sliced_input, ng_sliced_filter, ng_strides, ng_dilations,
        ng_padding_below, ng_padding_above);
    ng_args.push_back(ng_conv);
  }

  size_t ng_concatenation_axis = 1;  // channel axis
  std::shared_ptr<ng::Node> ng_concat =
      make_shared<ng::op::Concat>(ng_args, ng_concatenation_axis);

  BatchToTensorflow(is_nhwc, ng_concat);
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
  auto shape_size = shape.size();
  if (dim_vec[0] < 0) {
    // allow range [-rank(input) - 1, rank(input)]
    // where -1 append new axis at the end
    dim_vec[0] = shape_size + dim_vec[0] + 1;
  }
  auto out_shape = shape;
  out_shape.insert(out_shape.begin() + size_t(dim_vec[0]), 1);
  std::vector<size_t> shape_dimensions(shape.size());
  std::iota(shape_dimensions.begin(), shape_dimensions.end(), 0);
  std::shared_ptr<ng::Node> ng_expand_dim =
      make_shared<ng::op::Reshape>(ng_input, shape_dimensions, out_shape);

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

  ng::Shape ng_output_shape(dims_vec.size());
  ng::AxisSet ng_axis_set;
  for (size_t i = 0; i < dims_vec.size(); ++i) {
    ng_output_shape[i] = dims_vec[i];
    ng_axis_set.insert(i);
  }
  SaveNgOp(ng_op_map, op->name(), make_shared<ng::op::Broadcast>(
                                      ng_value, ng_output_shape, ng_axis_set));
  return Status::OK();
}

static Status TranslateFloorDivOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  auto ng_floordiv = [](std::shared_ptr<ng::Node> ng_input1,
                        std::shared_ptr<ng::Node> ng_input2) {
    return std::make_shared<ng::op::Floor>(
        std::make_shared<ng::op::Divide>(ng_input1, ng_input2));
  };
  return TranslateBinaryOp(op, static_input_map, ng_op_map, ng_floordiv);
}

static Status TranslateFloorModOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  auto ng_floormod = [](std::shared_ptr<ng::Node> ng_input1,
                        std::shared_ptr<ng::Node> ng_input2) {
    auto floordiv = std::make_shared<ng::op::Floor>(
        std::make_shared<ng::op::Divide>(ng_input1, ng_input2));
    return std::make_shared<ng::op::Subtract>(
        ng_input1, std::make_shared<ng::op::Multiply>(floordiv, ng_input2));
  };
  return TranslateBinaryOp(op, static_input_map, ng_op_map, ng_floormod);
}

static Status TranslateFusedBatchNormOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  bool tf_is_training;
  if (GetNodeAttr(op->attrs(), "is_training", &tf_is_training) !=
      Status::OK()) {
    NGRAPH_VLOG(3) << "is_training attribute not present, setting to true";
    tf_is_training = true;
  }

  NGRAPH_VLOG(3) << "is_training: " << tf_is_training;

  shared_ptr<ng::Node> ng_input, ng_scale, ng_offset, ng_mean, ng_variance;
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

  BatchToNGraph(is_nhwc, ng_input);

  std::shared_ptr<ng::Node> ng_batch_norm;

  if (tf_is_training) {
    ng_batch_norm = make_shared<ng::op::BatchNormTraining>(tf_epsilon, ng_scale,
                                                           ng_offset, ng_input);

    shared_ptr<ngraph::Node> ng_y, ng_mean, ng_variance;
    ng_y = make_shared<ng::op::GetOutputElement>(ng_batch_norm, 0);
    ng_mean = make_shared<ng::op::GetOutputElement>(ng_batch_norm, 1);
    ng_variance = make_shared<ng::op::GetOutputElement>(ng_batch_norm, 2);

    BatchToTensorflow(is_nhwc, ng_y);

    SaveNgOp(ng_op_map, op->name(), ng_y);
    SaveNgOp(ng_op_map, op->name(), ng_mean);
    SaveNgOp(ng_op_map, op->name(), ng_variance);
    // Output reserve_space_1: A 1D Tensor for the computed batch mean, to be
    // reused in the gradient computation.
    SaveNgOp(ng_op_map, op->name(), ng_mean);
    // Output reserve_space_2: A 1D Tensor for the computed batch variance
    //(inverted variance in the cuDNN case), to be reused in the gradient
    // computation.
    SaveNgOp(ng_op_map, op->name(), ng_variance);
  } else {
    ng_batch_norm = make_shared<ng::op::BatchNormInference>(
        tf_epsilon, ng_scale, ng_offset, ng_input, ng_mean, ng_variance);
    BatchToTensorflow(is_nhwc, ng_batch_norm);
    SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
  }

  SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
  return Status::OK();
}

static Status TranslateFusedBatchNormGradOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCount(op, 5));

  bool tf_is_training;
  // We only support is_training=true case. We marked rejection for the case
  // is_training=false.
  if (GetNodeAttr(op->attrs(), "is_training", &tf_is_training) !=
      Status::OK()) {
    NGRAPH_VLOG(3) << "is_training attribute not present, setting to true";
    tf_is_training = true;
  }

  NGRAPH_VLOG(3) << "is_training: " << tf_is_training;

  shared_ptr<ng::Node> ng_delta;
  shared_ptr<ng::Node> ng_input;
  shared_ptr<ng::Node> ng_scale;
  shared_ptr<ng::Node> ng_mean;
  shared_ptr<ng::Node> ng_variance;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_delta, &ng_input,
                                   &ng_scale, &ng_mean, &ng_variance));

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "FusedBatchnormGrad data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << "data_format: " << tf_data_format;

  float tf_epsilon;
  if (GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon) != Status::OK()) {
    NGRAPH_VLOG(3) << "epsilon attribute not present, setting to 0.0001";
    tf_epsilon = 0.0001;
  }

  NGRAPH_VLOG(3) << "epsilon: " << tf_epsilon;

  // TODO: We are temporarily supplying a fake value for beta here
  // (all zero, same shape/et as scale/gamma), because Tensorflow does not give
  // beta to us. This should work because nGraph should not actually use beta.
  // The nGraph op may change to discard this parameter. Update this when nGraph
  // does.
  shared_ptr<ng::Node> ng_beta = std::make_shared<ngraph::op::Constant>(
      ng_scale->get_element_type(), ng_scale->get_shape(),
      std::vector<std::string>{ng::shape_size(ng_scale->get_shape()), "0"});

  BatchToNGraph(is_nhwc, ng_input);
  BatchToNGraph(is_nhwc, ng_delta);

  std::shared_ptr<ng::Node> ng_batch_norm_backprop;

  ng_batch_norm_backprop = make_shared<ng::op::BatchNormTrainingBackprop>(
      tf_epsilon, ng_scale, ng_beta, ng_input, ng_mean, ng_variance, ng_delta);

  shared_ptr<ngraph::Node> ng_input_delta_op =
      make_shared<ng::op::GetOutputElement>(ng_batch_norm_backprop, 0);
  shared_ptr<ngraph::Node> ng_scale_delta_op =
      make_shared<ng::op::GetOutputElement>(ng_batch_norm_backprop, 1);
  shared_ptr<ngraph::Node> ng_beta_delta_op =
      make_shared<ng::op::GetOutputElement>(ng_batch_norm_backprop, 2);

  BatchToTensorflow(is_nhwc, ng_input_delta_op);

  SaveNgOp(ng_op_map, op->name(), ng_input_delta_op);
  SaveNgOp(ng_op_map, op->name(), ng_scale_delta_op);
  SaveNgOp(ng_op_map, op->name(), ng_beta_delta_op);
  // Output reserve_space_3: Unused placeholder to match the mean input
  // in FusedBatchNorm.
  std::shared_ptr<ng::Node> output_mean = make_shared<ngraph::op::Constant>(
      ng_mean->get_element_type(), ng::Shape{}, std::vector<std::string>{""});
  SaveNgOp(ng_op_map, op->name(), output_mean);
  // Output reserve_space_4: Unused placeholder to match the variance input
  // in FusedBatchNorm.
  std::shared_ptr<ng::Node> output_variance = make_shared<ngraph::op::Constant>(
      ng_variance->get_element_type(), ng::Shape{},
      std::vector<std::string>{""});
  SaveNgOp(ng_op_map, op->name(), output_variance);

  return Status::OK();
}

static Status TranslateIdentityOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_arg;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_arg));
  SaveNgOp(ng_op_map, op->name(), ng_arg);
  return Status::OK();
}

static Status TranslateL2LossOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto const_2 = make_shared<ng::op::Constant>(
      ng_input->get_element_type(), ng::Shape{}, std::vector<std::string>{"2"});

  std::shared_ptr<ng::Node> ng_pow =
      make_shared<ng::op::Multiply>(ng_input, ng_input);

  size_t input_rank = ng_input->get_shape().size();
  ng::AxisSet axes;
  for (auto i = 0; i < input_rank; ++i) {
    axes.insert(i);
  }

  std::shared_ptr<ng::Node> ng_sum = make_shared<ng::op::Sum>(ng_pow, axes);
  std::shared_ptr<ng::Node> ng_l2loss =
      make_shared<ng::op::Divide>(ng_sum, const_2);
  SaveNgOp(ng_op_map, op->name(), ng_l2loss);
  return Status::OK();
}

static Status TranslateMatMulOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_lhs, &ng_rhs));

  // Transpose arguments if requested.
  bool transpose_a = false;
  bool transpose_b = false;

  if (GetNodeAttr(op->attrs(), "transpose_a", &transpose_a) == Status::OK() &&
      transpose_a) {
    ng_lhs = ng::builder::numpy_transpose(ng_lhs, ng::AxisVector{1, 0});
  }
  if (GetNodeAttr(op->attrs(), "transpose_b", &transpose_b) == Status::OK() &&
      transpose_b) {
    ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng::AxisVector{1, 0});
  }

  // The default axis count for nGraph's Dot op is 1, which is just what
  // we need here.
  SaveNgOp(ng_op_map, op->name(), make_shared<ngraph::op::Dot>(ng_lhs, ng_rhs));
  return Status::OK();
}

static Status TranslateMaxOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_max_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_max_op));

  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> max_axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &max_axes));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(max_axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(max_axes.size());
  std::transform(
      max_axes.begin(), max_axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  ng::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

  std::shared_ptr<ng::Node> ng_max =
      make_shared<ng::op::Max>(ng_input, ng_reduction_axes);

  // If keep_dims is specified we need to reshape to put back the reduced
  // axes, with length 1.
  if (tf_keep_dims) {
    ng::Shape ng_result_shape_with_keep(input_rank);

    for (size_t i = 0; i < input_rank; i++) {
      ng_result_shape_with_keep[i] =
          ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
    }

    ng::AxisVector ng_axis_order(ng_max->get_shape().size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

    ng_max = make_shared<ng::op::Reshape>(ng_max, ng_axis_order,
                                          ng_result_shape_with_keep);
  }

  SaveNgOp(ng_op_map, op->name(), ng_max);
  return Status::OK();
}

static Status TranslateMaxPoolOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
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
  BatchToNGraph(is_nhwc, ng_input);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for MaxPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_maxpool =
      make_shared<ng::op::MaxPool>(ng_input, ng_kernel_shape, ng_strides,
                                   ng_padding_below, ng_padding_above);

  BatchToTensorflow(is_nhwc, ng_maxpool);

  NGRAPH_VLOG(3) << "maxpool outshape: {" << ng::join(ng_maxpool->get_shape())
                 << "}";

  SaveNgOp(ng_op_map, op->name(), ng_maxpool);
  return Status::OK();
}

static Status TranslateMaxPoolGradOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_grad;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input, nullptr, &ng_grad));

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
        "MaxPoolGrad data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");
  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);

  BatchedOpParamToNGraph(is_nhwc, ng_input->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(is_nhwc, ng_input);
  BatchToNGraph(is_nhwc, ng_grad);

  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_maxpool_backprop =
      make_shared<ng::op::MaxPoolBackprop>(ng_input, ng_grad, ng_kernel_shape,
                                           ng_strides, ng_padding_below,
                                           ng_padding_above);
  BatchToTensorflow(is_nhwc, ng_maxpool_backprop);
  NGRAPH_VLOG(3) << "maxpoolbackprop outshape: {"
                 << ng::join(ng_maxpool_backprop->get_shape()) << "}";
  SaveNgOp(ng_op_map, op->name(), ng_maxpool_backprop);
  return Status::OK();
}

static Status TranslateMeanOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_axes_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_axes_op));

  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> mean_axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &mean_axes));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = ng_input->get_shape().size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(mean_axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(mean_axes.size());
  std::transform(
      mean_axes.begin(), mean_axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  ng::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

  std::shared_ptr<ng::Node> ng_mean =
      ng::builder::mean(ng_input, ng_reduction_axes);

  // If keep_dims is specified we need to reshape to put back the reduced
  // axes, with length 1.
  if (tf_keep_dims) {
    ng::Shape ng_result_shape_with_keep(input_rank);

    for (size_t i = 0; i < input_rank; i++) {
      ng_result_shape_with_keep[i] =
          ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
    }

    ng::AxisVector ng_axis_order(ng_mean->get_shape().size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

    ng_mean = make_shared<ng::op::Reshape>(ng_mean, ng_axis_order,
                                           ng_result_shape_with_keep);
  }

  SaveNgOp(ng_op_map, op->name(), ng_mean);
  return Status::OK();
}

static Status TranslateMinOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_min_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_min_op));

  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> min_axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &min_axes));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(min_axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(min_axes.size());
  std::transform(
      min_axes.begin(), min_axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  ng::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

  std::shared_ptr<ng::Node> ng_min =
      make_shared<ng::op::Min>(ng_input, ng_reduction_axes);

  // If keep_dims is specified we need to reshape to put back the reduced
  // axes, with length 1.
  if (tf_keep_dims) {
    ng::Shape ng_result_shape_with_keep(input_rank);

    for (size_t i = 0; i < input_rank; i++) {
      ng_result_shape_with_keep[i] =
          ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
    }

    ng::AxisVector ng_axis_order(ng_min->get_shape().size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

    ng_min = make_shared<ng::op::Reshape>(ng_min, ng_axis_order,
                                          ng_result_shape_with_keep);
  }

  SaveNgOp(ng_op_map, op->name(), ng_min);
  return Status::OK();
}

static Status TranslatePackOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 1));

  ng::NodeVector ng_concat_inputs;

  for (size_t i = 0; i < op->num_inputs(); ++i) {
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
    output_shape[(i < concat_axis) ? i : i + 1] = input_shape[i];
  }
  output_shape[concat_axis] = op->num_inputs();

  ng::AxisVector ng_axis_order(input_rank);
  std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

  if (concat_axis == input_rank) {
    // need to add extra dimension before we concatenate
    // along it
    ng::Shape extended_shape = input_shape;
    extended_shape.push_back(1);
    for (size_t i = 0; i < ng_concat_inputs.size(); ++i) {
      ng_concat_inputs[i] = make_shared<ng::op::Reshape>(
          ng_concat_inputs[i], ng_axis_order, extended_shape);
    }
    ng_axis_order.push_back(input_rank);
  }

  auto concat = make_shared<ng::op::Concat>(ng_concat_inputs, concat_axis);
  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Reshape>(concat, ng_axis_order, output_shape));
  return Status::OK();
}

static Status TranslatePadOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_paddings_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_paddings_op));

  std::vector<int64> paddings;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &paddings));

  NGRAPH_VLOG(3) << "{" << ng::join(paddings) << "}";

  if (paddings.size() % 2 != 0) {
    return errors::InvalidArgument(
        "Constant node for paddings does not have an even number of "
        "elements");
  }

  ng::Shape padding_below(paddings.size() / 2);
  ng::Shape padding_above(paddings.size() / 2);
  ng::Shape padding_interior(paddings.size() / 2);

  for (size_t i = 0; i < paddings.size() / 2; i++) {
    padding_below[i] = paddings[2 * i];
    padding_above[i] = paddings[2 * i + 1];
    padding_interior[i] = 0;
  }

  NGRAPH_VLOG(3) << "{" << ng::join(padding_below) << "}";
  NGRAPH_VLOG(3) << "{" << ng::join(padding_above) << "}";

  // For PadV1 it seems the value is always zero.
  auto pad_val_op = make_shared<ng::op::Constant>(
      ng_input->get_element_type(), ng::Shape{}, std::vector<std::string>{"0"});
  auto pad_op = make_shared<ng::op::Pad>(ng_input, pad_val_op, padding_below,
                                         padding_above, padding_interior);

  SaveNgOp(ng_op_map, op->name(), pad_op);
  return Status::OK();
}

static Status TranslateProdOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_axes_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_axes_op));

  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> prod_axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &prod_axes));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(prod_axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(prod_axes.size());
  std::transform(
      prod_axes.begin(), prod_axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  ng::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

  std::shared_ptr<ng::Node> ng_prod =
      make_shared<ng::op::Product>(ng_input, ng_reduction_axes);

  // If keep_dims is specified we need to reshape to put back the reduced
  // axes, with length 1.
  if (tf_keep_dims) {
    ng::Shape ng_result_shape_with_keep(input_rank);

    for (size_t i = 0; i < input_rank; i++) {
      ng_result_shape_with_keep[i] =
          ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
    }

    ng::AxisVector ng_axis_order(ng_prod->get_shape().size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

    ng_prod = make_shared<ng::op::Reshape>(ng_prod, ng_axis_order,
                                           ng_result_shape_with_keep);
  }

  SaveNgOp(ng_op_map, op->name(), ng_prod);
  return Status::OK();
}

static Status TranslateReciprocalOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [](std::shared_ptr<ng::Node> n) {
        // Create a constant tensor populated with the value -1.
        // (1/x = x^(-1))
        auto et = n->get_element_type();
        auto shape = n->get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-1");
        auto ng_exponent =
            std::make_shared<ng::op::Constant>(et, shape, constant_values);

        // Raise each element of the input to the power -1.
        return std::make_shared<ng::op::Power>(n, ng_exponent);
      });
}

static void ComputeScaleOffsetFolded(const uint& num_bits,
                                     const bool& unsigned_type,
                                     const bool& scaled, const float min_range,
                                     const float max_range, float* scale,
                                     int* offset) {
  // ======= 1. Definitions =======
  // min<Q>, max<Q>: the min and max of the quantized data type
  // range<Q> = qmax<Q> - qmin<Q> (eg, 255 for u8, i8)
  // max<R>, min<R>: The max and min of the real type
  // range<R> = max<R> - min<R>
  // ======= 2. Unquantized maths =======
  // In what follows, we discuss the computations assuming
  // Q and R are unquantized to simplify the discussion
  // 2.1 In min-max representation:
  // max<R> maps to max<Q>, min<R> maps to min<Q>, R x maps to Q y
  // y = max<Q> - ((max<R> - x) * range<Q>) / range<R>
  // or, y = (range<Q> * x / range<R>) - ((min<R> * range<Q>) / range<R>)
  // 2.2 In scale-zero representation: y = (x/s) + z
  // Therefore, s = range<R> / range<Q>
  // and z = min<Q> - (min<R> / s)
  // ======= 3. Quantized maths =======
  // 3.1 Quantize: y = cast<Q>(clamp(round(x / s) + z, min<Q>, max<Q>)))
  // 3.2 Dequantize: x = cast<R>(y - z) * s
  // ======= 4. Notes  =======
  // 4.1 Scale s is of type R, zero/offset z is of type Q
  // 4.2 Type matching dictates, (min<R> / s) must evaluate to type Q
  // Hence, it needs to be rounded to integral type
  // 4.3 Scaled mode. Consider Q = int8, which represents [-128, 127]
  // But scaled mode is symmetric. Hence it represents [-127, 127]
  // 4.4 By definition, scaled mode includes 0. For others, we force the
  // inclusion of 0 in the range

  int scaled_elide = scaled ? 1 : 0;
  int min_type = 0;
  if (!unsigned_type) {
    min_type = -((1 << (num_bits - 1)) - scaled_elide);
  }
  uint raise_to = num_bits - (unsigned_type ? 0 : 1);
  int max_type = (1 << raise_to) - 1;
  float adj_min_range, adj_max_range;
  if (scaled) {
    auto abs_min_range = std::abs(min_range);
    auto abs_max_range = std::abs(max_range);
    auto range_boundary = std::max(abs_min_range, abs_max_range);
    adj_min_range = unsigned_type ? 0 : -range_boundary;
    adj_max_range = range_boundary;
  } else {
    // TODO: Adjust range or fail?
    adj_min_range = std::min(min_range, 0.0f);
    adj_max_range = std::max(max_range, 0.0f);
  }
  *scale = (adj_max_range - adj_min_range) / (max_type - min_type);
  // TODO: should it be: round(adj_min_range / *scale) (or floor)?
  *offset = min_type - std::lround(adj_min_range / *scale);

  // TODO: currently both TranslateQuantizeV2Op and TranslateDequantizeOp assume
  // static input and we compute the scale-offset into ng_consts. Later we need
  // to support converting min-max to scale-offset using a ngraph subgraph
}

template <typename T>
Status QuantizeAndDequantizeV2Helper(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    const bool& range_given, const bool& signed_input, const int& num_bits,
    float* scale_out) {
  // TODO: currently handling only float, generalize later?
  T min_range, max_range;
  if (range_given) {
    std::vector<T> input_min, input_max;
    TF_RETURN_IF_ERROR(
        GetStaticInputVector(op, 1, static_input_map, &input_min));
    TF_RETURN_IF_ERROR(
        GetStaticInputVector(op, 2, static_input_map, &input_max));
    if (input_min.size() != 1) {
      return errors::InvalidArgument(
          "QuantizeAndDequantizeV2 Op: input_min must be scalar. Got a vector "
          "of size, ",
          input_min.size());
    }
    if (input_max.size() != 1) {
      return errors::InvalidArgument(
          "QuantizeAndDequantizeV2 Op: input_max must be scalar. Got a vector "
          "of size, ",
          input_max.size());
    }
    min_range = input_min[0];
    max_range = input_max[0];
    if (min_range > max_range) {
      return errors::InvalidArgument(
          "Expected QuantizeAndDequantizeV2's input_min <= input_max but got, "
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
  T scale, inverse_scale;
  if (scale_from_min_side < scale_from_max_side) {
    scale = scale_from_min_side;
    inverse_scale = min_range / min_quantized;
    // max_range = max_quantized * inverse_scale;
  } else {
    scale = scale_from_max_side;
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
  auto ng_scale = std::make_shared<ng::op::Constant>(
      ng_r_et, ng::Shape(), std::vector<float>({scale}));
  auto ng_offset = std::make_shared<ng::op::Constant>(ng_q_et, ng::Shape(),
                                                      std::vector<int>({0}));
  ng::op::Quantize::RoundMode ng_round_mode =
      ng::op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;
  auto ng_quant = make_shared<ng::op::Quantize>(
      ng_input, ng_scale, ng_offset, ng_q_et, ng::AxisSet(), ng_round_mode);
  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Dequantize>(ng_quant, ng_scale, ng_offset,
                                           ng_r_et, ng::AxisSet()));

  // TODO: what of clamping?
  return Status::OK();
}

static Status TranslateQuantizedMaxPoolOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_min, &ng_max));
  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  bool is_nhwc = true;  // TODO, is this correct?
  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);
  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_input->get_output_shape(0),
                         ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(is_nhwc, ng_input);
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_padding_below, ng_padding_above);

  // Creating and passing dummy nodes to ScaledQuantizedMaxPool because it does
  // not use them. If it ever starts using min/max, the dummy min-max would
  // cause it to fail
  shared_ptr<ng::Node> dummy_min(nullptr), dummy_max(nullptr);
  std::shared_ptr<ng::Node> ng_quant_maxpool =
      ng::builder::ScaledQuantizedMaxPool(ng_input, ng_kernel_shape, ng_strides,
                                          ng_padding_below, ng_padding_above,
                                          dummy_min, dummy_max);
  BatchToTensorflow(is_nhwc, ng_quant_maxpool);
  SaveNgOp(ng_op_map, op->name(), ng_quant_maxpool);
  // For maxpool input min-max remains unchanged and is just propagated along
  // https://github.com/tensorflow/tensorflow/blob/9590c4c32dd4346ea5c35673336f5912c6072bf2/tensorflow/core/kernels/quantized_pooling_ops.cc#L99
  SaveNgOp(ng_op_map, op->name(), ng_min);
  SaveNgOp(ng_op_map, op->name(), ng_max);
  return Status::OK();
}

static Status TranslateQuantizeV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, nullptr, nullptr));

  std::vector<float> ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &ng_min));
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &ng_max));
  if (ng_min.size() != 1) {
    return errors::InvalidArgument(
        "QuantizeV2 Op: Min must be scalar. Got a vector of size, ",
        ng_min.size());
  }
  if (ng_max.size() != 1) {
    return errors::InvalidArgument(
        "QuantizeV2 Op: Max must be scalar. Got a vector of size, ",
        ng_max.size());
  }

  if (ng_min[0] > ng_max[0]) {
    return errors::InvalidArgument(
        "Expected QuantizeV2's min <= max but got, "
        "min = ",
        ng_min[0], " and max = ", ng_max[0]);
  }

  string mode;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "mode", &mode));

  // TODO: Since, currently only ng::HALF_AWAY_FROM_ZERO is supported,
  // just reading this value here, but not using it for now.
  string round_mode;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "round_mode", &round_mode));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "T", &dtype));

  int num_bits;
  bool unsigned_type;
  switch (dtype) {
    case DT_QINT8:
      num_bits = 8;
      unsigned_type = false;
      break;
    case DT_QUINT8:
      num_bits = 8;
      unsigned_type = true;
      break;
    default:
      return errors::InvalidArgument(
          "Expected QuantizeV2's datatype to be of int8 or uint8 but got ",
          DataTypeString(dtype));
  }

  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantize_op.cc#L107
  // Form this line here, we see that min and max is a single value.
  // Hence picking only the first element from the ng_min, ng_max vectors

  float ng_scale_val;
  int ng_offset_val;
  try {
    ComputeScaleOffsetFolded(num_bits, unsigned_type,
                             (mode.compare("SCALED") == 0), ng_min[0],
                             ng_max[0], &ng_scale_val, &ng_offset_val);
  } catch (const std::exception& e) {
    return errors::Internal("Unhandled exception in ComputeScaleOffset: ",
                            op->name(), " (", op->type_string(), ")\n",
                            op->def().DebugString(), "\n", "what(): ",
                            e.what());
  }

  auto ng_scale = std::make_shared<ng::op::Constant>(
      ng::element::f32, ng::Shape(), std::vector<float>({ng_scale_val}));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  // Cast offset appropriately? Currently using int
  auto ng_offset = std::make_shared<ng::op::Constant>(
      ng_et, ng::Shape(), std::vector<int>({ng_offset_val}));

  // TODO: Only RoundMode = HALF_AWAY_FROM_ZERO is supported, for now.
  // Support HALF_TO_EVEN later
  ng::op::Quantize::RoundMode ng_round_mode =
      ng::op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;

  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Quantize>(ng_input, ng_scale, ng_offset, ng_et,
                                         ng::AxisSet(), ng_round_mode));
  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Constant>(ng::element::f32, ng::Shape(),
                                         std::vector<float>({ng_min[0]})));
  // TODO: For quantizev2 revisit output min-max (which would change in case
  // input min-max are too close. For now just propagating inputs
  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Constant>(ng::element::f32, ng::Shape(),
                                         std::vector<float>({ng_max[0]})));
  return Status::OK();
}

static Status TranslateDequantizeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, nullptr, nullptr));

  std::vector<float> ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &ng_min));
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &ng_max));
  if (ng_min.size() != 1) {
    return errors::InvalidArgument(
        "Dequantize Op: Min must be scalar. Got a vector of size, ",
        ng_min.size());
  }
  if (ng_max.size() != 1) {
    return errors::InvalidArgument(
        "Dequantize Op: Max must be scalar. Got a vector of size, ",
        ng_max.size());
  }

  if (ng_min[0] > ng_max[0]) {
    return errors::InvalidArgument(
        "Expected Dequantize's min <= max but got, "
        "min = ",
        ng_min[0], " and max = ", ng_max[0]);
  }

  string mode;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "mode", &mode));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "T", &dtype));

  int num_bits;
  bool unsigned_type;
  switch (dtype) {
    case DT_QINT8:
      num_bits = 8;
      unsigned_type = false;
      break;
    case DT_QUINT8:
      num_bits = 8;
      unsigned_type = true;
      break;
    default:
      return errors::InvalidArgument(
          "Expected Dequantize's datatype to be of int8 or uint8 but got ",
          dtype);
  }

  float ng_scale_val;
  int ng_offset_val;
  try {
    ComputeScaleOffsetFolded(num_bits, unsigned_type,
                             (mode.compare("SCALED") == 0), ng_min[0],
                             ng_max[0], &ng_scale_val, &ng_offset_val);
  } catch (const std::exception& e) {
    return errors::Internal("Unhandled exception in ComputeScaleOffset: ",
                            op->name(), " (", op->type_string(), ")\n",
                            op->def().DebugString(), "\n", "what(): ",
                            e.what());
  }

  auto ng_scale = std::make_shared<ng::op::Constant>(
      ng::element::f32, ng::Shape(), std::vector<float>({ng_scale_val}));

  ng::element::Type ng_et;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

  // Cast offset appropriately? Currently using int
  auto ng_offset = std::make_shared<ng::op::Constant>(
      ng_et, ng::Shape(), std::vector<int>({ng_offset_val}));

  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Dequantize>(ng_input, ng_scale, ng_offset,
                                           ng::element::f32, ng::AxisSet()));
  return Status::OK();
}

static Status TranslateReluOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  SaveNgOp(ng_op_map, op->name(), make_shared<ng::op::Relu>(ng_input));
  return Status::OK();
}

static Status TranslateRelu6Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto constant_6 = make_shared<ng::op::Constant>(
      ng_input->get_element_type(), ng_input->get_shape(),
      std::vector<std::string>(ng::shape_size(ng_input->get_shape()), "6"));
  auto relu6_op = make_shared<ng::op::Minimum>(
      make_shared<ng::op::Relu>(ng_input), constant_6);

  SaveNgOp(ng_op_map, op->name(), relu6_op);
  return Status::OK();
}

static Status TranslateReluGradOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_arg, ng_delta;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_delta, &ng_arg));

  auto ng_relu_grad = std::make_shared<ng::op::ReluBackprop>(ng_arg, ng_delta);
  SaveNgOp(ng_op_map, op->name(), ng_relu_grad);
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

  size_t output_rank = shape.size();
  size_t num_input_elements = ng::shape_size(ng_input->get_shape());

  //
  // If there is a single "-1" in the result shape, we have to auto-infer
  // the length of that dimension.
  //
  size_t inferred_pos;
  size_t product_of_rest = 1;
  bool seen_inferred = false;
  for (size_t i = 0; i < output_rank; i++) {
    if (shape[i] == -1) {
      if (seen_inferred) {
        return errors::InvalidArgument(
            "Multiple -1 dimensions in result shape");
      }
      inferred_pos = i;
      seen_inferred = true;
    } else {
      product_of_rest *= shape[i];
    }
  }

  if (seen_inferred) {
    if (num_input_elements % product_of_rest != 0) {
      NGRAPH_VLOG(3) << "{" << ng::join(ng_input->get_shape()) << "}";
      NGRAPH_VLOG(3) << "{" << ng::join(shape) << "}";
      return errors::InvalidArgument(
          "Product of known dimensions (", product_of_rest,
          ") does not evenly divide the number of input elements (",
          num_input_elements, ")");
    }
    shape[inferred_pos] = num_input_elements / product_of_rest;
  }

  //
  // Convert the values from the constant into an nGraph::Shape, and
  // construct the axis order while we are at it.
  //
  ng::Shape ng_shape(output_rank);

  for (size_t i = 0; i < output_rank; i++) {
    ng_shape[i] = shape[i];
  }

  ng::AxisVector ng_axis_order(ng_input->get_shape().size());
  std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Reshape>(ng_input, ng_axis_order, ng_shape));
  return Status::OK();
}

static Status TranslateRsqrtOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [](std::shared_ptr<ng::Node> n) {
        // Create a constant tensor populated with the value -1/2.
        // (1/sqrt(x) = x^(-1/2))
        auto et = n->get_element_type();
        auto shape = n->get_shape();
        std::vector<std::string> constant_values(ng::shape_size(shape), "-0.5");
        auto ng_exponent =
            std::make_shared<ng::op::Constant>(et, shape, constant_values);

        // Raise each element of the input to the power -0.5.
        return std::make_shared<ng::op::Power>(n, ng_exponent);
      });
}

static Status TranslateShapeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  // the shape of the input tensor which will be the value to the Constant Op
  auto input_shape = ng_input->get_shape();

  // the rank of the input tensor which will be the shape to the Constant Op
  auto rank = input_shape.size();

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  // the inputs to the Constant Op
  ng::element::Type type;
  TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

  auto shape = ng::Shape(1, rank);

  std::vector<int> values(rank);
  for (int i = 0; i < rank; i++) {
    values[i] = input_shape[i];
  }
  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Constant>(type, shape, values));
  return Status::OK();
}

static Status TranslateSigmoidGradOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  shared_ptr<ng::Node> ng_delta;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_delta));

  auto ng_mul = ng_input * ng_delta;
  auto ng_subtract = make_shared<ng::op::Constant>(ng_input->get_element_type(),
                                                   ng_input->get_shape(),
                                                   std::vector<int>({1})) -
                     ng_input;
  auto ng_result = ng_mul * ng_subtract;

  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

static Status TranslateSigmoidOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto exp_op =
      make_shared<ng::op::Exp>(make_shared<ng::op::Negative>(ng_input));
  auto constant_1 = make_shared<ng::op::Constant>(
      ng_input->get_element_type(), ng_input->get_shape(),
      std::vector<std::string>(ng::shape_size(ng_input->get_shape()), "1"));

  auto denominator_op = make_shared<ng::op::Add>(constant_1, exp_op);

  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Divide>(constant_1, denominator_op));
  return Status::OK();
}

static Status TranslateSizeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
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
  auto ng_result = make_shared<ng::op::Constant>(type, ng::Shape(0),
                                                 std::vector<int64>({result}));

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
  auto ng_slice = make_shared<ng::op::Slice>(ng_input, l, u);
  SaveNgOp(ng_op_map, op->name(), ng_slice);
  return Status::OK();
}

static Status TranslateSnapshotOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_arg;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_arg));

  SaveNgOp(ng_op_map, op->name(), ng_arg);
  return Status::OK();
}

static Status TranslateSoftmaxOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  auto ng_input_shape = ng_input->get_shape();

  // We apply softmax on the 2nd dimension by following TF
  // And we restrict the softmax input argument to be 2D for now
  ng::AxisSet ng_axes_softmax;
  auto shape_size = ng_input_shape.size();

  if (shape_size != 2) {
    return errors::InvalidArgument("TF Softmax logits must be 2-dimensional");
  }

  ng_axes_softmax.insert(1);

  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Softmax>(ng_input, ng_axes_softmax));
  return Status::OK();
}

// Translate SpaceToDepthOp
static Status TranslateSpaceToDepthOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
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
    // NCHW or NCHW_VEC_C
    case 1:
      height_index = 2;
      width_index = 3;
      channel_index = 1;
      break;
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

  for (size_t counter_height = 0; counter_height < block_size;
       counter_height++) {
    for (size_t counter_width = 0; counter_width < block_size;
         counter_width++) {
      std::vector<size_t> begin = {0, 0, 0, 0};
      begin[width_index] = counter_width;
      begin[height_index] = counter_height;
      std::vector<size_t> strides = {1, 1, 1, 1};
      strides[width_index] = size_t(block_size);
      strides[height_index] = size_t(block_size);
      strided_slice_result.push_back(
          make_shared<ng::op::Slice>(ng_input, begin, upper, strides));
    }
  }

  SaveNgOp(ng_op_map, op->name(), make_shared<ngraph::op::Concat>(
                                      strided_slice_result, channel_index));
  return Status::OK();
}

static Status TranslateSparseSoftmaxCrossEntropyWithLogitsOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  // TF op Inputs:
  //  1. Logits/Features:
  //    Shape : [BatchSize, NumOfClasses]
  //     Type : float
  //  2. Label
  //    Shape : [BatchSize]
  //    Range : [0, NumOfClasses)
  //     Type : int
  shared_ptr<ng::Node> ng_features, ng_labels;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_features, &ng_labels));

  ng::Shape ng_features_shape = ng_features->get_shape();
  ng::Shape ng_labels_shape = ng_labels->get_shape();
  NGRAPH_VLOG(3) << " number of classes " << ng_features_shape[1];
  NGRAPH_VLOG(3) << " Batch " << ng_features_shape[0];

  // Logits/Features must be 2-d shape
  if (ng_features_shape.size() != 2) {
    return errors::InvalidArgument(
        " Logits/Features must be shape 2-D, but got shape ",
        ng::join(ng_features_shape), " while building op ", op->type_string());
  }

  // Labels must be 1-d shape
  if (ng_labels_shape.size() != 1) {
    return errors::InvalidArgument(" Labels must be shape 1-D, but got shape ",
                                   ng::join(ng_labels_shape),
                                   " while building op ", op->type_string());
  }

  // Logits/Features and Labels must have the same first dimension
  if (ng_labels_shape[0] != ng_features_shape[0]) {
    return errors::InvalidArgument(
        " Logits/Features and Labels must have the same first dimension, got "
        "Logits shape ",
        ng::join(ng_features_shape), " and Labels shape ",
        ng::join(ng_labels_shape), " while building op ", op->type_string());
  }

  // Logits dimension 1 must be >0, i.e. NumOfClasses>0
  if (ng_features_shape[1] <= 0) {
    return errors::InvalidArgument(
        " Logits/Features must have atleast one class but got shape ",
        ng::join(ng_features_shape), " while building op ", op->type_string());
  }

  // ** Check invalid label index **
  // Labels should be in range [0, NumOfClasses): Cannot do this check here
  // If the labels are out of range, we would get nGraph Exception while
  // computing y_true using ng::op::OneHot

  // To implement a numericaly stable and precise implementation,
  // for this op, the implementation is inspired from the tf kernel
  // of this op found in
  // /tensorflow/core/kernels/sparse_xent_op.h
  // /tensorflow/core/kernels/sparse_xent_op.cc

  // axis for operation is 1
  ng::AxisSet ng_axes_class;
  ng_axes_class.insert(1);

  // compute max(logits) and broadcast to shape [B, NC]
  auto max_logits = make_shared<ng::op::Broadcast>(
      make_shared<ng::op::Max>(ng_features, ng_axes_class), ng_features_shape,
      ng_axes_class);

  // logits_normalized : (logits - max_logits)
  auto logits_normalized =
      make_shared<ng::op::Subtract>(ng_features, max_logits);

  // y_pred = exp(logits_normalized) / sum(exp(logits_normalized))
  auto exp_logits = make_shared<ng::op::Exp>(logits_normalized);
  auto sum_exp_logits = make_shared<ng::op::Broadcast>(
      make_shared<ng::op::Sum>(exp_logits, ng_axes_class), ng_features_shape,
      ng_axes_class);
  auto predicted_prob = make_shared<ng::op::Divide>(exp_logits, sum_exp_logits);

  // y_true : one_hot_float_labels
  auto ng_onehot_labels =
      make_shared<ng::op::OneHot>(ng_labels, ng_features_shape, 1);

  auto ng_onehot_labels_float = make_shared<ng::op::Convert>(
      ng_onehot_labels, ng_features->get_element_type());

  // Output 1
  // loss = sum[labels * {sum(log(exp(logits_normalized)))
  // - logits_normalized }]
  auto ng_loss = make_shared<ng::op::Sum>(
      make_shared<ng::op::Multiply>(
          make_shared<ng::op::Subtract>(
              make_shared<ng::op::Log>(sum_exp_logits), logits_normalized),
          ng_onehot_labels_float),
      ng_axes_class);

  // Output 2
  // backprop = y_pred - y_true
  auto ng_backprop =
      make_shared<ng::op::Subtract>(predicted_prob, ng_onehot_labels_float);

  SaveNgOp(ng_op_map, op->name(), ng_loss);
  SaveNgOp(ng_op_map, op->name(), ng_backprop);
  return Status::OK();
}

static Status TranslateSplitOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, nullptr, &ng_input));

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
  int split_dim = split_dim_vec[0];

  int size = shape[split_dim] / num_split;
  int cursor = 0;

  for (size_t i = 0; i < num_split; ++i) {
    lower[split_dim] = cursor;
    cursor += size;
    upper[split_dim] = cursor;

    std::string output_name = op->name();
    SaveNgOp(ng_op_map, op->name(),
             make_shared<ng::op::Slice>(ng_input, lower, upper));
  }
  return Status::OK();
}

static Status TranslateSplitVOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_length, ng_split_dim;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input, &ng_length, &ng_split_dim));

  std::vector<int> lengths;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &lengths));

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
      GetStaticInputVector(op, 2, static_input_map, &split_dim_vec));
  int split_dim = split_dim_vec[0];
  int cursor = 0;

  for (int i = 0; i < lengths.size(); ++i) {
    lower[split_dim] = cursor;
    cursor += lengths[i];
    upper[split_dim] = cursor;
    SaveNgOp(ng_op_map, op->name(),
             make_shared<ng::op::Slice>(ng_input, lower, upper));
  }
  return Status::OK();
}

static Status TranslateSquareOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(op, static_input_map, ng_op_map,
                          [](std::shared_ptr<ng::Node> n) {
                            return std::make_shared<ng::op::Multiply>(n, n);
                          });
}

static Status TranslateSquaredDifferenceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateBinaryOp(
      op, static_input_map, ng_op_map,
      [](std::shared_ptr<ng::Node> input1, std::shared_ptr<ng::Node> input2) {
        auto ng_diff = std::make_shared<ng::op::Subtract>(input1, input2);
        return std::make_shared<ng::op::Multiply>(ng_diff, ng_diff);
      });
}

static Status TranslateSqueezeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  std::vector<int32> tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "squeeze_dims", &tf_axis));
  std::set<int> axis_set(tf_axis.begin(), tf_axis.end());

  size_t input_dims = ng_input->get_shape().size();

  ng::Shape input_shape = ng_input->get_shape();
  std::vector<int> dims;

  if (axis_set.size() == 0) {
    for (size_t i = 0; i < input_dims; i++) {
      if (input_shape[i] > 1) {
        dims.push_back(input_shape[i]);
      }
    }
  } else {
    for (size_t i = 0; i < input_dims; i++) {
      bool skip = false;
      if (axis_set.find(i) != axis_set.end()) {
        if (input_shape[i] == 1) {
          skip = true;
        } else {
          throw errors::InvalidArgument(
              "Tried to explicitly squeeze "
              "dimension ",
              i, " but dimension was not 1: ", input_shape[i]);
        }
      }
      if (!skip) {
        dims.push_back(input_shape[i]);
      }
    }
  }

  ng::Shape output_shape(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    output_shape[i] = dims[i];
  }

  ng::AxisVector ng_axis_order(ng_input->get_shape().size());
  std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

  SaveNgOp(ng_op_map, op->name(),
           make_shared<ng::op::Reshape>(ng_input, ng_axis_order, output_shape));
  return Status::OK();
}

static Status TranslateStridedSliceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  // TODO refactor StrideSlice with Slice op
  shared_ptr<ng::Node> ng_input, ng_begin, ng_size, ng_stride;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, &ng_input, &ng_begin, &ng_size, &ng_stride));

  int tf_shrink_axis_mask;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "shrink_axis_mask", &tf_shrink_axis_mask));

  std::vector<int64> lower_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &lower_vec));

  std::vector<int64> end_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &end_vec));

  std::vector<int64> stride_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 3, static_input_map, &stride_vec));

  NGRAPH_VLOG(3) << "Begin input for StridedSlice: " << ng::join(lower_vec);
  NGRAPH_VLOG(3) << "End input for StridedSlice: " << ng::join(end_vec);

  auto& input_shape = ng_input->get_shape();
  NGRAPH_VLOG(3) << "Input shape for StridedSlice: " << ng::join(input_shape);

  if (lower_vec.size() == end_vec.size() && end_vec.size() == 1) {
    for (size_t i = end_vec.size(); i < input_shape.size(); ++i) {
      lower_vec.push_back(0);
      end_vec.push_back(0);
    }
  }
  NGRAPH_VLOG(3) << "extended Begin input for StridedSlice: "
                 << ng::join(lower_vec);
  NGRAPH_VLOG(3) << "extended End input for StridedSlice: "
                 << ng::join(end_vec);

  if (std::any_of(lower_vec.begin(), lower_vec.end(),
                  [](int i) { return i < 0; })) {
    std::transform(lower_vec.begin(), lower_vec.end(), input_shape.begin(),
                   lower_vec.begin(), [](int first, int second) {
                     if (first < 0) {
                       return second + first;
                     } else {
                       return first;
                     }
                   });
  }
  if (std::any_of(end_vec.begin(), end_vec.end(),
                  [](int i) { return i <= 0; })) {
    std::transform(end_vec.begin(), end_vec.end(), input_shape.begin(),
                   end_vec.begin(), [](int first, int second) {
                     if (first < 0) {
                       return second + first;
                     } else if (first == 0) {
                       return second;
                     } else {
                       return first;
                     }
                   });
    NGRAPH_VLOG(3) << "Transform end input for StridedSlice: "
                   << ng::join(end_vec);
  }

  for (size_t i = stride_vec.size(); i < end_vec.size(); ++i) {
    stride_vec.push_back(1);
  }
  NGRAPH_VLOG(3) << "stride input for StridedSlice: " << ng::join(stride_vec);

  std::vector<size_t> l(lower_vec.begin(), lower_vec.end());
  std::vector<size_t> u(end_vec.begin(), end_vec.end());
  std::vector<size_t> s(stride_vec.begin(), stride_vec.end());

  std::shared_ptr<ng::Node> ng_strided_slice =
      make_shared<ng::op::Slice>(ng_input, l, u, s);

  NGRAPH_VLOG(3) << " NG Lower Vector " << ng::join(lower_vec);
  NGRAPH_VLOG(3) << " NG End Vector " << ng::join(end_vec);
  NGRAPH_VLOG(3) << " NG Stride Vector " << ng::join(stride_vec);

  vector<size_t> output_shape;
  if (tf_shrink_axis_mask) {
    int64 shrink_axis_mask = tf_shrink_axis_mask;
    vector<size_t> output_shape;

    for (int i = 0; i < lower_vec.size(); i++) {
      if ((shrink_axis_mask & 1) != 1) {
        output_shape.push_back(end_vec[i] - lower_vec[i]);
      }
      shrink_axis_mask >>= 1;
    }

    NGRAPH_VLOG(3) << "Shrink axis mask " << tf_shrink_axis_mask;

    ng::Shape ng_final_shape(output_shape);
    ng::AxisVector ng_axis_order(input_shape.size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

    NGRAPH_VLOG(3) << " Output  shape " << ng::join(output_shape);
    NGRAPH_VLOG(3) << " NG  axis order " << ng::join(ng_axis_order);

    ng_strided_slice = make_shared<ng::op::Reshape>(
        ng_strided_slice, ng_axis_order, ng_final_shape);
  }

  SaveNgOp(ng_op_map, op->name(), ng_strided_slice);
  return Status::OK();
}

static Status TranslateSumOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_axes_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_axes_op));

  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> sum_axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &sum_axes));

  ng::Shape input_shape = ng_input->get_shape();
  size_t input_rank = input_shape.size();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(sum_axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(sum_axes.size());
  std::transform(
      sum_axes.begin(), sum_axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  ng::AxisSet ng_reduction_axes(ng_reduction_axes_vect);

  std::shared_ptr<ng::Node> ng_sum =
      make_shared<ng::op::Sum>(ng_input, ng_reduction_axes);

  // If keep_dims is specified we need to reshape to put back the reduced
  // axes, with length 1.
  if (tf_keep_dims) {
    ng::Shape ng_result_shape_with_keep(input_rank);

    for (size_t i = 0; i < input_rank; i++) {
      ng_result_shape_with_keep[i] =
          ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
    }

    ng::AxisVector ng_axis_order(ng_sum->get_shape().size());
    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);

    ng_sum = make_shared<ng::op::Reshape>(ng_sum, ng_axis_order,
                                          ng_result_shape_with_keep);
  }

  SaveNgOp(ng_op_map, op->name(), ng_sum);
  return Status::OK();
}

// Computes the gradient for tanh of 'x' w.r.t its input
// grad = dy * (1 - y * y)
// where y = tanh(x) and dy is the corresponding input gradient
static Status TranslateTanhGradOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input, ng_delta;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input, &ng_delta));

  auto ng_sq = std::make_shared<ng::op::Multiply>(ng_input, ng_input);
  ng::Shape input_shape = ng_input->get_shape();
  std::vector<std::string> const_values(ng::shape_size(input_shape), "1");
  auto ng_const = make_shared<ng::op::Constant>(ng_input->get_element_type(),
                                                input_shape, const_values);
  auto ng_sub = make_shared<ng::op::Subtract>(ng_const, ng_sq);
  auto ng_result = make_shared<ng::op::Multiply>(ng_delta, ng_sub);

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
  for (int i = 0; i < ng_input_shape.size(); i++) {
    if (multiples[i] == 0) {
      is_empty = true;
    }
    output_shape[i] = ng_input_shape[i] * multiples[i];
  }
  if (is_empty) {
    SaveNgOp(ng_op_map, op->name(),
             make_shared<ngraph::op::Constant>(
                 ng_input->get_element_type(), output_shape,
                 std::vector<std::string>(ng::shape_size(output_shape), "0")));
  } else {
    for (int i = 0; i < ng_input_shape.size(); i++) {
      if (multiples[i] < 0) {
        return errors::InvalidArgument("Expected multiples[", i,
                                       "] >= 0, but got ", multiples[i]);
      }
      vector<shared_ptr<ng::Node>> tmp_tensors;
      for (int k = 0; k < multiples[i]; k++) {
        tmp_tensors.push_back(ng_output);
      }
      auto ng_concat = make_shared<ngraph::op::Concat>(tmp_tensors, i);
      ng_output = ng_concat;
    }
    SaveNgOp(ng_op_map, op->name(), ng_output);
  }
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

  ng::AxisVector ng_axis_order;
  ng_axis_order.reserve(permutation.size());

  NGRAPH_VLOG(3) << ng::join(permutation);

  for (auto i : permutation) {
    ng_axis_order.push_back(i);
  }

  NGRAPH_VLOG(3) << ng::join(ng_axis_order);

  SaveNgOp(ng_op_map, op->name(),
           ng::builder::numpy_transpose(ng_input, ng_axis_order));
  return Status::OK();
}

static Status TranslateUnpackOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
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
    if (i != unpack_axis) {
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
    auto slice =
        make_shared<ngraph::op::Slice>(ng_input, lower_bound, upper_bound);
    auto reshaped =
        make_shared<ng::op::Reshape>(slice, ng_axis_order, output_shape);
    SaveNgOp(ng_op_map, op->name(), reshaped);
  }
  return Status::OK();
}

static Status TranslateZerosLikeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  shared_ptr<ng::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, &ng_input));

  ng::Shape input_shape = ng_input->get_shape();
  std::vector<std::string> const_values(ng::shape_size(input_shape), "0");
  auto ng_result = make_shared<ng::op::Constant>(ng_input->get_element_type(),
                                                 input_shape, const_values);

  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

const static std::map<
    const string,
    const function<Status(const Node*, const std::vector<const Tensor*>&,
                          Builder::OpMap&)>>
    TRANSLATE_OP_MAP{
        {"Abs", TranslateUnaryOp<ngraph::op::Abs>},
        {"Add", TranslateBinaryOp<ngraph::op::Add>},
        {"AddN", TranslateAddNOp},
        {"Any", TranslateAnyOp},
        {"All", TranslateAllOp},
        {"ArgMax", TranslateArgMaxOp},
        {"ArgMin", TranslateArgMinOp},
        {"AvgPool", TranslateAvgPoolOp},
        {"AvgPoolGrad", TranslateAvgPoolGradOp},
        {"BatchMatMul", TranslateBatchMatMulOp},
        {"BiasAdd", TranslateBiasAddOp},
        {"BiasAddGrad", TranslateBiasAddGradOp},
        {"Cast", TranslateCastOp},
        {"ConcatV2", TranslateConcatV2Op},
        {"Const", TranslateConstOp},
        {"Conv2D", TranslateConv2DOp},
        {"Conv2DBackpropFilter", TranslateConv2DBackpropFilterOp},
        {"Conv2DBackpropInput", TranslateConv2DBackpropInputOp},
        {"DepthToSpace", TranslateDepthToSpaceOp},
        {"DepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
        {"Dequantize", TranslateDequantizeOp},
        {"Equal", TranslateBinaryOp<ngraph::op::Equal>},
        {"Exp", TranslateUnaryOp<ngraph::op::Exp>},
        {"ExpandDims", TranslateExpandDimsOp},
        {"Fill", TranslateFillOp},
        {"Floor", TranslateUnaryOp<ngraph::op::Floor>},
        {"FloorDiv", TranslateFloorDivOp},
        {"FloorMod", TranslateFloorModOp},
        {"FusedBatchNorm", TranslateFusedBatchNormOp},
        {"FusedBatchNormGrad", TranslateFusedBatchNormGradOp},
        {"Greater", TranslateBinaryOp<ngraph::op::Greater>},
        {"GreaterEqual", TranslateBinaryOp<ngraph::op::GreaterEq>},
        {"HorovodAllreduce", TranslateAllreduceOp},
        {"Identity", TranslateIdentityOp},
        {"L2Loss", TranslateL2LossOp},
        {"Less", TranslateBinaryOp<ngraph::op::Less>},
        {"LessEqual", TranslateBinaryOp<ngraph::op::LessEq>},
        {"Log", TranslateUnaryOp<ngraph::op::Log>},
        {"LogicalAnd", TranslateBinaryOp<ngraph::op::And>},
        {"LogicalNot", TranslateUnaryOp<ngraph::op::Not>},
        {"LogicalOr", TranslateBinaryOp<ngraph::op::Or>},
        {"MatMul", TranslateMatMulOp},
        {"Max", TranslateMaxOp},
        {"Maximum", TranslateBinaryOp<ngraph::op::Maximum>},
        {"MaxPool", TranslateMaxPoolOp},
        {"MaxPoolGrad", TranslateMaxPoolGradOp},
        {"Mean", TranslateMeanOp},
        {"Min", TranslateMinOp},
        {"Minimum", TranslateBinaryOp<ngraph::op::Minimum>},
        {"Mul", TranslateBinaryOp<ngraph::op::Multiply>},
        {"Neg", TranslateUnaryOp<ngraph::op::Negative>},
        // Do nothing! NoOps sometimes get placed on nGraph for bureaucratic
        // reasons, but they have no data flow inputs or outputs.
        {"NoOp", [](const Node*, const std::vector<const Tensor*>&,
                    Builder::OpMap&) { return Status::OK(); }},
        {"Pack", TranslatePackOp},
        {"Pad", TranslatePadOp},
        {"Pow", TranslateBinaryOp<ngraph::op::Power>},
        // PreventGradient is just Identity in data-flow terms, so reuse that.
        {"PreventGradient", TranslateIdentityOp},
        {"Prod", TranslateProdOp},
        {"QuantizeAndDequantizeV2", TranslateQuantizeAndDequantizeV2Op},
        {"QuantizedMaxPool", TranslateQuantizedMaxPoolOp},
        {"QuantizeV2", TranslateQuantizeV2Op},
        {"RealDiv", TranslateBinaryOp<ngraph::op::Divide>},
        {"Reciprocal", TranslateReciprocalOp},
        {"Relu", TranslateReluOp},
        {"Relu6", TranslateRelu6Op},
        {"ReluGrad", TranslateReluGradOp},
        {"Reshape", TranslateReshapeOp},
        {"Rsqrt", TranslateRsqrtOp},
        {"Shape", TranslateShapeOp},
        {"Sigmoid", TranslateSigmoidOp},
        {"SigmoidGrad", TranslateSigmoidGradOp},
        {"Size", TranslateSizeOp},
        {"Sign", TranslateUnaryOp<ngraph::op::Sign>},
        {"Slice", TranslateSliceOp},
        {"Snapshot", TranslateSnapshotOp},
        {"Softmax", TranslateSoftmaxOp},
        {"SpaceToDepth", TranslateSpaceToDepthOp},
        {"SparseSoftmaxCrossEntropyWithLogits",
         TranslateSparseSoftmaxCrossEntropyWithLogitsOp},
        {"Split", TranslateSplitOp},
        {"SplitV", TranslateSplitVOp},
        {"Sqrt", TranslateUnaryOp<ngraph::op::Sqrt>},
        {"Square", TranslateSquareOp},
        {"SquaredDifference", TranslateSquaredDifferenceOp},
        {"Squeeze", TranslateSqueezeOp},
        {"StridedSlice", TranslateStridedSliceOp},
        {"Sub", TranslateBinaryOp<ngraph::op::Subtract>},
        {"Sum", TranslateSumOp},
        {"Tanh", TranslateUnaryOp<ngraph::op::Tanh>},
        {"TanhGrad", TranslateTanhGradOp},
        {"Tile", TranslateTileOp},
        {"Transpose", TranslateTransposeOp},
        {"Unpack", TranslateUnpackOp},
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
  GetReversePostOrder(*input_graph, &ordered);

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

    if (n->type_string() == "_Arg") {
      tf_params.push_back(n);
    } else if (n->type_string() == "_Retval") {
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

    auto ng_param = make_shared<ng::op::Parameter>(ng_et, ng_shape);
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
