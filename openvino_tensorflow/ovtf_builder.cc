/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef _WIN32
#define _GNU_SOURCE
#include <dlfcn.h>
#endif

#include <memory>
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"

#include "openvino/frontend/tensorflow/extension/conversion.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"

#include "api.h"
#include "logging/ovtf_log.h"
#include "openvino_tensorflow/backend_manager.h"
#include "openvino_tensorflow/default_opset.h"
#include "openvino_tensorflow/layout_conversions.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_builder.h"
#include "openvino_tensorflow/ovtf_utils.h"
#include "openvino_tensorflow/pass/transpose_sinking.h"

#include "openvino_tensorflow/tf_conversion_extensions/src/conversion_extensions.hpp"

using tensorflow::int32;
using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

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

// Check to make sure the axis dimension for reduction are in within range.
// Returns error if axis is out of range. Otherwise returns Status::OK().
static Status CheckAxisDimInRange(std::vector<int64> axes, size_t rank) {
  for (auto i : axes) {
    if (i < (int)-rank || i >= (int)rank) {
      return errors::InvalidArgument("Axis Dimension is out of range. Got ", i,
                                     ", should be in range [-", rank, ", ",
                                     rank, ")");
    }
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
//    Builder::OpMap& ng_op_map        - The TF-to-OV op map.
//    std::string op_name              - Name of the op.
//
//    ov::Output<ov::Node> output_node - ov::Node to store
//

static void SaveNgOp(Builder::OpMap& ng_op_map, const std::string& op_name,
                     ov::Output<ov::Node> output_node) {
  // no need to try-catch, map[key] will create vector object
  // if not exists
  ng_op_map[op_name].push_back(output_node);
}

void Builder::SetTracingInfo(const std::string& op_name,
                             const ov::Output<ov::Node> ng_node) {
  auto node = ng_node.get_node_shared_ptr();
  node->set_friendly_name(op_name + "/" + node->get_name());
  // node->add_provenance_tag(op_name);
  if (api::IsLoggingPlacement()) {
    cout << "TF_to_NG: " << op_name << " --> " << node << endl;
  }
}

template <class TOpType, class... TArg>
ov::Output<ov::Node> ConstructNgNode(const std::string& op_name,
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
//      ov::Output<ov::Node> ng_input;
//      try {
//        ng_input = ng_op_map.at(tf_input->name());
//      } catch (const std::out_of_range&) {
//        return errors::NotFound(tf_input->name(),
//                                    " is not found in the ng_op_map");
//      }
//
// Into 2 lines:
//
//      ov::Output<ov::node> ng_input;
//      TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, &ng_input))
//
//
//
// Parameters:
//    Builder::OpMap& ng_op_map     - The TF-to-OV op map.
//    Node* op                  - TF op being translated.
//    input_idx                     - index of input
//
//    ov::Output<ov::Node> *result  - ov::Node pointer where result
//                                    will be written
//
//

static Status GetInputNode(const Builder::OpMap& ng_op_map, const Node* op,
                           size_t input_idx, ov::Output<ov::Node>& result) {
  // input op may have resulted in more than one ov::Node (eg. Split)
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
  std::vector<ov::Output<ov::Node>> ng_op;
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
                            size_t index, ov::Output<ov::Node>& result,
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

static Status GetStaticInputNode(
    const Node* op, int64 input_index,
    const std::vector<const Tensor*>& static_input_map, DataType dt,
    ov::Output<ov::Node>& node_) {
  ov::element::Type type;
  TF_RETURN_IF_ERROR(util::TFDataTypeToNGraphElementType(dt, &type));
  switch (dt) {
    case DataType::DT_FLOAT: {
      std::vector<float> vec_float;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_float));
      node_ = ConstructNgNode<opset::Constant>(op->name(), type, ov::Shape{},
                                               vec_float[0]);
    } break;
    case DataType::DT_DOUBLE: {
      std::vector<double> vec_double;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_double));
      node_ = ConstructNgNode<opset::Constant>(op->name(), type, ov::Shape{},
                                               vec_double[0]);
    } break;
    case DataType::DT_INT32: {
      std::vector<int32> vec_i32;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_i32));
      node_ = ConstructNgNode<opset::Constant>(op->name(), type, ov::Shape{},
                                               vec_i32[0]);
    } break;
    case DataType::DT_INT64: {
      std::vector<int64> vec_i64;
      TF_RETURN_IF_ERROR(
          GetStaticInputVector(op, input_index, static_input_map, &vec_i64));
      node_ = ConstructNgNode<opset::Constant>(op->name(), type, ov::Shape{},
                                               vec_i64[0]);
    } break;
    default:
      return errors::Internal("GetStaticInputNode: TF data type ",
                              DataType_Name(dt), " not supported.");
      break;
  }
  return Status::OK();
}

// Taken from: tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc
// Extract values from a Const op to `values`. Returns true if succeeds.
//
// Modified with an extra `VecT` parameter to handle the case where the type
// in the vector does not match TensorFlow's notion of what the C++ type
// should be (e.g. when T is `bool`, we actually need a vector of `char` for
// compatibility with OpenVINO).
template <typename T, typename VecT = T>
static Status ValuesFromConstNode(const NodeDef& node,
                                  TensorShapeProto* const_tensor_shape,
                                  std::vector<VecT>* values) {
  if (node.op() != "Const") {
    return errors::InvalidArgument("Node not a Const");
  }

  if (node.attr().at("dtype").type() != DataTypeToEnum<T>::value) {
    std::stringstream ss;
    ss << "Invalid data type defined for Const. Defined: "
       << node.attr().at("dtype").type();
    return errors::InvalidArgument(ss.str());
  }

  // TensorProto represents the content of the tensor in either <type>_val or
  // tensor_content.
  const TensorProto& tensor = node.attr().at("value").tensor();
  typename checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
      checkpoint::MutableTensorProtoData<T>(const_cast<TensorProto*>(&tensor));

  const TensorShapeProto& shape = tensor.tensor_shape();
  *const_tensor_shape = shape;
  if (!tensor_values->empty() && tensor.has_tensor_shape()) {
    // When tensor_shape is set, theoretically the representation of the data
    // could be compressed. So, before copying values to the returned vector,
    // make sure no compression happens.
    if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values->size()) {
      values->insert(values->end(), tensor_values->begin(),
                     tensor_values->end());
      return Status::OK();
    }
  }

  const auto tensor_content_size = tensor.tensor_content().size();
  CHECK_EQ(0, tensor_content_size % sizeof(VecT))
      << " tensor_content_size (" << tensor_content_size
      << ") is not a multiple of " << sizeof(VecT);

  // If tensor_content_size is zero, we'll have to take the values from
  // int_val, float_val, etc.
  if (tensor_content_size == 0) {
    int64 n_elements = 1;
    for (auto i = 0; i < shape.dim_size(); i++) {
      if (shape.dim(i).size() < 0) {
        return errors::InvalidArgument(
            "Const node has empty tensor and an unknown dimension size");
      }
      n_elements *= shape.dim(i).size();
    }
    values->resize(n_elements);

    // Extract dtype and size of the values provided in proto
    auto& tensor = node.attr().at("value").tensor();
    auto dt = node.attr().at("dtype").type();
    int64 val_size = 0;
    switch (dt) {
      // TODO(amprocte/NGRAPH-2502): there are more element types to support
      // here
      case DT_INT32:
        val_size = tensor.int_val_size();
        break;
      case DT_INT64:
        val_size = tensor.int64_val_size();
        break;
      case DT_FLOAT:
        val_size = tensor.float_val_size();
        break;
      case DT_BOOL:
        val_size = tensor.bool_val_size();
        break;
      case DT_DOUBLE:
        val_size = tensor.double_val_size();
        break;
      default:
        OVTF_VLOG(0) << "Const node has empty tensor and we don't know how to "
                        "handle this element type";
        OVTF_VLOG(0) << node.DebugString();
        OVTF_VLOG(0) << shape.DebugString();
        return errors::Unimplemented("Encountered unknown element type ",
                                     DataType_Name(dt), " on an empty tensor");
    }

    auto val_lastsaved = (T)0;  // cast
    for (auto i = 0; i < n_elements; i++) {
      // Default value set to 0 and typecasted to the dtype of the const op
      auto val_i = (T)0;
      // If no values are specified for the const node, fill all the values with
      // default value or return error based on the TF version
      if (val_size == 0) {
#if (TF_MAJOR_VERSION > 1 && TF_MINOR_VERSION >= 7)
        (*values)[i] = val_i;
#else
        return errors::InvalidArgument("Empty values vector");
#endif
        continue;
      }

      // If the values are same for all the elements or repeating after certain
      // index, then copy the single value or the last occured value for all the
      // remaining indices
      if (i < val_size) {
        switch (dt) {
          // TODO(amprocte/NGRAPH-2502): there are more element types to support
          // here
          case DT_INT32:
            val_i = tensor.int_val()[i];
            break;
          case DT_INT64:
            val_i = tensor.int64_val()[i];
            break;
          case DT_FLOAT:
            val_i = tensor.float_val()[i];
            break;
          case DT_BOOL:
            val_i = tensor.bool_val()[i];
            break;
          case DT_DOUBLE:
            val_i = tensor.double_val()[i];
            break;
          default:
            OVTF_VLOG(0)
                << "Const node has empty tensor and we don't know how to "
                   "handle this element type";
            OVTF_VLOG(0) << node.DebugString();
            OVTF_VLOG(0) << shape.DebugString();
            return errors::Unimplemented("Encountered unknown element type ",
                                         DataType_Name(dt),
                                         " on an empty tensor");
        }
        (*values)[i] = val_i;
        val_lastsaved = val_i;
      } else {
        (*values)[i] = val_lastsaved;
      }
    }
  } else {
    values->resize(tensor_content_size / sizeof(VecT));
    port::CopyToArray(tensor.tensor_content(),
                      reinterpret_cast<char*>(values->data()));
  }

  return Status::OK();
}

template <typename T>
static Status MakeConstOpForParam(const Tensor& tensor, string prov_tag,
                                  ov::element::Type ng_et, ov::Shape ng_shape,
                                  ov::Output<ov::Node>& ng_node) {
  vector<T> const_values;

  TensorDataToVector(tensor, &const_values);
  ng_node =
      ConstructNgNode<opset::Constant>(prov_tag, ng_et, ng_shape, const_values);

  return Status::OK();
}

// Helper for Builder::TranslateGraph ("Const" op)
template <typename T, typename VecT = T>
static Status MakeConstOp(const Node* op, ov::element::Type et,
                          ov::Output<ov::Node>& ng_node) {
  vector<VecT> const_values;
  TensorShapeProto shape_proto;

  TF_RETURN_IF_ERROR(
      ValuesFromConstNode<T, VecT>(op->def(), &shape_proto, &const_values));

  TensorShape const_shape(shape_proto);

  ov::Shape ng_shape;
  TF_RETURN_IF_ERROR(util::TFTensorShapeToNGraphShape(const_shape, &ng_shape));

  ng_node =
      ConstructNgNode<opset::Constant>(op->name(), et, ng_shape, const_values);
  return Status::OK();
}

const Builder::ConstMap& Builder::TF_NGRAPH_CONST_MAP() {
  static const Builder::ConstMap the_map = {
      {DataType::DT_FLOAT, make_pair(MakeConstOp<float>, ov::element::f32)},
      {DataType::DT_DOUBLE, make_pair(MakeConstOp<double>, ov::element::f64)},
      {DataType::DT_INT8, make_pair(MakeConstOp<int8>, ov::element::i8)},
      {DataType::DT_INT16, make_pair(MakeConstOp<int16>, ov::element::i16)},
      {DataType::DT_QINT8, make_pair(MakeConstOp<qint8>, ov::element::i8)},
      {DataType::DT_QUINT8, make_pair(MakeConstOp<quint8>, ov::element::u8)},
      {DataType::DT_QUINT16, make_pair(MakeConstOp<quint16>, ov::element::u16)},
      {DataType::DT_INT32, make_pair(MakeConstOp<int32>, ov::element::i32)},
      {DataType::DT_INT64, make_pair(MakeConstOp<int64>, ov::element::i64)},
      {DataType::DT_UINT8, make_pair(MakeConstOp<uint8>, ov::element::u8)},
      {DataType::DT_UINT16, make_pair(MakeConstOp<uint16>, ov::element::u16)},
      {DataType::DT_BOOL,
       make_pair(MakeConstOp<bool, char>, ov::element::boolean)}};
  return the_map;
}

// Helper function to translate a unary op.
//
// Parameters:
//
//    Node* op                   - TF op being translated. Must have one input.
//    const std::vector<const Tensor*>& static_input_map
//                               - the static input map
//    Builder::OpMap& ng_op_map  - The TF-to-OV op map.
//
//    std::function<ov::Output<ov::Node>(ov::Output<ov::Node>>
//      create_unary_op           - Function to construct the graph implementing
//                                 the unary op, given the input to the unop
//                                 as an argument.
//
// Example usage:
//
//  if (n->type_string == "Square") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp(n, static_input_map, ng_op_map,
//                       [] (ov::Output<ov::Node> n) {
//                           return
//                           (ov::Output<opset::Multiply>(n,n));
//                       });
//  }
static Status TranslateUnaryOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map,
    std::function<ov::Output<ov::Node>(ov::Output<ov::Node>)> create_unary_op) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
  auto ng_node = create_unary_op(ng_input);
  if (ng_node != ng_input) {
    Builder::SetTracingInfo(op->name(), ng_node);
  }
  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

// Helper function to translate a unary op in cases where there is a one-to-one
// mapping from TensorFlow ops to OpenVINO ops.
//
// Example usage:
//
//  if (n->type_string == "Abs") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp<ov::op::Abs>(n, static_input_map,
//    ng_op_map));
//  }
//
template <typename T>
static Status TranslateUnaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(op, static_input_map, ng_op_map,
                          [&op](ov::Output<ov::Node> n) {
                            return ConstructNgNode<T>(op->name(), n);
                          });
}

// Helper function to translate a binary op
// Parameters:
//
//    Node* op               - TF op being translated. Must have only two
//    inputs.
//    const std::vector<const Tensor*>& static_input_map - the static input map
//    Builder::OpMap& ng_op_map  - The TF-to-OV op map.
//    std::function<ov::Output<ov::Node>(ov::Output<ov::Node>,
//    ov::Output<ov::Node>)>
//    create_binary_op           - Function to construct the graph implementing
//                                 the binary op, given the 2 ng_inputs to the
//                                 binaryop
// Example Usage:
//
// if (op->type_string() == "SquaredDifference") {
//      TF_RETURN_IF_ERROR(TranslateBinaryOp(op, ng_op_map,
//         [](ov::Output<ov::Node> ng_input1, ov::Output<ov::Node>
//         ng_input2) {
//           auto ng_diff = ov::Output<opset::Subtract>(input1,
//           input2);
//           return ov::Output<opset::Multiply>(ng_diff,ng_diff);
//         }));
//    }
//

static Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map,
    std::function<ov::Output<ov::Node>(ov::Output<ov::Node>&,
                                       ov::Output<ov::Node>&)>
        create_binary_op) {
  ov::Output<ov::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs));
  auto ng_node = create_binary_op(ng_lhs, ng_rhs);
  if (ng_node != ng_lhs && ng_node != ng_rhs) {
    Builder::SetTracingInfo(op->name(), ng_node);
  }
  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

// Helper function to translate a binary op in cases where there is a one-to-one
// mapping from TensorFlow ops to OpenVINO ops.
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
      [&op](ov::Output<ov::Node>& ng_lhs, ov::Output<ov::Node>& ng_rhs) {
        return ConstructNgNode<T>(op->name(), ng_lhs, ng_rhs);
      });
}

static Status TranslateAddNOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  std::vector<ov::Output<ov::Node>> ng_arg_vec(op->num_inputs());

  for (int inp_idx = 0; inp_idx < op->num_inputs(); inp_idx++)
    TF_RETURN_IF_ERROR(
        GetInputNode(ng_op_map, op, inp_idx, ng_arg_vec[inp_idx]));
  auto ng_addn = std::accumulate(
      std::next(ng_arg_vec.begin()), ng_arg_vec.end(), ng_arg_vec.at(0),
      [&op](ov::Output<ov::Node> a, ov::Output<ov::Node> b) {
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
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  std::vector<int64> tf_dim;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &tf_dim));

  size_t input_rank = ng_input.get_partial_shape().rank().get_length();

  if (tf_dim.size() != 1) {
    return errors::InvalidArgument(
        "ArgMax Op: dimension must be scalar, operates on a single axis");
  }

  // If input dimension is negative, make it positive
  if (tf_dim[0] < 0) {
    OVTF_VLOG(3) << "Input dimension is negative, make it positive "
                 << tf_dim[0];
    tf_dim[0] = (int64)input_rank + tf_dim[0];
  }
  OVTF_VLOG(3) << "Axis along which to compute " << tf_dim[0];
  size_t k_axis = tf_dim[0];

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "output_type", &dtype));

  ov::element::Type ng_et;
  TF_RETURN_IF_ERROR(util::TFDataTypeToNGraphElementType(dtype, &ng_et));

  auto ng_k = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{}, std::vector<int64>({1}));

  std::string sort = "none";
  auto ng_topk =
      std::make_shared<opset::TopK>(ng_input, ng_k, k_axis, mode, sort, ng_et);
  auto ng_indices = ng_topk->output(1);
  int axis = ng_topk->get_axis();
  auto axis_to_remove = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{1}, std::vector<int64>({axis}));
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

template <unsigned int N>
static Status TranslateAvgPoolOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if ((tf_data_format != "NHWC") && (tf_data_format != "NCHW") &&
      (tf_data_format != "NDHWC")) {
    return errors::InvalidArgument(
        "AvgPool data format is none of NHWC, NCHW, or NDHWC");
  }

  bool is_nhwc = (tf_data_format == "NHWC") || (tf_data_format == "NDHWC");

  OVTF_VLOG(3) << ngraph::join(tf_strides);
  OVTF_VLOG(3) << ngraph::join(tf_ksize);
  OVTF_VLOG(3) << tf_padding_type;
  OVTF_VLOG(3) << tf_data_format;

  ov::Strides ng_strides(N);
  ov::Shape ng_kernel_shape(N);

  NHWCtoHW(is_nhwc, tf_strides, ng_strides);
  NHWCtoHW(is_nhwc, tf_ksize, ng_kernel_shape);
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  OVTF_VLOG(3) << "ng_strides: " << ngraph::join(ng_strides);
  OVTF_VLOG(3) << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

  ov::Shape ng_padding_below, ng_padding_above;

  ov::op::PadType auto_pad_type;
  if (tf_padding_type == "SAME")
    auto_pad_type = ov::op::PadType::SAME_UPPER;
  else if (tf_padding_type == "VALID")
    auto_pad_type = ov::op::PadType::VALID;

  // since we are using auto_pad, all the explicit padding arguments will be
  // ignored
  ov::Output<ov::Node> ng_avgpool = ConstructNgNode<opset::AvgPool>(
      op->name(), ng_input, ng_strides, ng_padding_below, ng_padding_above,
      ng_kernel_shape, true, ov::op::RoundingType::FLOOR, auto_pad_type);

  NCHWtoNHWC(op->name(), is_nhwc, ng_avgpool);
  OVTF_VLOG(3) << "avgpool outshape: {" << ngraph::join(ng_avgpool.get_shape())
               << "}";

  SaveNgOp(ng_op_map, op->name(), ng_avgpool);
  return Status::OK();
}

static Status TranslateBatchMatMulOp(const Node* op,
                                     const std::vector<const Tensor*>&,
                                     Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_x, ng_y;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_x, ng_y));

  bool adj_x, adj_y;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "adj_x", &adj_x));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "adj_y", &adj_y));

  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::MatMul>(
                                      op->name(), ng_x, ng_y, adj_x, adj_y));
  return Status::OK();
}

static Status TranslateBatchNDAndSpaceNDOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_block_shape, ng_crops,
      ng_block_shape_unused, ng_crops_unused;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_input, ng_block_shape_unused, ng_crops));

  // ng_crops should be of shape N=[ng_input.get_shape()).size()]
  // But TF's ng_crops input is limited only to the spatial dimensions (neither
  // batch nor innermost),
  // which would mean ngraph inputs have missing ng_crops[0] and ng_crops[N].
  // Hence, pad ng_crops with zeros at both ends
  std::vector<int> tf_block_shape;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 1, static_input_map, &tf_block_shape));

  auto N = (int)ng_input.get_partial_shape().rank().get_length();
  auto M = (int)tf_block_shape.size();

  ng_block_shape = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{static_cast<unsigned long>(M)},
      tf_block_shape);

  // return with input if rank < 2 as ngraph's impl doesn't support it
  if (N < 2) {
    SaveNgOp(ng_op_map, op->name(), ng_input);
    return Status::OK();
  }

  auto crops = ConstructNgNode<opset::Pad>(
      op->name(), ng_crops,
      make_shared<opset::Constant>(ng_crops.get_element_type(), ov::Shape{2},
                                   std::vector<int>{1, 0}),
      make_shared<opset::Constant>(ng_crops.get_element_type(), ov::Shape{2},
                                   std::vector<int>{N - M - 1, 0}),
      ov::op::PadMode::CONSTANT);

  // Padding needs to be done for block_shape as done for crops above but with
  // value=1
  auto block_shape_native = ConstructNgNode<opset::Pad>(
      op->name(), ng_block_shape,
      make_shared<opset::Constant>(ng_block_shape.get_element_type(),
                                   ov::Shape{1}, std::vector<int>{1}),
      make_shared<opset::Constant>(ng_block_shape.get_element_type(),
                                   ov::Shape{1}, std::vector<int>{N - M - 1}),
      make_shared<opset::Constant>(ng_block_shape.get_element_type(),
                                   ov::Shape{}, 1),
      ov::op::PadMode::CONSTANT);
  // block_shape, crops_begin and crops_end inputs must have same element type
  ov::element::Type crops_dtype = crops.get_element_type();
  auto block_shape = ConstructNgNode<opset::Convert>(
      op->name(), block_shape_native, crops_dtype);

  auto target_axis =
      make_shared<opset::Constant>(ov::element::i64, ov::Shape{}, 1);
  // split into two 1-D vectors crops_begin and crops_end along axis 1
  auto crops_split =
      ConstructNgNode<opset::Split>(op->name(), crops, target_axis, 2);

  // crops: [[0, 1], [1, 2], ...]
  // crops_split: [[[0], [1]], [[1], [2]], ...]
  // crops_begin: [0, 1, ...], crops_end: [1, 2, ...]
  auto axes = make_shared<opset::Constant>(ov::element::i32, ov::Shape{}, -1);
  auto crops_begin = ConstructNgNode<opset::Squeeze>(
      op->name(), crops_split.get_node()->outputs()[0], axes);
  auto crops_end = ConstructNgNode<opset::Squeeze>(
      op->name(), crops_split.get_node()->outputs()[1], axes);

  if (op->type_string() == "BatchToSpaceND") {
    auto ng_batch_to_space_nd = ConstructNgNode<opset::BatchToSpace>(
        op->name(), ng_input, block_shape, crops_begin, crops_end);
    SaveNgOp(ng_op_map, op->name(), ng_batch_to_space_nd);
  } else if (op->type_string() == "SpaceToBatchND") {
    auto ng_space_to_batch_nd = ConstructNgNode<opset::SpaceToBatch>(
        op->name(), ng_input, block_shape, crops_begin, crops_end);
    SaveNgOp(ng_op_map, op->name(), ng_space_to_batch_nd);
  } else {
    return errors::Unknown("Unknown Op Name: ", op->name());
  }

  return Status::OK();
}

static Status TranslateBiasAddOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_bias;
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

  auto ng_bias_rank = ng_bias.get_partial_shape().rank().get_length();
  if (ng_bias_rank != 1) {
    return errors::InvalidArgument(
        "Bias argument to BiasAdd does not have one dimension");
  }

  // We'll choose reshape over broadcast
  // Reshape the bias to (1, C, 1, ...) if input is channels-first.
  ov::Output<ov::Node> ng_bias_reshaped = ng_bias;
  if (tf_data_format == "NCHW") {
    auto ng_input_shape = ng_input.get_partial_shape();
    auto ng_input_rank = ng_input_shape.rank().get_length();
    auto channel_dim = ng_input_shape[1].get_length();
    std::vector<int64> target_shape(ng_input_rank);
    for (int64_t i = 0; i < ng_input_rank; i++) {
      if (i == 1) {
        target_shape[i] = channel_dim;
      } else {
        target_shape[i] = 1;
      }
    }
    auto target_shape_node = make_shared<opset::Constant>(
        ov::element::i64, ov::Shape{static_cast<unsigned long>(ng_input_rank)},
        target_shape);
    ng_bias_reshaped = ConstructNgNode<opset::Reshape>(
        op->name(), ng_bias, target_shape_node, false);
  }

  ov::Output<ov::Node> ng_add =
      ConstructNgNode<opset::Add>(op->name(), ng_input, ng_bias_reshaped);

  SaveNgOp(ng_op_map, op->name(), ng_add);
  return Status::OK();
}

static Status TranslateCastOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "DstT", &dtype));

  ov::element::Type ng_et;
  TF_RETURN_IF_ERROR(util::TFDataTypeToNGraphElementType(dtype, &ng_et));

  auto ng_input_dtype = ng_input.get_element_type();

  if (ng_et == ov::element::boolean && (ng_input_dtype == ov::element::f32 ||
                                        ng_input_dtype == ov::element::f64)) {
    auto zeros = ConstructNgNode<opset::Constant>(op->name(), ng_input_dtype,
                                                  ov::Shape{}, 0);
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<opset::NotEqual>(op->name(), ng_input, zeros));
    return Status::OK();
  }

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
    ov::Output<ov::Node> ng_first_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_first_arg));

    concat_axis += int64(ng_first_arg.get_partial_shape().rank().get_length());
  }

  ov::OutputVector ng_args;
  for (int i = 0; i < op->num_inputs() - 1; i++) {
    ov::Output<ov::Node> ng_arg;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, ng_arg));
    bool valid_input = true;

    if (ng_arg.get_partial_shape().is_static()) {
      auto inp_shape = ng_arg.get_shape();
      for (auto dim : inp_shape) {
        if (dim == 0) {
          valid_input = false;
          break;
        }
      }
    }
    if (valid_input) {
      ng_args.push_back(ng_arg);
    }
  }

  if (ng_args.empty()) {
    int concat_axis_out_dim_value = 0;
    ov::Output<ov::Node> ng_arg;
    ov::Shape inp_shape;
    for (int i = 0; i < op->num_inputs() - 1; i++) {
      TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, ng_arg));
      inp_shape = ng_arg.get_shape();
      concat_axis_out_dim_value += inp_shape[concat_axis];
    }
    CHECK(concat_axis >= 0);
    inp_shape[concat_axis] = concat_axis_out_dim_value;
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<opset::Constant>(
                 op->name(), ng_arg.get_element_type(), inp_shape, 0));
  } else {
    SaveNgOp(ng_op_map, op->name(),
             ConstructNgNode<opset::Concat>(op->name(), ng_args,
                                            size_t(concat_axis)));
  }
  return Status::OK();
}

static Status TranslateConstOp(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dtype", &dtype));

  ov::Output<ov::Node> ng_node;

  // For some reason the following do not work (no specialization of
  // tensorflow::checkpoint::SavedTypeTraits...)
  // case DataType::DT_UINT32:
  //   TF_RETURN_IF_ERROR(MakeConstOp<uint32>(op, ov::element::u32,
  //   &ng_node));
  //   break;
  // case DataType::DT_UINT64:
  //   TF_RETURN_IF_ERROR(MakeConstOp<uint64>(op, ov::element::u64,
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
  ov::Output<ov::Node> ng_input, ng_filter;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter));

  std::vector<int32> tf_strides;
  std::vector<int32> tf_dilations;
  std::vector<int32> tf_paddings;
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

  OVTF_VLOG(3) << ngraph::join(tf_strides);
  OVTF_VLOG(3) << ngraph::join(tf_dilations);
  OVTF_VLOG(3) << tf_padding_type;
  OVTF_VLOG(3) << tf_data_format;

  ov::Strides ng_strides(2);
  ov::Strides ng_dilations(2);
  ov::Shape ng_image_shape(2);
  ov::Shape ng_kernel_shape(2);

  NHWCtoHW(is_nhwc, tf_strides, ng_strides);
  if (ng_input.get_partial_shape().is_static()) {
    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
  }
  NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);

  OVTF_VLOG(3) << "ng_strides: " << ngraph::join(ng_strides);
  OVTF_VLOG(3) << "ng_dilations: " << ngraph::join(ng_dilations);
  OVTF_VLOG(3) << "ng_image_shape: " << ngraph::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  OVTF_VLOG(3) << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

  ov::Output<ov::Node> ng_conv;
  ov::op::PadType pad_type;
  ov::CoordinateDiff ng_padding_below;
  ov::CoordinateDiff ng_padding_above;

  if (tf_padding_type == "EXPLICIT") {
    TF_RETURN_IF_ERROR(
        GetNodeAttr(op->attrs(), "explicit_paddings", &tf_paddings));
    if (is_nhwc) {
      ng_padding_below.push_back(tf_paddings[2]);
      ng_padding_below.push_back(tf_paddings[4]);
      ng_padding_above.push_back(tf_paddings[3]);
      ng_padding_above.push_back(tf_paddings[5]);
    } else {
      ng_padding_below.push_back(tf_paddings[4]);
      ng_padding_below.push_back(tf_paddings[6]);
      ng_padding_above.push_back(tf_paddings[5]);
      ng_padding_above.push_back(tf_paddings[7]);
    }
    OVTF_VLOG(3) << " ========== EXPLICIT Padding ========== ";
    OVTF_VLOG(3) << "ng_padding_below: " << ngraph::join(ng_padding_below);
    OVTF_VLOG(3) << "ng_padding_above: " << ngraph::join(ng_padding_above);
    ng_conv = ConstructNgNode<opset::Convolution>(
        op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
        ng_padding_above, ng_dilations);
  } else if (tf_padding_type == "VALID") {
    ng_padding_below.assign(ng_image_shape.size(), 0);
    ng_padding_above.assign(ng_image_shape.size(), 0);
    ng_conv = ConstructNgNode<opset::Convolution>(
        op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
        ng_padding_above, ng_dilations);
  } else if (tf_padding_type == "SAME") {
    if (ng_input.get_partial_shape().is_static()) {
      OVTF_VLOG(3) << "========== SAME Padding - Static Shape ========== ";
      ov::Shape img_shape = {0, 0};
      img_shape.insert(img_shape.end(), ng_image_shape.begin(),
                       ng_image_shape.end());
      ov::infer_auto_padding(img_shape, ng_kernel_shape, ng_strides,
                             ng_dilations, ov::op::PadType::SAME_UPPER,
                             ng_padding_above, ng_padding_below);
      ng_conv = ConstructNgNode<opset::Convolution>(
          op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
          ng_padding_above, ng_dilations);
    } else {
      OVTF_VLOG(3) << "========== SAME Padding - Dynamic Shape ========== ";
      pad_type = ov::op::PadType::SAME_UPPER;
      ng_conv = ConstructNgNode<opset::Convolution>(
          op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
          ng_padding_above, ng_dilations, pad_type);
    }
  }

  NCHWtoNHWC(op->name(), is_nhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateConv2DBackpropInputOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_filter, ng_out_backprop, ng_unused;
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

  OVTF_VLOG(3) << ngraph::join(tf_strides);
  OVTF_VLOG(3) << ngraph::join(tf_dilations);
  OVTF_VLOG(3) << tf_padding_type;
  OVTF_VLOG(3) << tf_data_format;

  ov::Strides ng_strides(2);
  ov::Strides ng_dilations(2);
  ov::Shape ng_image_shape(2);
  ov::Shape ng_kernel_shape(2);
  ov::Shape ng_batch_shape(4);

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

  OVTF_VLOG(3) << "ng_strides: " << ngraph::join(ng_strides);
  OVTF_VLOG(3) << "ng_dilations: " << ngraph::join(ng_dilations);
  OVTF_VLOG(3) << "ng_image_shape: " << ngraph::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  Transpose<3, 2, 0, 1>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  OVTF_VLOG(3) << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

  ov::CoordinateDiff ng_padding_below;
  ov::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  auto ng_output_shape = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{ng_batch_shape.size() - 2},
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
  ov::Output<ov::Node> ng_input, ng_filter;
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

  OVTF_VLOG(3) << ngraph::join(tf_strides);
  OVTF_VLOG(3) << ngraph::join(tf_dilations);
  OVTF_VLOG(3) << tf_padding_type;
  OVTF_VLOG(3) << tf_data_format;

  ov::Strides ng_strides(3);
  ov::Strides ng_dilations(3);
  ov::Shape ng_image_shape(3);
  ov::Shape ng_kernel_shape(3);

  NHWCtoHW(is_ndhwc, tf_strides, ng_strides);
  NHWCtoHW(is_ndhwc, ng_input.get_shape(), ng_image_shape);
  NHWCtoHW(is_ndhwc, tf_dilations, ng_dilations);
  NHWCtoNCHW(op->name(), is_ndhwc, ng_input);

  OVTF_VLOG(3) << "ng_strides: " << ngraph::join(ng_strides);
  OVTF_VLOG(3) << "ng_dilations: " << ngraph::join(ng_dilations);
  OVTF_VLOG(3) << "ng_image_shape: " << ngraph::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  ng_kernel_shape[2] = ng_filter_shape[2];
  Transpose3D<4, 3, 0, 1, 2>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  OVTF_VLOG(3) << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

  ov::CoordinateDiff ng_padding_below;
  ov::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  ov::Output<ov::Node> ng_conv = ConstructNgNode<opset::Convolution>(
      op->name(), ng_input, ng_filter, ng_strides, ng_padding_below,
      ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_ndhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateConv3DBackpropInputV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_filter, ng_out_backprop, ng_unused;
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

  if (tf_data_format != "NDHWC" && tf_data_format != "NCDHW") {
    return errors::InvalidArgument(
        "Conv2DBackpropInput data format is neither NDHWC nor NCDHW: %s",
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

  bool is_ndhwc = (tf_data_format == "NDHWC");

  OVTF_VLOG(3) << ngraph::join(tf_strides);
  OVTF_VLOG(3) << ngraph::join(tf_dilations);
  OVTF_VLOG(3) << tf_padding_type;
  OVTF_VLOG(3) << tf_data_format;

  ov::Strides ng_strides(3);
  ov::Strides ng_dilations(3);
  ov::Shape ng_image_shape(3);
  ov::Shape ng_kernel_shape(3);
  ov::Shape ng_batch_shape(5);

  NHWCtoHW(is_ndhwc, tf_strides, ng_strides);
  NHWCtoHW(is_ndhwc, tf_dilations, ng_dilations);
  NHWCtoHW(is_ndhwc, tf_input_sizes, ng_image_shape);
  NHWCtoNCHW(op->name(), is_ndhwc, ng_out_backprop);
  if (is_ndhwc) {
    ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                      static_cast<unsigned long>(tf_input_sizes[4]),
                      static_cast<unsigned long>(tf_input_sizes[1]),
                      static_cast<unsigned long>(tf_input_sizes[2]),
                      static_cast<unsigned long>(tf_input_sizes[3])};
  } else {
    ng_batch_shape = {static_cast<unsigned long>(tf_input_sizes[0]),
                      static_cast<unsigned long>(tf_input_sizes[1]),
                      static_cast<unsigned long>(tf_input_sizes[2]),
                      static_cast<unsigned long>(tf_input_sizes[3]),
                      static_cast<unsigned long>(tf_input_sizes[4])};
  }

  OVTF_VLOG(3) << "ng_strides: " << ngraph::join(ng_strides);
  OVTF_VLOG(3) << "ng_dilations: " << ngraph::join(ng_dilations);
  OVTF_VLOG(3) << "ng_image_shape: " << ngraph::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];
  ng_kernel_shape[2] = ng_filter_shape[2];
  Transpose3D<4, 3, 0, 1, 2>(ng_filter);
  Builder::SetTracingInfo(op->name(), ng_filter);

  OVTF_VLOG(3) << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

  ov::CoordinateDiff ng_padding_below;
  ov::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  auto ng_output_shape = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{ng_batch_shape.size() - 2},
      vector<size_t>(ng_batch_shape.begin() + 2, ng_batch_shape.end()));

  auto ng_data = ConstructNgNode<opset::ConvolutionBackpropData>(
      op->name(), ng_out_backprop, ng_filter, ng_output_shape, ng_strides,
      ng_padding_below, ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_ndhwc, ng_data);
  SaveNgOp(ng_op_map, op->name(), ng_data);
  return Status::OK();
}

static Status TranslateCropAndResizeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  /// ng_input: [batch, image_height, image_width, depth]
  /// ng_boxes: [num_boxes, 4]; each box is a normalized [0.to 1.] co-ordinate
  /// [y1,
  /// x1, y2, x2]
  /// ng_box_ind: [num_boxes]; i-th ng_box_ind refers to the image to crop and
  /// ranges from 0 to batch
  /// ng_crop_size: [crop_height, crop_width];

  /// for each box b specified in ng_boxes:
  ///  1. crop ng_input[ng_box_ind[b]] w/ co-ordinates in ng_boxes
  ///  2. resize according to method

  ov::Output<ov::Node> ng_input, ng_boxes, ng_box_ind, ng_size;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_input, ng_boxes, ng_box_ind, ng_size));

  string tf_resize_method;
  float tf_extrapolation_value;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "method", &tf_resize_method));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "extrapolation_value", &tf_extrapolation_value));

  auto spatial_shape = ng_input.get_shape();
  auto image_height = spatial_shape[1];
  auto image_width = spatial_shape[2];
  auto image_depth = spatial_shape[3];

  std::vector<float> boxes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &boxes));

  std::vector<int64> box_ind;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &box_ind));

  std::vector<int64> crop_size;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 3, static_input_map, &crop_size));

  ov::OutputVector ng_crop_outputs(box_ind.size());
  if (box_ind.size() == 0) {
    SaveNgOp(
        ng_op_map, op->name(),
        ConstructNgNode<opset::Constant>(
            op->name(), ov::element::f32,
            ov::Shape{0, static_cast<unsigned long>(crop_size.at(0)),
                      static_cast<unsigned long>(crop_size.at(1)), image_depth},
            std::vector<float>({})));
  } else {
    for (int i = 0; i < box_ind.size(); i++) {
      int y1, x1, y2, x2;
      y1 = boxes.at(0 + i * 4) * (image_height - 1);
      x1 = boxes.at(1 + i * 4) * (image_width - 1);
      y2 = boxes.at(2 + i * 4) * (image_height - 1);
      x2 = boxes.at(3 + i * 4) * (image_width - 1);

      int crop_height = std::abs(y2 - y1);
      int crop_width = std::abs(x2 - x1);

      // account for flip crops when y1>y2 or x1>x2 with negative striding
      int stride_height = 1, stride_width = 1;
      if (y1 > y2) {
        y1 = y1 - image_height;
        y2 = y2 - image_height - 2;
        stride_height = -1;
      }
      if (x1 > x2) {
        x1 = x1 - image_height;
        x2 = x2 - image_height - 2;
        stride_width = -1;
      }

      auto begin = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i64, ov::Shape{4},
          std::vector<int64>({static_cast<int64>(box_ind[i]), y1, x1, 0}));
      auto end = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i64, ov::Shape{4},
          std::vector<int64>({static_cast<int64>(box_ind[i]) + 1, y2 + 1,
                              x2 + 1, static_cast<int64>(image_depth + 1)}));
      auto strides = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i64, ov::Shape{4},
          std::vector<int64>({1, stride_height, stride_width, 1}));

      // crop
      auto ng_crop = ConstructNgNode<opset::StridedSlice>(
          op->name(), ng_input, begin, end, strides, std::vector<int64_t>{},
          std::vector<int64_t>{});

      opset::Interpolate::InterpolateAttrs interpolate_attrs;
      // always corner aligned
      interpolate_attrs.coordinate_transformation_mode =
          opset::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;

      // TODO: handle the case when extrapolation value is greatger than 1.0
      // arguments for resizing
      auto ng_spatial_shape = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i32, ov::Shape{2},
          std::vector<int32>{crop_height, crop_width});
      auto ng_input_shape = ConstructNgNode<opset::Convert>(
          op->name(), ng_spatial_shape, ov::element::f32);
      auto ng_crop_size = ConstructNgNode<opset::Convert>(op->name(), ng_size,
                                                          ov::element::f32);
      auto ng_scales = ConstructNgNode<opset::Divide>(op->name(), ng_crop_size,
                                                      ng_input_shape);
      auto ng_axes = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i32, ov::Shape{2}, std::vector<int>({2, 3}));

      if (tf_resize_method == "bilinear") {
        interpolate_attrs.mode = opset::Interpolate::InterpolateMode::LINEAR;
      } else {  // nearest
        interpolate_attrs.mode = opset::Interpolate::InterpolateMode::NEAREST;
      }

      Transpose<0, 3, 1, 2>(ng_crop);
      auto ng_output = ConstructNgNode<opset::Interpolate>(
          op->name(), ng_crop, ng_size, ng_scales, ng_axes, interpolate_attrs);
      Transpose<0, 2, 3, 1>(ng_output);
      ng_crop_outputs.at(i) = ng_output;
    }

    auto ng_crop_and_resize =
        ConstructNgNode<opset::Concat>(op->name(), ng_crop_outputs, 0);

    SaveNgOp(ng_op_map, op->name(), ng_crop_and_resize);
  }
  return Status::OK();
}

static Status TranslateCTCGreedyDecoderOp(const Node* op,
                                          const std::vector<const Tensor*>&,
                                          Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_inputs, ng_sequence_length;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_inputs, ng_sequence_length));

  // Undo the transpose done before calling CTCGreedyDecoder such that,
  // ng_inputs represents a tensor of shape [N, T, C] where N, T and C
  // represent batch_size, time_steps and num_classes respectively.
  ov::Shape transpose_order{1, 0, 2};
  auto input_order = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::u64, ov::Shape{transpose_order.size()},
      transpose_order);
  ng_inputs = std::make_shared<opset::Transpose>(ng_inputs, input_order);

  // Create a mask that stores valid time step indices, we need this to
  // handle samples of varying sequence lengths.
  ov::Shape ng_inputs_shape = ng_inputs.get_shape();
  auto batch_size = static_cast<long>(ng_inputs_shape.at(0));
  auto max_time_steps = static_cast<long>(ng_inputs_shape.at(1));

  auto ng_indicator = ConstructNgNode<opset::Range>(
      op->name(), ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                                   ov::Shape{}, 0),
      ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                       ov::Shape{}, max_time_steps),
      ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                       ov::Shape{}, 1),
      ov::element::i64);
  ng_indicator = ConstructNgNode<opset::Unsqueeze>(
      op->name(), ng_indicator,
      ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                       ov::Shape{1}, std::vector<int64>({0})));
  ng_indicator = ConstructNgNode<opset::Tile>(
      op->name(), ng_indicator, ConstructNgNode<opset::Constant>(
                                    op->name(), ov::element::i64, ov::Shape{2},
                                    std::vector<int64>({batch_size, 1})));

  auto ng_sequence_endpoints = ConstructNgNode<opset::Unsqueeze>(
      op->name(), ng_sequence_length,
      ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                       ov::Shape{1}, std::vector<int64>({1})));
  ng_sequence_endpoints = ConstructNgNode<opset::Tile>(
      op->name(), ng_sequence_endpoints,
      ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i64, ov::Shape{2},
          std::vector<int64>({1, max_time_steps})));
  ng_sequence_endpoints = ConstructNgNode<opset::Convert>(
      op->name(), ng_sequence_endpoints, ov::element::i64);

  auto ng_valid_time_steps_mask = ConstructNgNode<opset::Less>(
      op->name(), ng_indicator, ng_sequence_endpoints);
  ng_valid_time_steps_mask = ConstructNgNode<opset::Convert>(
      op->name(), ng_valid_time_steps_mask, ng_inputs.get_element_type());

  // Compute negative log sum probabilities for each sample in the batch by
  // selecting class index greedily across all valid time steps.
  auto ng_max_axis = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{}, 2);
  auto ng_sum_axis = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{}, 1);
  auto ng_log_probs =
      ConstructNgNode<opset::ReduceMax>(op->name(), ng_inputs, ng_max_axis, 0);

  ng_log_probs = ConstructNgNode<opset::Multiply>(op->name(), ng_log_probs,
                                                  ng_valid_time_steps_mask);
  ng_log_probs = ConstructNgNode<opset::ReduceSum>(op->name(), ng_log_probs,
                                                   ng_sum_axis, 1);
  ng_log_probs = ConstructNgNode<opset::Multiply>(
      op->name(), ng_log_probs,
      ConstructNgNode<opset::Constant>(op->name(), ng_inputs.get_element_type(),
                                       ov::Shape{}, -1));

  bool merge_repeated;
  int blank_index;

  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "merge_repeated", &merge_repeated));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "blank_index", &blank_index));

  if (blank_index < 0) {
    int num_classes = static_cast<int>(ng_inputs_shape.at(2));
    blank_index = num_classes + blank_index;
  }
  auto ng_blank_index = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{}, blank_index);

  auto ng_ctc_outputs = make_shared<opset::CTCGreedyDecoderSeqLen>(
      ng_inputs, ng_sequence_length, ng_blank_index, merge_repeated,
      ov::element::i64);
  auto ng_ctc_decoded_classes = ng_ctc_outputs->output(0);

  auto ng_ignore_value = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{}, -1);
  auto ng_decoded_mask = ConstructNgNode<opset::NotEqual>(
      op->name(), ng_ctc_decoded_classes, ng_ignore_value);

  // CTCGreedyDecoderSeqLen returns dense tensor holding the decoded results.
  // Compute indices and values that represent the decoded results in sparse
  // format. Since NonZero is not supported on iGPU currently, we enable
  // CTCGreedyDecoder only for CPU.
  auto ng_indices =
      ConstructNgNode<opset::NonZero>(op->name(), ng_decoded_mask);

  ov::Shape indices_transpose_order{1, 0};
  auto ng_indices_transpose_order = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::u64, ov::Shape{2}, indices_transpose_order);
  ng_indices = ConstructNgNode<opset::Transpose>(op->name(), ng_indices,
                                                 ng_indices_transpose_order);
  auto ng_values = ConstructNgNode<opset::GatherND>(
      op->name(), ng_ctc_decoded_classes, ng_indices);

  // Compute the shape of the smallest dense tensor that can contain the sparse
  // matrix represented by ng_indices and ng_values.
  auto ng_batch_size = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{1},
      std::vector<long>({batch_size}));
  auto ng_ctc_decoded_sequence_lens = ConstructNgNode<opset::Convert>(
      op->name(), ng_ctc_outputs->output(1), ov::element::i64);
  auto ng_decoded_max_time_steps = ConstructNgNode<opset::ReduceMax>(
      op->name(), ng_ctc_decoded_sequence_lens,
      ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                       ov::Shape{}, 0),
      1);

  auto ng_decoded_shape = ConstructNgNode<opset::Concat>(
      op->name(), ov::OutputVector({ng_batch_size, ng_decoded_max_time_steps}),
      0);

  SaveNgOp(ng_op_map, op->name(), ng_indices);
  SaveNgOp(ng_op_map, op->name(), ng_values);
  SaveNgOp(ng_op_map, op->name(), ng_decoded_shape);
  SaveNgOp(ng_op_map, op->name(), ng_log_probs);

  return Status::OK();
}

static Status TranslateFusedCTCGreedyDecoder(const Node* op,
                                             const std::vector<const Tensor*>&,
                                             Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_inputs, ng_sequence_length;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_inputs, ng_sequence_length));

  // Undo the transpose done before calling CTCGreedyDecoder such that,
  // ng_inputs represents a tensor of shape [N, T, C] where N, T and C
  // represent batch_size, time_steps and num_classes respectively.
  ov::Shape transpose_order{1, 0, 2};
  auto input_order = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::u64, ov::Shape{transpose_order.size()},
      transpose_order);
  ng_inputs = std::make_shared<opset::Transpose>(ng_inputs, input_order);

  // Create a mask that stores valid time step indices, we need this to
  // handle samples of varying sequence lengths.
  ov::Shape ng_inputs_shape = ng_inputs.get_shape();
  auto batch_size = static_cast<long>(ng_inputs_shape.at(0));
  auto max_time_steps = static_cast<long>(ng_inputs_shape.at(1));

  auto ng_indicator = ConstructNgNode<opset::Range>(
      op->name(), ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                                   ov::Shape{}, 0),
      ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                       ov::Shape{}, max_time_steps),
      ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                       ov::Shape{}, 1),
      ov::element::i64);
  ng_indicator = ConstructNgNode<opset::Unsqueeze>(
      op->name(), ng_indicator,
      ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                       ov::Shape{1}, std::vector<int64>({0})));
  ng_indicator = ConstructNgNode<opset::Tile>(
      op->name(), ng_indicator, ConstructNgNode<opset::Constant>(
                                    op->name(), ov::element::i64, ov::Shape{2},
                                    std::vector<int64>({batch_size, 1})));

  auto ng_sequence_endpoints = ConstructNgNode<opset::Unsqueeze>(
      op->name(), ng_sequence_length,
      ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                       ov::Shape{1}, std::vector<int64>({1})));
  ng_sequence_endpoints = ConstructNgNode<opset::Tile>(
      op->name(), ng_sequence_endpoints,
      ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i64, ov::Shape{2},
          std::vector<int64>({1, max_time_steps})));
  ng_sequence_endpoints = ConstructNgNode<opset::Convert>(
      op->name(), ng_sequence_endpoints, ov::element::i64);

  auto ng_valid_time_steps_mask = ConstructNgNode<opset::Less>(
      op->name(), ng_indicator, ng_sequence_endpoints);
  ng_valid_time_steps_mask = ConstructNgNode<opset::Convert>(
      op->name(), ng_valid_time_steps_mask, ng_inputs.get_element_type());

  // Compute negative log sum probabilities for each sample in the batch by
  // selecting class index greedily across all valid time steps.
  auto ng_max_axis = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{}, 2);
  auto ng_sum_axis = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{}, 1);
  auto ng_log_probs =
      ConstructNgNode<opset::ReduceMax>(op->name(), ng_inputs, ng_max_axis, 0);

  ng_log_probs = ConstructNgNode<opset::Multiply>(op->name(), ng_log_probs,
                                                  ng_valid_time_steps_mask);
  ng_log_probs = ConstructNgNode<opset::ReduceSum>(op->name(), ng_log_probs,
                                                   ng_sum_axis, 1);
  ng_log_probs = ConstructNgNode<opset::Multiply>(
      op->name(), ng_log_probs,
      ConstructNgNode<opset::Constant>(op->name(), ng_inputs.get_element_type(),
                                       ov::Shape{}, -1));

  bool merge_repeated;
  int blank_index;

  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "merge_repeated", &merge_repeated));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "blank_index", &blank_index));

  if (blank_index < 0) {
    int num_classes = static_cast<int>(ng_inputs_shape.at(2));
    blank_index = num_classes + blank_index;
  }
  auto ng_blank_index = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{}, blank_index);

  auto ng_ctc_outputs = make_shared<opset::CTCGreedyDecoderSeqLen>(
      ng_inputs, ng_sequence_length, ng_blank_index, merge_repeated,
      ov::element::i64);
  auto ng_ctc_decoded_classes = ng_ctc_outputs->output(0);

  // save dummy outputs at index 0, 1 and 2 as we do not require these.
  auto ng_zeros = ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                                   ov::Shape{}, 0);
  SaveNgOp(ng_op_map, op->name(), ng_zeros);
  SaveNgOp(ng_op_map, op->name(), ng_zeros);
  SaveNgOp(ng_op_map, op->name(), ng_zeros);
  SaveNgOp(ng_op_map, op->name(), ng_log_probs);

  // save output for SparseToDense, since ops further in the graph require it
  auto edges =
      std::vector<const Edge*>(op->out_edges().begin(), op->out_edges().end());
  SaveNgOp(ng_op_map, edges.at(1)->dst()->name(), ng_ctc_decoded_classes);

  return Status::OK();
}

static Status TranslateCumsumOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_x, ng_axis;
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
  ov::Output<ov::Node> ng_input;
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
  ov::Output<ov::Node> depth_to_space = ConstructNgNode<opset::DepthToSpace>(
      op->name(), ng_input, ng_mode, block_size);
  NCHWtoNHWC(op->name(), is_nhwc, depth_to_space);
  SaveNgOp(ng_op_map, op->name(), depth_to_space);
  return Status::OK();
}

static Status TranslateDepthwiseConv2dNativeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_filter;
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

  OVTF_VLOG(3) << ngraph::join(tf_strides);
  OVTF_VLOG(3) << ngraph::join(tf_dilations);
  OVTF_VLOG(3) << tf_padding_type;
  OVTF_VLOG(3) << tf_data_format;

  ov::Strides ng_strides(2);
  ov::Strides ng_dilations(2);
  ov::Shape ng_image_shape(2);
  ov::Shape ng_kernel_shape(2);

  NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
  NHWCtoHW(is_nhwc, tf_strides, ng_strides);
  NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);

  OVTF_VLOG(3) << "ng_strides: " << ngraph::join(ng_strides);
  OVTF_VLOG(3) << "ng_dilations: " << ngraph::join(ng_dilations);
  OVTF_VLOG(3) << "ng_image_shape: " << ngraph::join(ng_image_shape);

  auto& ng_filter_shape = ng_filter.get_shape();
  ng_kernel_shape[0] = ng_filter_shape[0];
  ng_kernel_shape[1] = ng_filter_shape[1];

  OVTF_VLOG(3) << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

  ov::CoordinateDiff ng_padding_below;
  ov::CoordinateDiff ng_padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, ng_padding_below,
                       ng_padding_above);

  // H W I M -> H W I 1 M
  auto filter_shape = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::u64, ov::Shape{5},
      ov::Shape{ng_filter_shape[0], ng_filter_shape[1], ng_filter_shape[2], 1,
                ng_filter_shape[3]});
  auto reshaped_filter = ConstructNgNode<opset::Reshape>(op->name(), ng_filter,
                                                         filter_shape, false);

  // H W I 1 M -> I M 1 H W
  auto order = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{5}, vector<int64>{2, 4, 3, 0, 1});
  auto transposed_filter =
      ConstructNgNode<opset::Transpose>(op->name(), reshaped_filter, order);

  auto ng_conv = ConstructNgNode<opset::GroupConvolution>(
      op->name(), ng_input, transposed_filter, ng_strides, ng_padding_below,
      ng_padding_above, ng_dilations);

  NCHWtoNHWC(op->name(), is_nhwc, ng_conv);
  SaveNgOp(ng_op_map, op->name(), ng_conv);
  return Status::OK();
}

static Status TranslateEluOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  // No alpha in TF, so default to 1.0
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Elu>(op->name(), ng_input, 1.0));
  return Status::OK();
}

static Status TranslateExpandDimsOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  std::vector<int64> dims;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &dims));
  auto ng_dims = ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                                  ov::Shape{dims.size()}, dims);
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Unsqueeze>(op->name(), ng_input, ng_dims));
  return Status::OK();
}

static Status TranslateFakeQuantWithMinMaxVarsOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_min, ng_max;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, ng_min));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 2, ng_max));

  bool narrow_range = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "narrow_range", &narrow_range));
  int64 num_bits;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_bits", &num_bits));

  auto levels = std::pow(2, num_bits) - int(narrow_range);

  auto min_less_max = ConstructNgNode<opset::Less>(
      op->name() + "/if_min_less_max", ng_min, ng_max);
  auto minimum = ConstructNgNode<opset::Select>(op->name() + "/minimum",
                                                min_less_max, ng_min, ng_max);
  auto maximum = ConstructNgNode<opset::Select>(op->name() + "/maximum",
                                                min_less_max, ng_max, ng_min);

  auto zero =
      ConstructNgNode<opset::Constant>(op->name(), ng_min.get_element_type(),
                                       ov::Shape{}, std::vector<int>({0}));

  auto min_greater_zero = ConstructNgNode<opset::Greater>(
      op->name() + "/if_minimum_greater_zero", minimum, zero);
  auto max_minus_min = ConstructNgNode<opset::Subtract>(
      op->name() + "/max_minus_min", maximum, minimum);
  minimum = ConstructNgNode<opset::Select>(op->name() + "/first_adj_min",
                                           min_greater_zero, zero, minimum);
  maximum = ConstructNgNode<opset::Select>(
      op->name() + "/first_adj_max", min_greater_zero, max_minus_min, maximum);

  auto max_less_zero = ConstructNgNode<opset::Less>(
      op->name() + "/if_max_less_zero", maximum, zero);
  auto min_minus_max = ConstructNgNode<opset::Subtract>(
      op->name() + "/min_minus_max", minimum, maximum);
  minimum = ConstructNgNode<opset::Select>(
      op->name() + "/second_adj_min", max_less_zero, min_minus_max, minimum);
  maximum = ConstructNgNode<opset::Select>(op->name() + "/second_adj_max",
                                           max_less_zero, zero, maximum);

  auto float_range = ConstructNgNode<opset::Subtract>(
      op->name() + "/float_range", maximum, minimum);
  auto quant_min_value = int(narrow_range);
  auto quant_max_value = std::pow(2, num_bits) - 1;
  float value = static_cast<float>(quant_max_value - quant_min_value);
  auto int_range = ConstructNgNode<opset::Constant>(
      op->name() + "/int_range", ov::element::f32, ov::Shape{},
      std::vector<float>({value}));
  auto scale = ConstructNgNode<opset::Divide>(op->name() + "/scale",
                                              float_range, int_range);
  auto descaled_min = ConstructNgNode<opset::Divide>(
      op->name() + "/descaled_min", minimum, scale);
  auto rounded_descaled_min = ConstructNgNode<opset::Round>(
      op->name() + "/rounded_descaled_min", descaled_min,
      opset::Round::RoundMode::HALF_TO_EVEN);
  auto min_adj = ConstructNgNode<opset::Multiply>(op->name() + "/min_adj",
                                                  scale, rounded_descaled_min);
  auto adjustment = ConstructNgNode<opset::Subtract>(
      op->name() + "/limits_adjustment", min_adj, minimum);
  auto max_adj =
      ConstructNgNode<opset::Add>(op->name() + "/max_adj", maximum, adjustment);

  auto ng_input_shape = ng_input.get_shape();
  if (ng_input_shape.size() == 4) Transpose<0, 3, 1, 2>(ng_input);
  auto ng_output = ConstructNgNode<opset::FakeQuantize>(
      op->name(), ng_input, min_adj, max_adj, min_adj, max_adj, levels);
  if (ng_input_shape.size() == 4) Transpose<0, 2, 3, 1>(ng_output);

  SaveNgOp(ng_op_map, op->name(), ng_output);

  return Status::OK();
}

static Status TranslateFillOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_value, ng_dims;
  // TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_dims, ng_value));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_dims));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, ng_value));
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Broadcast>(op->name(), ng_value, ng_dims));
  return Status::OK();
}

static Status TranslateFloorDivOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  auto floordiv_fn = [&op](ov::Output<ov::Node> x, ov::Output<ov::Node> y) {
    return ConstructNgNode<opset::Floor>(
        op->name(), ConstructNgNode<opset::Divide>(op->name(), x, y));
  };
  return TranslateBinaryOp(op, static_input_map, ng_op_map, floordiv_fn);
}

static Status TranslateFusedBatchNormOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_scale, ng_offset, ng_mean, ng_variance;
  bool is_v3 = op->type_string() == "FusedBatchNormV3";
  bool is_Ex = op->type_string() == "_FusedBatchNormEx";

  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_scale, ng_offset,
                                   ng_mean, ng_variance));

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "Conv2D data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  OVTF_VLOG(3) << "data_format: " << tf_data_format;

  float tf_epsilon;
  if (GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon) != Status::OK()) {
    OVTF_VLOG(3) << "epsilon attribute not present, setting to 0.0001";
    // TensorFlow default
    tf_epsilon = 0.0001;
  }

  OVTF_VLOG(3) << "epsilon: " << tf_epsilon;

  NHWCtoNCHW(op->name(), is_nhwc, ng_input);

  auto ng_batch_norm = ConstructNgNode<opset::BatchNormInference>(
      op->name(), ng_input, ng_scale, ng_offset, ng_mean, ng_variance,
      tf_epsilon);
  NCHWtoNHWC(op->name(), is_nhwc, ng_batch_norm);

  if (is_Ex) {
    string activation_mode;
    TF_RETURN_IF_ERROR(
        GetNodeAttr(op->attrs(), "activation_mode", &activation_mode));

    if (activation_mode == "Relu") {
      auto relu_op = ConstructNgNode<opset::Relu>(op->name(), ng_batch_norm);
      SaveNgOp(ng_op_map, op->name(), relu_op);
    } else {
      return errors::Unimplemented(
          "Unsupported _FusedBatchNormEx activation mode in " + op->name());
    }
  } else {
    SaveNgOp(ng_op_map, op->name(), ng_batch_norm);
    SaveNgOp(ng_op_map, op->name(), ng_mean);
    SaveNgOp(ng_op_map, op->name(), ng_variance);
    SaveNgOp(ng_op_map, op->name(), ng_mean);      // reserve_space_1
    SaveNgOp(ng_op_map, op->name(), ng_variance);  // reserve_space_2
    if (is_v3) {
      // FusedBatchNormV3 has 6 outputs
      SaveNgOp(ng_op_map, op->name(), ng_mean);  // reserve_space_3
    }
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

  ov::Output<ov::Node> ng_lhs, ng_rhs, ng_bias, ng_matmul;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_lhs, ng_rhs, ng_bias));
  ng_matmul = ConstructNgNode<opset::MatMul>(op->name(), ng_lhs, ng_rhs,
                                             transpose_a, transpose_b);

  auto ng_matmul_shape = ng_matmul.get_shape();
  auto ng_bias_rank = ng_bias.get_partial_shape().rank().get_length();

  if (ng_bias_rank != 1) {
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

// See .../tensorflow/include/tensorflow/cc/ops/array_ops.h
// and .../openvino/ngraph/core/include/ngraph/op/gather.hpp
static Status TranslateGatherOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_input_indices;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_input_indices));

  auto ng_axis = ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                                  ov::Shape{}, 0);

  auto gather_op = ConstructNgNode<opset::Gather>(op->name(), ng_input,
                                                  ng_input_indices, ng_axis);

  SaveNgOp(ng_op_map, op->name(), gather_op);
  return Status::OK();
}

static Status TranslateGatherV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_input_coords, ng_unused;
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
  size_t ng_input_rank = ng_input.get_partial_shape().rank().get_length();
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
      op->name(), ov::element::i64, ov::Shape{tf_axis.size()}, tf_axis);

  auto gather_op = ConstructNgNode<opset::Gather>(op->name(), ng_input,
                                                  ng_input_coords, ng_axis);

  SaveNgOp(ng_op_map, op->name(), gather_op);
  return Status::OK();
}

static Status TranslateGatherNdOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_input_indices;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_input_indices));

  int batch_dims = 0;
  // TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "batch_dims", &batch_dims));

  auto gathernd_op = ConstructNgNode<opset::GatherND>(
      op->name(), ng_input, ng_input_indices, batch_dims);

  SaveNgOp(ng_op_map, op->name(), gathernd_op);
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

  auto CreateNgConv = [&](ov::Output<ov::Node>& ng_input,
                          ov::Output<ov::Node>& ng_filter,
                          ov::Output<ov::Node>& ng_conv) {
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

    OVTF_VLOG(3) << ngraph::join(tf_strides);
    OVTF_VLOG(3) << ngraph::join(tf_dilations);
    OVTF_VLOG(3) << tf_padding_type;
    OVTF_VLOG(3) << tf_data_format;

    ov::Strides ng_strides(2);
    ov::Strides ng_dilations(2);
    ov::Shape ng_image_shape(2);
    ov::Shape ng_kernel_shape(2);

    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    if (ng_input.get_partial_shape().is_static()) {
      NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    }
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(op->name(), is_nhwc, ng_input);

    OVTF_VLOG(3) << "ng_strides: " << ngraph::join(ng_strides);
    OVTF_VLOG(3) << "ng_dilations: " << ngraph::join(ng_dilations);
    OVTF_VLOG(3) << "ng_image_shape: " << ngraph::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    Transpose<3, 2, 0, 1>(ng_filter);
    Builder::SetTracingInfo(op->name(), ng_filter);

    OVTF_VLOG(3) << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

    ov::CoordinateDiff ng_padding_below;
    ov::CoordinateDiff ng_padding_above;
    // Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
    //                      ng_strides, ng_dilations, ng_padding_below,
    //                      ng_padding_above);
    std::vector<int32> tf_paddings;
    ov::op::PadType pad_type;
    if (tf_padding_type == "EXPLICIT") {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(op->attrs(), "explicit_paddings", &tf_paddings));
      if (is_nhwc) {
        ng_padding_below.push_back(tf_paddings[2]);
        ng_padding_below.push_back(tf_paddings[4]);
        ng_padding_above.push_back(tf_paddings[3]);
        ng_padding_above.push_back(tf_paddings[5]);
      } else {
        ng_padding_below.push_back(tf_paddings[4]);
        ng_padding_below.push_back(tf_paddings[6]);
        ng_padding_above.push_back(tf_paddings[5]);
        ng_padding_above.push_back(tf_paddings[7]);
      }
      OVTF_VLOG(3) << " ========== EXPLICIT Padding ========== ";
      OVTF_VLOG(3) << "ng_padding_below: " << ngraph::join(ng_padding_below);
      OVTF_VLOG(3) << "ng_padding_above: " << ngraph::join(ng_padding_above);
      ng_conv = ConstructNgNode<opset::Convolution>(
          op->name() + "_FusedConv2D_Conv", ng_input, ng_filter, ng_strides,
          ng_padding_below, ng_padding_above, ng_dilations);
    } else if (tf_padding_type == "VALID") {
      ng_padding_below.assign(ng_image_shape.size(), 0);
      ng_padding_above.assign(ng_image_shape.size(), 0);
      ng_conv = ConstructNgNode<opset::Convolution>(
          op->name() + "_FusedConv2D_Conv", ng_input, ng_filter, ng_strides,
          ng_padding_below, ng_padding_above, ng_dilations);
    } else if (tf_padding_type == "SAME") {
      if (ng_input.get_partial_shape().is_static()) {
        OVTF_VLOG(3) << "========== SAME Padding - Static Shape ========== ";
        ov::Shape img_shape = {0, 0};
        img_shape.insert(img_shape.end(), ng_image_shape.begin(),
                         ng_image_shape.end());
        ov::infer_auto_padding(img_shape, ng_kernel_shape, ng_strides,
                               ng_dilations, ov::op::PadType::SAME_UPPER,
                               ng_padding_above, ng_padding_below);
        ng_conv = ConstructNgNode<opset::Convolution>(
            op->name() + "_FusedConv2D_Conv", ng_input, ng_filter, ng_strides,
            ng_padding_below, ng_padding_above, ng_dilations);
      } else {
        OVTF_VLOG(3) << "========== SAME Padding - Dynamic Shape ========== ";
        pad_type = ov::op::PadType::SAME_UPPER;
        ng_conv = ConstructNgNode<opset::Convolution>(
            op->name() + "_FusedConv2D_Conv", ng_input, ng_filter, ng_strides,
            ng_padding_below, ng_padding_above, ng_dilations, pad_type);
      }
    }
    return Status::OK();
  };

  if (VecStrCmp(fused_ops, {"BiasAdd"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu6"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "LeakyRelu"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Elu"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Add", "Relu"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Add"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Add", "LeakyRelu"})) {
    ov::Output<ov::Node> ng_input, ng_filter, ng_bias, ng_conv, ng_input2;
    if (VecStrCmp(fused_ops, {"BiasAdd", "Add", "Relu"}) ||
        VecStrCmp(fused_ops, {"BiasAdd", "Add"}) ||
        VecStrCmp(fused_ops, {"BiasAdd", "Add", "LeakyRelu"})) {
      if (num_args != 2) {
        return errors::InvalidArgument(
            "FusedConv2DBiasAdd has incompatible num_args");
      }
      TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_filter,
                                       ng_bias, ng_input2));
    } else {
      if (num_args != 1) {
        return errors::InvalidArgument(
            "FusedConv2DBiasAdd has incompatible num_args");
      }
      TF_RETURN_IF_ERROR(
          GetInputNodes(ng_op_map, op, ng_input, ng_filter, ng_bias));
    }

    TF_RETURN_IF_ERROR(CreateNgConv(ng_input, ng_filter, ng_conv));

    auto ng_conv_rank = ng_conv.get_partial_shape().rank().get_length();
    auto ng_bias_rank = ng_bias.get_partial_shape().rank().get_length();
    if (ng_bias_rank != 1) {
      return errors::InvalidArgument(
          "Bias argument to BiasAdd does not have one dimension");
    }

    std::vector<size_t> reshape_pattern_values(ng_conv_rank, 1U);
    reshape_pattern_values[1] = ng_bias.get_shape().front();
    auto reshape_pattern = make_shared<opset::Constant>(
        ov::element::u64, ov::Shape{reshape_pattern_values.size()},
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
    } else if (VecStrCmp(fused_ops, {"BiasAdd", "LeakyRelu"})) {
      float tf_leakyrelu_alpha;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(op->attrs(), "leakyrelu_alpha", &tf_leakyrelu_alpha));
      auto ng_leakyrelu_alpha = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::f32, ov::Shape{}, tf_leakyrelu_alpha);
      auto ng_alphax = ConstructNgNode<opset::Multiply>(
          op->name(), ng_leakyrelu_alpha, ng_add);
      auto ng_lrelu = ConstructNgNode<opset::Maximum>(
          op->name() + "_FusedConv2D_LeakyRelu", ng_alphax, ng_add);
      NCHWtoNHWC(op->name(), is_nhwc, ng_lrelu);
      SaveNgOp(ng_op_map, op->name(), ng_lrelu);
    } else if (VecStrCmp(fused_ops, {"BiasAdd", "Elu"})) {
      float tf_elu_alpha = 1.0;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(op->attrs(), "leakyrelu_alpha", &tf_elu_alpha));
      auto ng_elu = ConstructNgNode<opset::Elu>(op->name() + "_FusedConv2D_Elu",
                                                ng_add, tf_elu_alpha);
      NCHWtoNHWC(op->name(), is_nhwc, ng_elu);
      SaveNgOp(ng_op_map, op->name(), ng_elu);
    } else if (VecStrCmp(fused_ops, {"BiasAdd", "Add", "Relu"})) {
      NHWCtoNCHW(op->name(), is_nhwc, ng_input2);
      auto ng_add2 = ConstructNgNode<opset::Add>(
          op->name() + "_FusedConv2D_Add", ng_add, ng_input2);
      auto ng_relu = ConstructNgNode<opset::Relu>(
          op->name() + "_FusedConv2D_Relu", ng_add2);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu);
      SaveNgOp(ng_op_map, op->name(), ng_relu);
    } else if (VecStrCmp(fused_ops, {"BiasAdd", "Add"})) {
      ov::Output<ov::Node> ng_add_inp;
      // TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 3, ng_add_inp));
      NCHWtoNHWC(op->name(), is_nhwc, ng_add);
      auto ng_out = ConstructNgNode<opset::Add>(
          op->name() + "_FusedConv2D_BiasAdd_Add", ng_add, ng_input2);
      SaveNgOp(ng_op_map, op->name(), ng_out);
    } else if (VecStrCmp(fused_ops, {"BiasAdd", "Add", "LeakyRelu"})) {
      NHWCtoNCHW(op->name(), is_nhwc, ng_input2);
      auto ng_add2 = ConstructNgNode<opset::Add>(
          op->name() + "_FusedConv2D_Add", ng_add, ng_input2);
      float tf_leakyrelu_alpha;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(op->attrs(), "leakyrelu_alpha", &tf_leakyrelu_alpha));
      auto ng_leakyrelu_alpha = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::f32, ov::Shape{}, tf_leakyrelu_alpha);
      auto ng_alphax = ConstructNgNode<opset::Multiply>(
          op->name(), ng_leakyrelu_alpha, ng_add2);
      auto ng_alrelu = ConstructNgNode<opset::Maximum>(
          op->name() + "_FusedConv2D_Add_LeakyRelu", ng_alphax, ng_add2);
      NCHWtoNHWC(op->name(), is_nhwc, ng_alrelu);
      SaveNgOp(ng_op_map, op->name(), ng_alrelu);
    } else {
      NCHWtoNHWC(op->name(), is_nhwc, ng_add);
      SaveNgOp(ng_op_map, op->name(), ng_add);
    }
  } else if (VecStrCmp(fused_ops, {"FusedBatchNorm"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "Relu6"}) ||
             VecStrCmp(fused_ops, {"FusedBatchNorm", "LeakyRelu"})) {
    if (num_args != 4) {
      return errors::InvalidArgument(
          "FusedConv2D with FusedBatchNorm has incompatible num_args");
    }

    ov::Output<ov::Node> ng_input, ng_filter, ng_conv, ng_scale, ng_offset,
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
    } else if (VecStrCmp(fused_ops, {"FusedBatchNorm", "LeakyRelu"})) {
      float tf_leakyrelu_alpha;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(op->attrs(), "leakyrelu_alpha", &tf_leakyrelu_alpha));
      auto ng_leakyrelu_alpha = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::f32, ov::Shape{}, tf_leakyrelu_alpha);
      auto ng_alphax = ConstructNgNode<opset::Multiply>(
          op->name(), ng_leakyrelu_alpha, ng_batch_norm);
      auto ng_lrelu = ConstructNgNode<opset::Maximum>(
          op->name() + "_FusedConv2D_BatchNormLeakyRelu", ng_alphax,
          ng_batch_norm);
      NCHWtoNHWC(op->name(), is_nhwc, ng_lrelu);
      SaveNgOp(ng_op_map, op->name(), ng_lrelu);
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

static Status TranslateFusedDepthwiseConv2dNativeOp(
    const Node* op, const std::vector<const Tensor*>&,
    Builder::OpMap& ng_op_map) {
  int num_args;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_args", &num_args));

  std::vector<string> fused_ops;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "fused_ops", &fused_ops));

  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));
  bool is_nhwc = (tf_data_format == "NHWC");

  auto CreateNgDepthwiseConv = [&](ov::Output<ov::Node>& ng_input,
                                   ov::Output<ov::Node>& ng_filter,
                                   ov::Output<ov::Node>& ng_conv) {
    std::vector<int32> tf_strides;
    std::vector<int32> tf_dilations;
    std::string tf_padding_type;
    std::string tf_data_format;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
    TF_RETURN_IF_ERROR(
        GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
      return errors::InvalidArgument(
          "DepthwiseConv2D data format is neither NHWC nor NCHW");
    }

    bool is_nhwc = (tf_data_format == "NHWC");

    OVTF_VLOG(3) << ngraph::join(tf_strides);
    OVTF_VLOG(3) << ngraph::join(tf_dilations);
    OVTF_VLOG(3) << tf_padding_type;
    OVTF_VLOG(3) << tf_data_format;

    ov::Strides ng_strides(2);
    ov::Strides ng_dilations(2);
    ov::Shape ng_image_shape(2);
    ov::Shape ng_kernel_shape(2);

    NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_nhwc, tf_strides, ng_strides);
    NHWCtoHW(is_nhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(op->name(), is_nhwc, ng_input);

    OVTF_VLOG(3) << "ng_strides: " << ngraph::join(ng_strides);
    OVTF_VLOG(3) << "ng_dilations: " << ngraph::join(ng_dilations);
    OVTF_VLOG(3) << "ng_image_shape: " << ngraph::join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];

    OVTF_VLOG(3) << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

    ov::CoordinateDiff ng_padding_below;
    ov::CoordinateDiff ng_padding_above;
    Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                         ng_strides, ng_dilations, ng_padding_below,
                         ng_padding_above);

    // H W I M -> H W I 1 M
    auto filter_shape = ConstructNgNode<opset::Constant>(
        op->name(), ov::element::u64, ov::Shape{5},
        ov::Shape{ng_filter_shape[0], ng_filter_shape[1], ng_filter_shape[2], 1,
                  ng_filter_shape[3]});
    auto reshaped_filter = ConstructNgNode<opset::Reshape>(
        op->name(), ng_filter, filter_shape, false);

    // H W I 1 M -> I M 1 H W
    auto order = ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                                  ov::Shape{5},
                                                  vector<int64>{2, 4, 3, 0, 1});
    auto transposed_filter =
        ConstructNgNode<opset::Transpose>(op->name(), reshaped_filter, order);

    ng_conv = ConstructNgNode<opset::GroupConvolution>(
        op->name(), ng_input, transposed_filter, ng_strides, ng_padding_below,
        ng_padding_above, ng_dilations);

    return Status::OK();
  };

  if (VecStrCmp(fused_ops, {"BiasAdd"}) ||
      VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
    if (num_args != 1) {
      return errors::InvalidArgument(
          "FusedDepthwiseConv2dNativeBiasAdd has incompatible num_args");
    }

    ov::Output<ov::Node> ng_input, ng_filter, ng_bias, ng_conv;
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, ng_input, ng_filter, ng_bias));

    TF_RETURN_IF_ERROR(CreateNgDepthwiseConv(ng_input, ng_filter, ng_conv));

    auto ng_conv_rank = ng_conv.get_partial_shape().rank().get_length();
    auto ng_bias_rank = ng_bias.get_partial_shape().rank().get_length();
    if (ng_bias_rank != 1) {
      return errors::InvalidArgument(
          "Bias argument to BiasAdd does not have one dimension");
    }

    std::vector<size_t> reshape_pattern_values(ng_conv_rank, 1U);
    reshape_pattern_values[1] = ng_bias.get_shape().front();
    auto reshape_pattern = make_shared<opset::Constant>(
        ov::element::u64, ov::Shape{reshape_pattern_values.size()},
        reshape_pattern_values);
    auto ng_bias_reshaped = ConstructNgNode<opset::Reshape>(
        op->name(), ng_bias, reshape_pattern, false);

    auto ng_add = ConstructNgNode<opset::Add>(
        op->name() + "_FusedDepthwiseConv2dNative_BiasAdd", ng_conv,
        ng_bias_reshaped);

    if (VecStrCmp(fused_ops, {"BiasAdd", "Relu6"})) {
      auto ng_relu6 = ConstructNgNode<opset::Clamp>(
          op->name() + "_FusedDepthwiseConv2dNative_Relu6", ng_add, 0, 6);
      NCHWtoNHWC(op->name(), is_nhwc, ng_relu6);
      SaveNgOp(ng_op_map, op->name(), ng_relu6);
    } else {
      NCHWtoNHWC(op->name(), is_nhwc, ng_add);
      SaveNgOp(ng_op_map, op->name(), ng_add);
    }
  } else {
    return errors::Unimplemented("Unsupported _FusedDepthwiseConv2dNative " +
                                 absl::StrJoin(fused_ops, ","));
  }
  return Status::OK();
}

static Status TranslateIdentityOp(const Node* op,
                                  const std::vector<const Tensor*>&,
                                  Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_arg;
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
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  auto const_inf = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), ov::Shape{},
      std::vector<float>{std::numeric_limits<float>::infinity()});

  auto const_neg_inf = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), ov::Shape{},
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
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  std::vector<float> val;
  val.push_back(2.0);
  auto const_2 = ConstructNgNode<opset::Constant>(
      op->name(), ng_input.get_element_type(), ov::Shape{}, val[0]);

  auto ng_pow =
      ConstructNgNode<opset::Multiply>(op->name(), ng_input, ng_input);

  size_t input_rank = ng_input.get_partial_shape().rank().get_length();
  std::vector<int64> axes;
  for (size_t i = 0; i < input_rank; ++i) {
    axes.push_back(i);
  }

  auto ng_reduction_axes = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{axes.size()}, axes);
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
      op, static_input_map, ng_op_map, [&op](ov::Output<ov::Node> n) {
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> val_1(ov::shape_size(shape), "1");
        auto ng_const1 =
            ConstructNgNode<opset::Constant>(op->name(), et, shape, val_1);
        auto ng_add = ConstructNgNode<opset::Add>(op->name(), ng_const1, n);
        return ConstructNgNode<opset::Log>(op->name(), ng_add);
      });
}

static Status TranslateLRNOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_inp;
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
  // OpenVINO expects the input to be in NCHW format
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
  ov::Output<ov::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp));
  auto inp_shape = ng_inp.get_shape();
  size_t rank = inp_shape.size();
  int64 axes = rank - 1;

  auto ng_output = ConstructNgNode<opset::LogSoftmax>(op->name(), ng_inp, axes);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateLeakyReluOp(const Node* op,
                                   const std::vector<const Tensor*>&,
                                   Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp));
  float alpha = 0.0;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "alpha", &alpha));

  auto ng_alpha = ConstructNgNode<opset::Constant>(op->name(), ov::element::f32,
                                                   ov::Shape{1}, alpha);

  auto ng_output = ConstructNgNode<opset::PRelu>(op->name(), ng_inp, ng_alpha);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateMatMulOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_lhs, ng_rhs;
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
  ov::Output<ov::Node> ng_input;
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

  OVTF_VLOG(3) << ngraph::join(tf_strides);
  OVTF_VLOG(3) << ngraph::join(tf_ksize);
  OVTF_VLOG(3) << tf_padding_type;
  OVTF_VLOG(3) << tf_data_format;

  ov::Strides ng_strides(N);
  ov::Shape ng_image_shape(N);
  ov::Shape ng_kernel_shape(N);
  ov::Shape ng_dilations(N, 1);

  NHWCtoHW(is_nhwc, tf_strides, ng_strides);
  NHWCtoHW(is_nhwc, ng_input.get_shape(), ng_image_shape);
  NHWCtoHW(is_nhwc, tf_ksize, ng_kernel_shape);
  NHWCtoNCHW(op->name(), is_nhwc, ng_input);
  OVTF_VLOG(3) << "ng_strides: " << ngraph::join(ng_strides);
  OVTF_VLOG(3) << "ng_image_shape: " << ngraph::join(ng_image_shape);
  OVTF_VLOG(3) << "ng_kernel_shape: " << ngraph::join(ng_kernel_shape);

  ov::CoordinateDiff padding_below;
  ov::CoordinateDiff padding_above;
  Builder::MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape,
                       ng_strides, ng_dilations, padding_below, padding_above);

  // TODO: remove this once OpenVINO supports negative padding
  // (CoordinateDiff) for MaxPool
  ov::Shape ng_padding_below(padding_below.begin(), padding_below.end());
  ov::Shape ng_padding_above(padding_above.begin(), padding_above.end());

  auto ng_maxpool = ConstructNgNode<opset::MaxPool>(
      op->name(), ng_input, ng_strides, ng_dilations, ng_padding_below,
      ng_padding_above, ng_kernel_shape, ov::op::RoundingType::FLOOR);

  NCHWtoNHWC(op->name(), is_nhwc, ng_maxpool);

  OVTF_VLOG(3) << "maxpool outshape: {" << ngraph::join(ng_maxpool.get_shape())
               << "}";

  SaveNgOp(ng_op_map, op->name(), ng_maxpool);
  return Status::OK();
}

static Status TranslateMklSwishOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  auto ng_sigmoid = ConstructNgNode<opset::Sigmoid>(op->name(), ng_input);
  auto ng_result =
      ConstructNgNode<opset::Multiply>(op->name(), ng_input, ng_sigmoid);
  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

static Status TranslateNonMaxSuppressionOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_boxes, ng_scores, ng_max_output_size,
      ng_iou_threshold;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_boxes));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, ng_scores));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 2, ng_max_output_size));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 3, ng_iou_threshold));
  auto ng_axis_boxes = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{1}, std::vector<int64>({0}));
  auto ng_boxes_unsqueezed =
      ConstructNgNode<opset::Unsqueeze>(op->name(), ng_boxes, ng_axis_boxes);

  auto ng_axis_scores = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{2}, std::vector<int64>({0, 1}));
  auto ng_scores_unsqueezed =
      ConstructNgNode<opset::Unsqueeze>(op->name(), ng_scores, ng_axis_scores);

  const auto& op_type = op->type_string();
  std::shared_ptr<opset::NonMaxSuppression> ng_nms;
  if (op_type == "NonMaxSuppressionV5") {
    ov::Output<ov::Node> ng_score_threshold, ng_soft_nms_sigma;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 4, ng_score_threshold));
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 5, ng_soft_nms_sigma));
    // TODO: pad_to_max_output_size and then remove the corresponding constraint
    // check from OCM
    ng_nms = make_shared<opset::NonMaxSuppression>(
        ng_boxes_unsqueezed, ng_scores_unsqueezed, ng_max_output_size,
        ng_iou_threshold, ng_score_threshold, ng_soft_nms_sigma,
        opset::NonMaxSuppression::BoxEncodingType::CORNER, false,
        ov::element::Type_t::i32);
  } else if (op_type == "NonMaxSuppressionV4") {
    ov::Output<ov::Node> ng_score_threshold;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 4, ng_score_threshold));
    // TODO: pad_to_max_output_size and then remove the corresponding constraint
    // check from OCM
    ng_nms = make_shared<opset::NonMaxSuppression>(
        ng_boxes_unsqueezed, ng_scores_unsqueezed, ng_max_output_size,
        ng_iou_threshold, ng_score_threshold,
        opset::NonMaxSuppression::BoxEncodingType::CORNER, false,
        ov::element::Type_t::i32);
  } else if (op_type == "NonMaxSuppressionV3") {
    ov::Output<ov::Node> ng_score_threshold;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 4, ng_score_threshold));
    ng_nms = make_shared<opset::NonMaxSuppression>(
        ng_boxes_unsqueezed, ng_scores_unsqueezed, ng_max_output_size,
        ng_iou_threshold, ng_score_threshold,
        opset::NonMaxSuppression::BoxEncodingType::CORNER, false,
        ov::element::Type_t::i32);
  } else if (op_type == "NonMaxSuppressionV2" ||
             op_type == "NonMaxSuppression") {
    ng_nms = make_shared<opset::NonMaxSuppression>(
        ng_boxes_unsqueezed, ng_scores_unsqueezed, ng_max_output_size,
        ng_iou_threshold, opset::NonMaxSuppression::BoxEncodingType::CORNER,
        false, ov::element::Type_t::i32);
  } else {
    throw runtime_error(op_type + " is not supported");
  }

  std::string device;
  // Correct output variables dimensions for CPU device
  Status exec_status = BackendManager::GetBackendName(device);
  if (exec_status != Status::OK()) {
    throw runtime_error(exec_status.error_message());
  }

  if (device == "CPU") {
    // selected_indices output from OV doesn't have same structure as of TF for
    // CPU device for all the NMS ops
    auto begin = ConstructNgNode<opset::Constant>(
        op->name(), ov::element::i64, ov::Shape{2}, std::vector<int64>({0, 2}));
    auto end = ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                                ov::Shape{2},
                                                std::vector<int64>({0, -1}));
    auto ng_nms_selected_indices = ConstructNgNode<opset::StridedSlice>(
        op->name(), ng_nms->outputs()[0], begin, end,
        std::vector<int64_t>{1, 0}, std::vector<int64_t>{1, 0},
        std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 1});

    SaveNgOp(ng_op_map, op->name(), ng_nms_selected_indices);

    // selected_scores and valid_outputs shape from OV is not in sync with
    // TF output and needs extra transformation
    if (op_type == "NonMaxSuppressionV5") {
      // selected_scores needs same transformation as selected_indices
      auto ng_nms_selected_scores = ConstructNgNode<opset::StridedSlice>(
          op->name(), ng_nms->outputs()[1], begin, end,
          std::vector<int64_t>{1, 0}, std::vector<int64_t>{1, 0},
          std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 1});
      // valid_outputs is 1D tensor in case of OV, and 0D tensor in case of TF
      auto ng_squeeze_axis = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i32, ov::Shape{}, 0);
      auto valid_outputs = ConstructNgNode<opset::Squeeze>(
          op->name(), ng_nms->outputs()[2], ng_squeeze_axis);
      SaveNgOp(ng_op_map, op->name(), ng_nms_selected_scores);
      SaveNgOp(ng_op_map, op->name(), valid_outputs);
    } else if (op_type == "NonMaxSuppressionV4") {
      auto ng_squeeze_axis = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i32, ov::Shape{}, 0);
      auto valid_outputs = ConstructNgNode<opset::Squeeze>(
          op->name(), ng_nms->outputs()[1], ng_squeeze_axis);
      SaveNgOp(ng_op_map, op->name(), valid_outputs);
    }
  } else {
    // for GPU and MYRIAD the default output works properly
    // except for valid_outputs in NMSV5 and NMSV4
    SaveNgOp(ng_op_map, op->name(), ng_nms->outputs()[0]);
    if (op_type == "NonMaxSuppressionV5") {
      // valid_outputs is 1D tensor in case of OV, and 0D tensor in case of TF
      auto ng_squeeze_axis = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i32, ov::Shape{}, 0);
      auto valid_outputs = ConstructNgNode<opset::Squeeze>(
          op->name(), ng_nms->outputs()[2], ng_squeeze_axis);
      SaveNgOp(ng_op_map, op->name(), ng_nms->outputs()[1]);
      SaveNgOp(ng_op_map, op->name(), valid_outputs);
    } else if (op_type == "NonMaxSuppressionV4") {
      auto ng_squeeze_axis = ConstructNgNode<opset::Constant>(
          op->name(), ov::element::i32, ov::Shape{}, 0);
      auto valid_outputs = ConstructNgNode<opset::Squeeze>(
          op->name(), ng_nms->outputs()[1], ng_squeeze_axis);
      SaveNgOp(ng_op_map, op->name(), valid_outputs);
    }
  }
  return Status::OK();
}

static Status TranslateReduceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map,
    std::function<ov::Output<ov::Node>(ov::Output<ov::Node>,
                                       ov::Output<ov::Node>, const bool)>
        create_ng_node) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  bool tf_keep_dims;
  if (GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) != Status::OK()) {
    tf_keep_dims = false;
  }

  std::vector<int64> axes;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &axes));

  size_t input_rank = ng_input.get_partial_shape().rank().get_length();

  TF_RETURN_IF_ERROR(CheckAxisDimInRange(axes, input_rank));

  std::vector<size_t> ng_reduction_axes_vect(axes.size());
  std::transform(
      axes.begin(), axes.end(), ng_reduction_axes_vect.begin(),
      [input_rank](int idx) { return idx + (idx < 0 ? (int)input_rank : 0); });
  auto ng_reduction_axes = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{ng_reduction_axes_vect.size()},
      ng_reduction_axes_vect);

  ov::Output<ov::Node> ng_node =
      create_ng_node(ng_input, ng_reduction_axes, tf_keep_dims);

  SaveNgOp(ng_op_map, op->name(), ng_node);
  return Status::OK();
}

template <typename T>
static Status TranslateDirectReduceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  // ensure its either an arithmetic or a logical reduction
  if (!(std::is_base_of<ov::op::util::ArithmeticReduction, T>::value ||
        std::is_base_of<ov::op::util::LogicalReduction, T>::value)) {
    return errors::InvalidArgument(
        "Expected node to be either a valid logical or arithmetic reduction "
        "type");
  }
  return TranslateReduceOp(
      op, static_input_map, ng_op_map,
      [&op](ov::Output<ov::Node> ng_input,
            ov::Output<ov::Node> ng_reduction_axes, const bool keep_dims) {
        return ConstructNgNode<T>(op->name(), ng_input, ng_reduction_axes,
                                  keep_dims);
      });
}

static Status TranslateOneHotOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_features, ng_depth, ng_on, ng_off;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_features, ng_depth, ng_on, ng_off));

  int one_hot_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &one_hot_axis));

  auto ng_onehot = ConstructNgNode<opset::OneHot>(
      op->name(), ng_features, ng_depth, ng_on, ng_off, one_hot_axis);
  SaveNgOp(ng_op_map, op->name(), ng_onehot);
  return Status::OK();
}

static Status TranslatePackOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 1));

  int32 tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  auto ng_axis = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{1},
      std::vector<int64>({tf_axis}));

  ov::OutputVector ng_concat_inputs;
  for (tensorflow::int32 i = 0; i < op->num_inputs(); ++i) {
    ov::Output<ov::Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, i, ng_input));
    auto unsqueezed_input =
        ConstructNgNode<opset::Unsqueeze>(op->name(), ng_input, ng_axis);
    ng_concat_inputs.push_back(unsqueezed_input);
  }

  // if inputs shape is (2, 3, 4), and axis is 1, then we want
  // to create output_shape (2, num_inputs, 3, 4)
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Concat>(
                                      op->name(), ng_concat_inputs, tf_axis));
  return Status::OK();
}

// 3 different Pad Ops: Pad, PadV2, MirrorPad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/pad-v2
// See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mirror-pad
static Status TranslatePadOp(const Node* op,
                             const std::vector<const Tensor*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_paddings_op, pad_val_op, result_pad_op;

  // Set inputs and pad_val_op
  if (op->type_string() == "Pad" || op->type_string() == "MirrorPad") {
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_paddings_op));
    pad_val_op = ConstructNgNode<opset::Constant>(
        op->name(), ng_input.get_element_type(), ov::Shape(),
        std::vector<int>({0}));
  } else if (op->type_string() == "PadV2") {
    TF_RETURN_IF_ERROR(
        GetInputNodes(ng_op_map, op, ng_input, ng_paddings_op, pad_val_op));
  } else {
    return errors::InvalidArgument("Incorrect TF Pad OpType: " +
                                   op->type_string());
  }

  // Set pad_mode
  auto pad_mode = ov::op::PadMode::CONSTANT;
  if (op->type_string() == "MirrorPad") {
    std::string pad_mode_str;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "mode", &pad_mode_str));
    if (pad_mode_str == "REFLECT") {
      pad_mode = ov::op::PadMode::REFLECT;
    } else if (pad_mode_str == "SYMMETRIC") {
      pad_mode = ov::op::PadMode::SYMMETRIC;
    } else {
      return errors::InvalidArgument(pad_mode_str,
                                     " is not an allowed padding mode.");
    }
  }

  // Set pads_begin & pads_end (from the pad_val_op)
  std::vector<int64> paddings;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &paddings));
  OVTF_VLOG(3) << op->name() << " pads {" << ngraph::join(paddings) << "}";
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
      op->name(), ov::element::i64, ov::Shape{pad_begin.size()}, pad_begin);
  auto pads_end_node = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{pad_end.size()}, pad_end);

  // Create final Op
  result_pad_op =
      ConstructNgNode<opset::Pad>(op->name(), ng_input, pads_begin_node,
                                  pads_end_node, pad_val_op, pad_mode);

  SaveNgOp(ng_op_map, op->name(), result_pad_op);
  return Status::OK();
}

static Status TranslateRangeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_start, ng_stop, ng_step;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_start, ng_stop, ng_step));

  ov::element::Type out_type;
  TF_RETURN_IF_ERROR(
      util::TFDataTypeToNGraphElementType(op->output_type(0), &out_type));

  auto ng_range = ConstructNgNode<opset::Range>(op->name(), ng_start, ng_stop,
                                                ng_step, out_type);
  SaveNgOp(ng_op_map, op->name(), ng_range);
  return Status::OK();
}

static Status TranslateRankOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  auto input_rank =
      static_cast<int>(ng_input.get_partial_shape().rank().get_length());

  auto ng_rank = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i32, ov::Shape(),
      std::vector<int>({input_rank}));

  SaveNgOp(ng_op_map, op->name(), ng_rank);
  return Status::OK();
}

static Status TranslateReciprocalOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ov::Output<ov::Node> n) {
        // Create a constant tensor populated with the value -1.
        // (1/x = x^(-1))
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> constant_values(ov::shape_size(shape), "-1");
        auto ng_exponent = ConstructNgNode<opset::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -1.
        return ConstructNgNode<opset::Power>(op->name(), n, ng_exponent);
      });
}

static Status TranslateRelu6Op(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
  auto ng_input_rank = ng_input.get_partial_shape().rank().get_length();
  std::string device;
  // Enable transpose before and after only for CPU device
  Status exec_status = BackendManager::GetBackendName(device);
  if (exec_status != Status::OK()) {
    throw runtime_error(exec_status.error_message());
  }
  if (device == "CPU") {
    if (ng_input_rank == 4) Transpose<0, 3, 1, 2>(ng_input);
  }
  auto ng_output = ConstructNgNode<opset::Clamp>(op->name(), ng_input, 0, 6);
  if (device == "CPU") {
    if (ng_input_rank == 4) Transpose<0, 2, 3, 1>(ng_output);
  }
  SaveNgOp(ng_op_map, op->name(), ng_output);

  return Status::OK();
}

static Status TranslateReshapeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_shape_op;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_shape_op));

  std::vector<int64> shape;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &shape));

  OVTF_VLOG(3) << "Requested result shape: " << ngraph::join(shape);

  auto ng_shape = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{shape.size()}, shape);
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Reshape>(
                                      op->name(), ng_input, ng_shape, false));
  return Status::OK();
}

static Status TranslateRoundOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  // using default round mode "half_to_even" in openvino,
  // as TF has only that mode
  opset::Round::RoundMode round_mode = opset::Round::RoundMode::HALF_TO_EVEN;
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Round>(op->name(), ng_input, round_mode));
  return Status::OK();
}

static Status TranslateResizeBilinearOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_inp, ng_inp_sizes;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp, ng_inp_sizes));

  // Get Interpolate attributes
  using InterpolateV4Attrs = opset::Interpolate::InterpolateAttrs;
  InterpolateV4Attrs interpolate_attrs;
  interpolate_attrs.mode = opset::Interpolate::InterpolateMode::LINEAR;
  interpolate_attrs.shape_calculation_mode =
      opset::Interpolate::ShapeCalcMode::SIZES;
  bool align_corners = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "align_corners", &align_corners));
  if (align_corners)
    interpolate_attrs.coordinate_transformation_mode =
        opset::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;

  auto input_shape = ng_inp.get_shape();
  std::vector<uint64_t> spatial_shape = {input_shape[1], input_shape[2]};
  auto ng_spatial_shape = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i32, ov::Shape{2}, spatial_shape);
  auto ng_input_shape = ConstructNgNode<opset::Convert>(
      op->name(), ng_spatial_shape, ov::element::f32);
  auto ng_sizes = ConstructNgNode<opset::Convert>(op->name(), ng_inp_sizes,
                                                  ov::element::f32);
  auto ng_scales =
      ConstructNgNode<opset::Divide>(op->name(), ng_sizes, ng_input_shape);
  auto ng_axes = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i32, ov::Shape{2}, std::vector<int>({2, 3}));

  Transpose<0, 3, 1, 2>(ng_inp);
  auto ng_output = ConstructNgNode<opset::Interpolate>(
      op->name(), ng_inp, ng_inp_sizes, ng_scales, ng_axes, interpolate_attrs);
  Transpose<0, 2, 3, 1>(ng_output);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateResizeNearestNeighborOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_inp, ng_inp_sizes;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp, ng_inp_sizes));

  opset::Interpolate::InterpolateAttrs interpolate_attrs;
  interpolate_attrs.mode = opset::Interpolate::InterpolateMode::NEAREST;
  interpolate_attrs.shape_calculation_mode =
      opset::Interpolate::ShapeCalcMode::SIZES;
  bool align_corners = false;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "align_corners", &align_corners));
  if (align_corners) {
    interpolate_attrs.coordinate_transformation_mode =
        opset::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
  }
  interpolate_attrs.nearest_mode =
      opset::Interpolate::NearestMode::ROUND_PREFER_FLOOR;

  auto input_shape = ng_inp.get_shape();
  std::vector<uint64_t> spatial_shape = {input_shape[1], input_shape[2]};
  auto ng_spatial_shape = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i32, ov::Shape{2}, spatial_shape);
  auto ng_input_shape = ConstructNgNode<opset::Convert>(
      op->name(), ng_spatial_shape, ov::element::f32);
  auto ng_sizes = ConstructNgNode<opset::Convert>(op->name(), ng_inp_sizes,
                                                  ov::element::f32);
  auto ng_scales =
      ConstructNgNode<opset::Divide>(op->name(), ng_sizes, ng_input_shape);
  auto ng_axes = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i32, ov::Shape{2}, std::vector<int>({2, 3}));

  Transpose<0, 3, 1, 2>(ng_inp);
  auto ng_output = ConstructNgNode<opset::Interpolate>(
      op->name(), ng_inp, ng_inp_sizes, ng_scales, ng_axes, interpolate_attrs);
  Transpose<0, 2, 3, 1>(ng_output);
  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

static Status TranslateReverseOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_reversed_axis;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_reversed_axis));
  ov::op::v1::Reverse::Mode mode = ov::op::v1::Reverse::Mode::INDEX;
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<ov::op::v1::Reverse>(op->name(), ng_input,
                                                ng_reversed_axis, mode));
  return Status::OK();
}

static Status TranslateRsqrtOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](ov::Output<ov::Node> n) {
        // Create a constant tensor populated with the value -1/2.
        // (1/sqrt(x) = x^(-1/2))
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> constant_values(ov::shape_size(shape), "-0.5");
        auto ng_exponent = ConstructNgNode<opset::Constant>(
            op->name(), et, shape, constant_values);

        // Raise each element of the input to the power -0.5.
        return ConstructNgNode<opset::Power>(op->name(), n, ng_exponent);
      });
}

static Status TranslateScatterNdOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input_indices, ng_updates, ng_shape;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_input_indices, ng_updates, ng_shape));

  std::vector<size_t> shape;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &shape));

  auto ng_input = ConstructNgNode<opset::Constant>(
      op->name(), ng_updates.get_element_type(), ov::Shape(shape), 0);

  auto scatternd_op = ConstructNgNode<opset::ScatterNDUpdate>(
      op->name(), ng_input, ng_input_indices, ng_updates);

  SaveNgOp(ng_op_map, op->name(), scatternd_op);
  return Status::OK();
}

static Status TranslateShapeOp(const Node* op,
                               const std::vector<const Tensor*>&,
                               Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  ov::element::Type type;
  TF_RETURN_IF_ERROR(util::TFDataTypeToNGraphElementType(dtype, &type));

  // default output_type = element::i64
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::ShapeOf>(op->name(), ng_input, type));
  return Status::OK();
}

static Status TranslateSizeOp(const Node* op, const std::vector<const Tensor*>&,
                              Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

  // Size has an attribute to specify output, int32 or int64
  ov::element::Type type;
  TF_RETURN_IF_ERROR(util::TFDataTypeToNGraphElementType(dtype, &type));

  ov::Output<ov::Node> ng_result;
  if (ng_input.get_partial_shape().is_static()) {
    auto ng_input_shape = ng_input.get_shape();
    int64 result = 1;
    for (auto dim : ng_input_shape) {
      result *= dim;

      // make a scalar with value equals to result
      ng_result = ConstructNgNode<opset::Constant>(
          op->name(), type, ov::Shape(0), std::vector<int64>({result}));
    }
  } else {
    auto shape_of = ConstructNgNode<opset::ShapeOf>(op->name(), ng_input, type);
    auto axis = ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                                 ov::Shape{}, 0);
    ng_result = ConstructNgNode<opset::ReduceProd>(op->name(), shape_of, axis);
  }

  SaveNgOp(ng_op_map, op->name(), ng_result);
  return Status::OK();
}

static Status TranslateSliceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_begin, ng_size;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_begin, ng_size));

  std::vector<int64> begin_vec;
  std::vector<int64> size_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &begin_vec));
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 2, static_input_map, &size_vec));

  if (begin_vec.size() != size_vec.size())
    return errors::InvalidArgument(
        "Cannot translate slice op: size of begin = ", begin_vec.size(),
        ", size of size_vec = ", size_vec.size(), ". Expected them to match.");

  OVTF_VLOG(3) << "Begin input for Slice: " << ngraph::join(begin_vec);
  OVTF_VLOG(3) << "Size input for Slice: " << ngraph::join(size_vec);

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
      op->name(), ov::element::i64, ov::Shape{begin_vec.size()}, begin_vec);
  auto end = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{end_vec.size()}, end_vec);

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::StridedSlice>(op->name(), ng_input, begin,
                                                end, std::vector<int64_t>{},
                                                std::vector<int64_t>{}));
  return Status::OK();
}

static Status TranslateSoftmaxOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  auto rank = ng_input.get_partial_shape().rank().get_length();
  if (rank < 1) {
    return errors::InvalidArgument("TF Softmax logits must be >=1 dimension");
  }

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Softmax>(op->name(), ng_input, rank - 1));
  return Status::OK();
}

// TODO: Change the translation back to unary softplus
// after resolving mish fusion issue
static Status TranslateSoftPlusOp(const Node* op,
                                  const std::vector<const Tensor*>&,
                                  Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_inp;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp));
  auto exp = ConstructNgNode<opset::Exp>(op->name(), ng_inp);
  auto add_const = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::f32, ov::Shape{1}, 1);

  auto add = ConstructNgNode<opset::Add>(op->name(), exp, add_const);
  auto ng_output = ConstructNgNode<opset::Log>(op->name(), add);

  SaveNgOp(ng_op_map, op->name(), ng_output);
  return Status::OK();
}

// Translate SpaceToDepthOp
static Status TranslateSpaceToDepthOp(const Node* op,
                                      const std::vector<const Tensor*>&,
                                      Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
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

static Status TranslateSparseToDenseOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_indices, ng_dense_shape, ng_values, ng_zeros;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_indices, ng_dense_shape,
                                   ng_values, ng_zeros));
  // TODO: Check if validate indices needs any handling on OVTF side
  // bool validate_indices=true;
  // TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "validate_indices",
  // &validate_indices));

  // Broadcast Dense Matrix using input args of sparse tensor input
  auto ng_dense_tensor =
      ConstructNgNode<opset::Broadcast>(op->name(), ng_zeros, ng_dense_shape);

  // Scatter the values at the given indices
  auto ng_scatternd_op = ConstructNgNode<opset::ScatterNDUpdate>(
      op->name(), ng_dense_tensor, ng_indices, ng_values);

  SaveNgOp(ng_op_map, op->name(), ng_scatternd_op);
  return Status::OK();
}

static Status TranslateSplitOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, ng_input));
  // num_split : The number of ways to split. Must evenly divide
  // value.shape[split_dim]
  int32 num_split;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num_split", &num_split));

  auto rank = ng_input.get_partial_shape().rank().get_length();

  std::vector<int> split_dim_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(op, 0, static_input_map, &split_dim_vec));
  int split_dim = split_dim_vec[0] + (split_dim_vec[0] < 0 ? (int64)rank : 0);
  auto ng_split_dim = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::u64, ov::Shape{}, split_dim);
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
  ov::Output<ov::Node> ng_input, ng_split_length, ng_split_dim;

  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  ov::Shape shape = ng_input.get_shape();
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
  ng_split_dim = ConstructNgNode<opset::Constant>(op->name(), ov::element::i32,
                                                  ov::Shape{}, split_dim);

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
      op->name(), ov::element::i32, ov::Shape{split_lengths_vec.size()},
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
      op, static_input_map, ng_op_map, [&op](ov::Output<ov::Node> n) {
        return ConstructNgNode<opset::Multiply>(op->name(), n, n);
      });
}

static Status TranslateSqueezeOp(const Node* op,
                                 const std::vector<const Tensor*>&,
                                 Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
  size_t input_dims = ng_input.get_partial_shape().rank().get_length();

  std::vector<int32> tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "squeeze_dims", &tf_axis));

  // If input dimension is negative, make it positive
  for (size_t i = 0; i < tf_axis.size(); i++) {
    tf_axis[i] = tf_axis[i] < 0 ? (int32)(input_dims) + tf_axis[i] : tf_axis[i];
  }

  if (ng_input.get_partial_shape().is_static()) {
    // This conditional check is required only if the input shape is known
    if (input_dims > 0 && ng_input.get_shape()[0] == 0) {
      SaveNgOp(ng_op_map, op->name(),
               ConstructNgNode<opset::Constant>(
                   op->name(), ng_input.get_element_type(), ov::Shape{0},
                   std::vector<int>({0})));
      return Status::OK();
    }
  }
  auto ng_const = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i32, ov::Shape{tf_axis.size()}, tf_axis);

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Squeeze>(op->name(), ng_input, ng_const));
  return Status::OK();
}

static Status TranslateStridedSliceOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> input, begin, end, strides;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, input));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, begin));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 2, end));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 3, strides));

  int32 begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "begin_mask", &begin_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "end_mask", &end_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "new_axis_mask", &new_axis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "shrink_axis_mask", &shrink_axis_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ellipsis_mask", &ellipsis_mask));

  OVTF_VLOG(5) << "strided slice attributes: "
               << "  begin mask: " << begin_mask << "  end mask: " << end_mask
               << "  new axis mask: " << new_axis_mask
               << "  shrink axis mask: " << shrink_axis_mask
               << "  ellipsis mask: " << ellipsis_mask;

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

  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::StridedSlice>(
               op->name(), input, begin, end, strides, mask_to_vec(begin_mask),
               mask_to_vec(end_mask), mask_to_vec(new_axis_mask),
               mask_to_vec(shrink_axis_mask), mask_to_vec(ellipsis_mask)));
  return Status::OK();
}

static Status TranslateTileOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_multiples;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_multiples));

  std::vector<int64> multiples;
  TF_RETURN_IF_ERROR(GetStaticInputVector(op, 1, static_input_map, &multiples));

  auto ng_repeats = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{multiples.size()}, multiples);
  SaveNgOp(ng_op_map, op->name(),
           ConstructNgNode<opset::Tile>(op->name(), ng_input, ng_repeats));
  return Status::OK();
}

// Translate TopKV2 Op using ngraph core op TopK
static Status TranslateTopKV2Op(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_k;

  TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 1, ng_k));

  // ASSER(ng_input.get_partial_shape().rank().is_static(), "Input rank must be
  // static.");
  // TENSORFLOW_OP_VALIDATION(node,
  //                           input.get_partial_shape().rank().get_length() >=
  //                           1,
  //                           "Input rank must be greater than 0.");
  // axis along which to compute top k indices
  int64_t k_axis = ng_input.get_partial_shape().rank().get_length() - 1;
  bool sorted = true;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "sorted", &sorted));

  auto ng_result = std::make_shared<opset::TopK>(
      ng_input, ng_k, k_axis, opset::TopK::Mode::MAX,
      sorted ? opset::TopK::SortType::SORT_VALUES
             : opset::TopK::SortType::SORT_INDICES);

  ov::Output<ov::Node> ng_values = ng_result->output(0);
  Builder::SetTracingInfo(op->name(), ng_values);
  ov::Output<ov::Node> ng_indices = ng_result->output(1);
  Builder::SetTracingInfo(op->name(), ng_indices);

  SaveNgOp(ng_op_map, op->name(), ng_values);
  SaveNgOp(ng_op_map, op->name(), ng_indices);

  return Status::OK();
}

static Status TranslateTransposeOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_permutation;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_permutation));
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Transpose>(
                                      op->name(), ng_input, ng_permutation));
  return Status::OK();
}

static Status TranslateUnpackOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

  ov::Output<ov::Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
  int32 tf_axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
  int32 num_outputs;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num", &num_outputs));

  auto rank = ng_input.get_partial_shape().rank().get_length();
  // convert the negative unpack axis value to positive value
  if (tf_axis < 0) {
    tf_axis = rank + tf_axis;
  }
  for (int i = 0; i < num_outputs; ++i) {
    std::vector<int64_t> begin(rank, 0);
    std::vector<int64_t> end(rank, 0);
    CHECK(tf_axis >= 0);
    begin[tf_axis] = i;
    end[tf_axis] = i + 1;
    auto ng_begin = ConstructNgNode<opset::Constant>(
        op->name(), ov::element::i64, ov::Shape{begin.size()}, begin);
    auto ng_end = ConstructNgNode<opset::Constant>(op->name(), ov::element::i64,
                                                   ov::Shape{end.size()}, end);
    std::vector<int64_t> begin_mask(rank, 1);
    begin_mask[tf_axis] = 0;
    std::vector<int64_t> end_mask(rank, 1);
    end_mask[tf_axis] = 0;
    std::vector<int64_t> new_axis_mask(rank, 0);
    std::vector<int64_t> shrink_axis_mask(rank, 0);
    auto slice = ConstructNgNode<opset::StridedSlice>(
        op->name(), ng_input, ng_begin, ng_end, begin_mask, end_mask,
        new_axis_mask, shrink_axis_mask);
    auto squeeze_axis = ConstructNgNode<opset::Constant>(
        op->name(), ov::element::i32, ov::Shape{}, tf_axis);
    auto squeeze =
        ConstructNgNode<opset::Squeeze>(op->name(), slice, squeeze_axis);
    SaveNgOp(ng_op_map, op->name(), squeeze);
  }
  return Status::OK();
}

static Status TranslateXdivyOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_x, ng_y;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_x, ng_y));
  auto zero = ConstructNgNode<opset::Constant>(
      op->name(), ng_x.get_element_type(), ov::Shape{}, std::vector<int>({0}));
  auto x_is_zero = ConstructNgNode<opset::Equal>(op->name(), ng_x, zero);
  auto ng_xdivy = ConstructNgNode<opset::Divide>(op->name(), ng_x, ng_y);
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Select>(
                                      op->name(), x_is_zero, ng_x, ng_xdivy));
  return Status::OK();
}

static Status TranslateSelectOp(const Node* op,
                                const std::vector<const Tensor*>&,
                                Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_cond, ng_input1, ng_input2;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_cond, ng_input1, ng_input2));
  // condition could be of same shape as inputs, or scalar, or it shoud be of
  // rank 1 and the dimension must match the first dimension of the inputs
  auto cond_rank = ng_cond.get_partial_shape().rank().get_length();
  // input1 and input2 are of same shape
  auto x_rank = ng_input1.get_partial_shape().rank().get_length();

  ov::Output<ov::Node> ng_select;
  // if condition is of rank 1, the broadcasting of condition to input shape
  // will not give correct output. Condition needs to reshaped to same rank as
  // of input with all other dimensions as 1.
  if (cond_rank == 1) {
    // Translation will not work with dynamic shape
    auto cond_shape = ng_cond.get_shape();
    auto augment_shape = ov::Shape(x_rank, 1);
    augment_shape.at(0) = ov::shape_size(cond_shape);
    auto ng_augmented_cond_shape = ConstructNgNode<opset::Constant>(
        op->name(), ov::element::i64, ov::Shape{augment_shape.size()},
        augment_shape);
    auto ng_augmented_cond = ConstructNgNode<opset::Reshape>(
        op->name(), ng_cond, ng_augmented_cond_shape, false);
    ng_select = ConstructNgNode<opset::Select>(op->name(), ng_augmented_cond,
                                               ng_input1, ng_input2);
  } else {
    // if condition is scalar, other thank rank 1, or condition and input are of
    // same shape (internally condition would be broadcasted to same shape as
    // that of input), in this case select op would work element wise
    ng_select = ConstructNgNode<opset::Select>(op->name(), ng_cond, ng_input1,
                                               ng_input2);
  }
  SaveNgOp(ng_op_map, op->name(), ng_select);
  return Status::OK();
}

static Status TranslateWhereOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_cond;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_cond));
  auto non_zero = ConstructNgNode<opset::NonZero>(op->name(), ng_cond);
  auto transpose_order = ConstructNgNode<opset::Constant>(
      op->name(), ov::element::i64, ov::Shape{2}, std::vector<int64_t>({1, 0}));
  SaveNgOp(ng_op_map, op->name(), ConstructNgNode<opset::Transpose>(
                                      op->name(), non_zero, transpose_order));
  return Status::OK();
}

static Status TranslateZerosLikeOp(const Node* op,
                                   const std::vector<const Tensor*>&,
                                   Builder::OpMap& ng_op_map) {
  ov::Output<ov::Node> ng_input, ng_result;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  if (ng_input.get_partial_shape().is_static()) {
    ov::Shape input_shape = ng_input.get_shape();
    std::vector<std::string> const_values(ov::shape_size(input_shape), "0");
    ng_result = ConstructNgNode<opset::Constant>(
        op->name(), ng_input.get_element_type(), input_shape, const_values);
  } else {
    auto input_shape = ConstructNgNode<opset::ShapeOf>(op->name(), ng_input);
    auto zero = ConstructNgNode<opset::Constant>(
        op->name(), ng_input.get_element_type(), ov::Shape{1}, 0);
    ng_result =
        ConstructNgNode<opset::Broadcast>(op->name(), zero, input_shape);
  }
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
        {"AvgPool", TranslateAvgPoolOp<2>},
        {"AvgPool3D", TranslateAvgPoolOp<3>},
        {"BatchMatMul", TranslateBatchMatMulOp},
        {"BatchMatMulV2", TranslateBatchMatMulOp},
        {"BatchToSpaceND", TranslateBatchNDAndSpaceNDOp},
        {"BiasAdd", TranslateBiasAddOp},
        {"Cast", TranslateCastOp},
        {"Ceil", TranslateUnaryOp<opset::Ceiling>},
        {"ConcatV2", TranslateConcatV2Op},
        {"Const", TranslateConstOp},
        {"Conv2D", TranslateConv2DOp},
        {"Conv2DBackpropInput", TranslateConv2DBackpropInputOp},
        {"Conv3D", TranslateConv3DOp},
        {"Conv3DBackpropInputV2", TranslateConv3DBackpropInputV2Op},
        {"Cos", TranslateUnaryOp<opset::Cos>},
        {"Cosh", TranslateUnaryOp<opset::Cosh>},
        {"CropAndResize", TranslateCropAndResizeOp},
        {"CTCGreedyDecoder", TranslateCTCGreedyDecoderOp},
        {"Cumsum", TranslateCumsumOp},
        {"DepthToSpace", TranslateDepthToSpaceOp},
        {"DepthwiseConv2dNative", TranslateDepthwiseConv2dNativeOp},
        {"Elu", TranslateEluOp},
        {"Equal", TranslateBinaryOp<opset::Equal>},
        {"Exp", TranslateUnaryOp<opset::Exp>},
        {"ExpandDims", TranslateExpandDimsOp},
        {"FakeQuantWithMinMaxVars", TranslateFakeQuantWithMinMaxVarsOp},
        {"Fill", TranslateFillOp},
        {"Floor", TranslateUnaryOp<opset::Floor>},
        {"FloorDiv", TranslateFloorDivOp},
        {"FloorMod", TranslateBinaryOp<opset::FloorMod>},
        {"FusedBatchNorm", TranslateFusedBatchNormOp},
        {"FusedBatchNormV2", TranslateFusedBatchNormOp},
        {"FusedBatchNormV3", TranslateFusedBatchNormOp},
        // FusedCTCGreedyDecoder combines CTCGreedyDecoder and the following
        // SparseToDense op
        {"FusedCTCGreedyDecoder", TranslateFusedCTCGreedyDecoder},
        {"Gather", TranslateGatherOp},
        {"GatherV2", TranslateGatherV2Op},
        {"GatherNd", TranslateGatherNdOp},
        {"_FusedBatchNormEx", TranslateFusedBatchNormOp},
        {"_FusedConv2D", TranslateFusedConv2DOp},
        {"_FusedDepthwiseConv2dNative", TranslateFusedDepthwiseConv2dNativeOp},
        {"_FusedMatMul", TranslateFusedMatMulOp},
        {"Greater", TranslateBinaryOp<opset::Greater>},
        {"GreaterEqual", TranslateBinaryOp<opset::GreaterEqual>},
        {"Identity", TranslateIdentityOp},
        {"IsFinite", TranslateIsFiniteOp},
        {"L2Loss", TranslateL2LossOp},
        {"LogSoftmax", TranslateLogSoftmaxOp},
        {"LeakyRelu", TranslateLeakyReluOp},
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
        {"_MklSwish", TranslateMklSwishOp},
        {"NonMaxSuppression", TranslateNonMaxSuppressionOp},
        {"NonMaxSuppressionV2", TranslateNonMaxSuppressionOp},
        {"NonMaxSuppressionV3", TranslateNonMaxSuppressionOp},
        {"NonMaxSuppressionV4", TranslateNonMaxSuppressionOp},
        {"NonMaxSuppressionV5", TranslateNonMaxSuppressionOp},
        {"Mean", TranslateDirectReduceOp<opset::ReduceMean>},
        {"Min", TranslateDirectReduceOp<opset::ReduceMin>},
        {"Minimum", TranslateBinaryOp<opset::Minimum>},
        {"MirrorPad", TranslatePadOp},
        {"Mul", TranslateBinaryOp<opset::Multiply>},
        {"Mod", TranslateBinaryOp<opset::Mod>},
        {"Neg", TranslateUnaryOp<opset::Negative>},
        {"NotEqual", TranslateBinaryOp<opset::NotEqual>},
        // Do nothing! NoOps sometimes get placed on OpenVINO Model for
        // bureaucratic
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
        {"Range", TranslateRangeOp},
        {"Rank", TranslateRankOp},
        {"RealDiv", TranslateBinaryOp<opset::Divide>},
        {"Reciprocal", TranslateReciprocalOp},
        {"Relu", TranslateUnaryOp<opset::Relu>},
        {"Relu6", TranslateRelu6Op},
        {"Reshape", TranslateReshapeOp},
        {"Round", TranslateRoundOp},
        {"ResizeBilinear", TranslateResizeBilinearOp},
        {"ResizeNearestNeighbor", TranslateResizeNearestNeighborOp},
        {"Reverse", TranslateReverseOp},
        {"ReverseV2", TranslateReverseOp},
        {"Rsqrt", TranslateRsqrtOp},
        {"ScatterNd", TranslateScatterNdOp},
        {"Select", TranslateSelectOp},
        {"Shape", TranslateShapeOp},
        {"Sigmoid", TranslateUnaryOp<opset::Sigmoid>},
        {"Sin", TranslateUnaryOp<opset::Sin>},
        {"Sinh", TranslateUnaryOp<opset::Sinh>},
        {"Size", TranslateSizeOp},
        {"Sign", TranslateUnaryOp<opset::Sign>},
        {"Slice", TranslateSliceOp},
        {"Snapshot", TranslateIdentityOp},
        {"Softmax", TranslateSoftmaxOp},
        {"Softplus", TranslateSoftPlusOp},
        {"SpaceToBatchND", TranslateBatchNDAndSpaceNDOp},
        {"SpaceToDepth", TranslateSpaceToDepthOp},
        {"SparseToDense", TranslateSparseToDenseOp},
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
        {"Where", TranslateWhereOp},
        {"Xdivy", TranslateXdivyOp},
        {"ZerosLike", TranslateZerosLikeOp}};

static bool UseFusedCTCGreedyDecoder(const Node* op) {
  // we support fusing only when CTCGreedyDecoder is followed by SparseToDense
  // op resulting in two edges
  // any op consuming the log probabilities from CTCGreedyDecoder will result
  // in an additional edge, hence producing a total of three edges.
  if (!(op->out_edges().size() == 2 || op->out_edges().size() == 3)) {
    return false;
  }
  auto edges =
      std::vector<const Edge*>(op->out_edges().begin(), op->out_edges().end());

  // check port mappings
  if (edges.at(op->out_edges().size() - 2)->dst()->type_string() ==
          "SparseToDense" &&
      edges.at(op->out_edges().size() - 1)->dst()->type_string() ==
          "SparseToDense" &&
      edges.at(op->out_edges().size() - 2)->src_output() == 0 &&
      edges.at(op->out_edges().size() - 2)->dst_input() == 0 &&
      edges.at(op->out_edges().size() - 1)->src_output() == 1 &&
      edges.at(op->out_edges().size() - 1)->dst_input() == 2) {
    return true;
  }
  return false;
}

Status Builder::TranslateGraph(
    const std::vector<TensorShape>& inputs,
    const std::vector<const Tensor*>& static_input_map,
    const Graph* input_graph, const string name,
    shared_ptr<ov::Model>& ng_function) {
  ov::ResultVector ng_result_list;
  std::vector<Tensor> tf_input_tensors;
  TranslateGraph(inputs, static_input_map, input_graph, name, ng_function,
                 ng_result_list, tf_input_tensors);
  return Status::OK();
}

Status Builder::TranslateGraph(
    const std::vector<TensorShape>& inputs,
    const std::vector<const Tensor*>& static_input_map,
    const Graph* input_graph, const string name,
    shared_ptr<ov::Model>& ng_function, ov::ResultVector& zero_dim_outputs,
    const std::vector<Tensor>& tf_input_tensors) {
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
          "Encountered a control flow op in the openvino_tensorflow: ",
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
  // vector of generated OpenVINO Output<Node>.
  //
  Builder::OpMap ng_op_map;

  //
  // Populate the parameter list, and also put parameters into the op map.
  //
  ov::ParameterVector ng_parameter_list(tf_params.size());
  ov::ParameterVector ng_func_parameter_list;
  ng_func_parameter_list.reserve(tf_params.size());
  // Variables to constant conversion is disabled by default
  bool convert_var_const = false;
  const char* convert_var_const_env =
      std::getenv("OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS");

  if (!(convert_var_const_env == nullptr)) {
    // Array should have either "0" or "1"
    char env_value = convert_var_const_env[0];
    if (env_value == '1') {
      OVTF_VLOG(1) << "OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS is Enabled ";
      convert_var_const = true;
    }
  }

  for (auto parm : tf_params) {
    DataType dtype;
    if (GetNodeAttr(parm->attrs(), "T", &dtype) != Status::OK()) {
      return errors::InvalidArgument("No data type defined for _Arg");
    }
    int64_t index;
    if (GetNodeAttr(parm->attrs(), "index", &index) != Status::OK()) {
      return errors::InvalidArgument("No index defined for _Arg");
    }

    ov::element::Type ng_et;
    TF_RETURN_IF_ERROR(util::TFDataTypeToNGraphElementType(dtype, &ng_et));

    ov::Shape ng_shape;
    TF_RETURN_IF_ERROR(
        util::TFTensorShapeToNGraphShape(inputs[index], &ng_shape));

    string prov_tag;
    GetNodeAttr(parm->attrs(), "_prov_tag", &prov_tag);
    auto ng_param =
        ConstructNgNode<opset::Parameter>(prov_tag, ng_et, ng_shape);

    auto ng_shape_check = [ng_shape]() {
      if (ng_shape.size() > 0) {
        for (auto val : ng_shape) {
          if (val == 0) return true;
        }
      }
      return false;
    };

    bool is_variable = false;
    if (convert_var_const && !tf_input_tensors.empty()) {
      try {
        GetNodeAttr(parm->attrs(), "_is_variable", &is_variable);
      } catch (const std::exception&) {
        OVTF_VLOG(1) << "Parameter " << parm->name() << " is not a variable";
      }
    }

    if (ng_shape_check()) {
      std::vector<std::string> constant_values(ov::shape_size(ng_shape), "0");
      auto ng_const_input = ConstructNgNode<opset::Constant>(
          prov_tag, ng_et, ng_shape, constant_values);
      SaveNgOp(ng_op_map, parm->name(), ng_const_input);
    } else {
      if (is_variable) {
        ov::Output<ov::Node> ng_const_input;
        const Tensor input_tensor = tf_input_tensors[index];
        switch (dtype) {
          case DT_FLOAT:
            MakeConstOpForParam<float>(input_tensor, prov_tag, ng_et, ng_shape,
                                       ng_const_input);
            break;
          case DT_DOUBLE:
            MakeConstOpForParam<double>(input_tensor, prov_tag, ng_et, ng_shape,
                                        ng_const_input);
            break;
          case DT_INT8:
            MakeConstOpForParam<int8>(input_tensor, prov_tag, ng_et, ng_shape,
                                      ng_const_input);
            break;
          case DT_INT16:
            MakeConstOpForParam<int16>(input_tensor, prov_tag, ng_et, ng_shape,
                                       ng_const_input);
            break;
          case DT_INT32:
            MakeConstOpForParam<int32>(input_tensor, prov_tag, ng_et, ng_shape,
                                       ng_const_input);
            break;
          case DT_INT64:
            MakeConstOpForParam<int64>(input_tensor, prov_tag, ng_et, ng_shape,
                                       ng_const_input);
            break;
          case DT_UINT8:
            MakeConstOpForParam<uint8>(input_tensor, prov_tag, ng_et, ng_shape,
                                       ng_const_input);
            break;
          case DT_UINT16:
            MakeConstOpForParam<uint16>(input_tensor, prov_tag, ng_et, ng_shape,
                                        ng_const_input);
            break;
          case DT_UINT32:
            MakeConstOpForParam<uint32>(input_tensor, prov_tag, ng_et, ng_shape,
                                        ng_const_input);
            break;
          case DT_UINT64:
            MakeConstOpForParam<uint64>(input_tensor, prov_tag, ng_et, ng_shape,
                                        ng_const_input);
            break;
          case DT_BOOL:
            MakeConstOpForParam<bool>(input_tensor, prov_tag, ng_et, ng_shape,
                                      ng_const_input);
            break;
          default:
            return errors::Internal("Tensor has element type ",
                                    DataType_Name(dtype),
                                    "; don't know how to convert");
        }

        SaveNgOp(ng_op_map, parm->name(), ng_const_input);
      } else
        SaveNgOp(ng_op_map, parm->name(), ng_param);
    }
    CHECK(index >= 0);
    ng_parameter_list[index] =
        ov::as_type_ptr<opset::Parameter>(ng_param.get_node_shared_ptr());
    ng_parameter_list[index]->get_rt_info().insert({"index", ov::Any(index)});
  }

  //
  // Now create the OpenVINO ops from TensorFlow ops.
  //
  for (auto op : tf_ops) {
    auto op_type = op->type_string();

    if (HasNodeAttr(op->def(), "_ovtf_translated")) {
      OVTF_VLOG(5) << "Skipping op " << op->name() << " which is " << op_type
                   << " since it has been processed";
      continue;
    }

    OVTF_VLOG(2) << "Constructing op " << op->name() << " which is " << op_type;

    // FusedCTCGreedyDecoder
    if (op_type == "CTCGreedyDecoder") {
      if (UseFusedCTCGreedyDecoder(op)) {
        OVTF_VLOG(5) << "Using Fused translation for CTCGreedyDecoder "
                     << "and SparseToDense Ops";
        auto edges = std::vector<const Edge*>(op->out_edges().begin(),
                                              op->out_edges().end());
        edges.at(1)->dst()->AddAttr("_ovtf_translated", true);
        op_type = "FusedCTCGreedyDecoder";
      } else
        OVTF_VLOG(5) << "Not using Fused translation for "
                     << "CTCGreedyDecoder and SparseToDense Ops";
    }

    const function<Status(const Node*, const std::vector<const Tensor*>&,
                          Builder::OpMap&)>* op_fun;

    try {
      op_fun = &(TRANSLATE_OP_MAP.at(op_type));
    } catch (const std::out_of_range&) {
      // -----------------------------
      // Catch-all for unsupported ops
      // -----------------------------
      OVTF_VLOG(3) << "No translation handler registered for op: " << op->name()
                   << " (" << op_type << ")";
      OVTF_VLOG(3) << op->def().DebugString();
      return errors::InvalidArgument(
          "No translation handler registered for op: ", op->name(), " (",
          op_type, ")\n", op->def().DebugString());
    }

    try {
      TF_RETURN_IF_ERROR((*op_fun)(op, static_input_map, ng_op_map));
    } catch (const std::exception& e) {
      return errors::Internal("Unhandled exception in op handler: ", op->name(),
                              " (", op_type, ")\n", op->def().DebugString(),
                              "\n", "what(): ", e.what());
    }
  }

  //
  // Populate the result list.
  //
  ov::ResultVector ng_result_list;
  ng_result_list.resize(tf_ret_vals.size());
  ov::ResultVector ng_func_result_list;
  ng_func_result_list.reserve(tf_params.size());

  for (auto n : tf_ret_vals) {
    // Make sure that this _Retval only has one input node.
    if (n->num_inputs() != 1) {
      return errors::InvalidArgument("_Retval has ", n->num_inputs(),
                                     " inputs, should have 1");
    }

    int64_t index;
    if (GetNodeAttr(n->attrs(), "index", &index) != Status::OK()) {
      return errors::InvalidArgument("No index defined for _Retval");
    }

    CHECK(index >= 0);
    ov::Output<ov::Node> result;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, n, 0, result));
    auto ng_result = ConstructNgNode<opset::Result>(n->name(), result);
    ng_result_list[index] =
        ov::as_type_ptr<opset::Result>(ng_result.get_node_shared_ptr());
    ng_result_list[index]->get_rt_info().insert({"index", ov::Any(index)});
  }

  auto param_dim_check = [ng_parameter_list](int i) {
    auto param_shape_list = ng_parameter_list[i]->get_shape();
    for (auto dim : param_shape_list) {
      if (dim == 0) return true;
    }
    return false;
  };

  for (int i = 0; i < ng_parameter_list.size(); i++) {
    if (!(ng_parameter_list[i]->get_shape().size() > 0 && param_dim_check(i))) {
      ng_func_parameter_list.push_back(ng_parameter_list[i]);
    }
  }

  // Get the result nodes with valid dim values
  auto result_dim_check = [ng_result_list](int i) {
    auto res_shape_list = ng_result_list[i]->get_shape();
    for (auto dim : res_shape_list) {
      if (dim == 0) return true;
    }
    return false;
  };

  for (int i = 0; i < ng_result_list.size(); i++) {
    if (ng_result_list[i]->is_dynamic() ||
        !(ng_result_list[i]->get_shape().size() > 0 && result_dim_check(i))) {
      ng_func_result_list.push_back(ng_result_list[i]);
    } else {
      zero_dim_outputs.push_back(ng_result_list[i]);
    }
  }

  //
  // Create the OpenVINO Model.
  //
  try {
    ng_function = make_shared<ov::Model>(ng_func_result_list,
                                         ng_func_parameter_list, name);
  } catch (const std::exception& exp) {
    return errors::Internal("Failed to create OpenVINO Model for " + name +
                            ": " + string(exp.what()));
  }

  //
  // Apply additional passes on the OpenVINO Model here.
  //
  {
    ov::pass::Manager passes;
    if (util::GetEnv("OPENVINO_TF_CONSTANT_FOLDING") == "1") {
      passes.register_pass<ov::pass::ConstantFolding>();
    }
    if (util::GetEnv("OPENVINO_TF_TRANSPOSE_SINKING") != "0") {
      passes.register_pass<pass::TransposeSinking>();
    }
    passes.run_passes(ng_function);
  }
  OVTF_VLOG(5) << "Done with passes";
  //
  // Request row-major layout on results.
  //
  NGRAPH_SUPPRESS_DEPRECATED_START
  for (auto result : ng_function->get_results()) {
    result->set_needs_default_layout(true);
  }
  NGRAPH_SUPPRESS_DEPRECATED_END
  return Status::OK();
}

std::mutex Builder::m_translate_lock_;

// Initialize the lib path as empty string
std::string Builder::m_tf_conversion_extensions_lib_path = "";

void Builder::SetLibPath(const std::string& tf_conversion_extensions_so_path) {
  // TODO: Add check if the lib_path exists, otherwise throw error
  m_tf_conversion_extensions_lib_path = tf_conversion_extensions_so_path;
}
ov::frontend::FrontEnd::Ptr Builder::m_frontend_ptr =
    std::make_shared<ov::frontend::tensorflow::FrontEnd>();

Status Builder::TranslateGraphWithTFFE(
    const std::vector<TensorShape>& inputs, const Graph* input_graph,
    const string name, std::shared_ptr<ov::Model>& ng_function,
    ov::ResultVector& zero_dim_outputs,
    const std::vector<Tensor>& tf_input_tensors) {
  std::lock_guard<std::mutex> lock(m_translate_lock_);
  vector<Node*> ordered;
  GetReversePostOrder(*input_graph, &ordered, NodeComparatorName());

  // Assign attributes for constant inputs
  for (const auto n : ordered) {
    if (n->IsSink() || n->IsSource()) {
      continue;
    }

    if (n->IsControlFlow()) {
      return errors::Unimplemented(
          "Encountered a control flow op in the openvino_tensorflow: ",
          n->DebugString());
    }

    if (n->IsArg()) {
      bool static_input = false;
      try {
        if (Status::OK() !=
            GetNodeAttr(n->attrs(), "_static_input", &static_input)) {
          n->AddAttr("_static_input", false);
          static_input = false;
        }
      } catch (const std::exception&) {
        // TODO: What are the expected exceptions?
        // Non-existing attribute names are already handled above.
        OVTF_VLOG(1) << "Parameter " << n->name()
                     << " is not a static input to any node";
      }
      if (util::GetEnv("OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS") != "0") {
        bool is_variable = false;
        try {
          GetNodeAttr(n->attrs(), "_is_variable", &is_variable);
        } catch (const std::exception&) {
          OVTF_VLOG(1) << "Parameter " << n->name() << " is not a variable";
        }
        if (is_variable) n->AddAttr("_static_input", true);
        static_input |= is_variable;
      }
      if (static_input) {
        DataType dtype;
        if (GetNodeAttr(n->attrs(), "T", &dtype) != Status::OK()) {
          return errors::InvalidArgument("No data type defined for _Arg");
        }
        int64_t index;
        if (GetNodeAttr(n->attrs(), "index", &index) != Status::OK()) {
          return errors::InvalidArgument("No index defined for _Arg");
        }
        const Tensor tensor = tf_input_tensors[index];
        n->AddAttr("_static_value", tensor);
        n->AddAttr("_static_dtype", dtype);
      }
      try {
        std::string prov_tag;
        if (Status::OK() != GetNodeAttr(n->attrs(), "_prov_tag", &prov_tag)) {
          // TODO: Assign a proper prov tag instead of an empty string.
          n->AddAttr("_prov_tag", prov_tag);
        }
      } catch (const std::exception&) {
        // TODO: What are the expected exceptions?
        // Non-existing attribute names are already handled above.
        OVTF_VLOG(1) << "Parameter " << n->name()
                     << " does not have a _prov_tag assigned";
      }
    }
  }

  std::shared_ptr<OVTFGraphIterator> giter =
      std::make_shared<OVTFGraphIterator>(ordered);

  ov::frontend::tensorflow::GraphIterator::Ptr gi_ptr = giter;
  ov::Any gany(gi_ptr);

  //std::vector<ov::Shape> indexed_shape;
  std::vector<ov::PartialShape> indexed_shape;
  indexed_shape.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    ov::Shape ng_shape;
    TF_RETURN_IF_ERROR(util::TFTensorShapeToNGraphShape(inputs[i], &ng_shape));
    indexed_shape.push_back(ng_shape);
  }

  // Add the OV extension lib
  static bool once = true;
  if (once) {
#ifdef _WIN32
#define EXT ".dll"
#elif __APPLE__
#define EXT ".dylib"
#else
#define EXT ".so"
#endif

    // This would set the conversion extension path for c++ library
    if (m_tf_conversion_extensions_lib_path.empty()) {
      std::string lib_path = "";
#ifdef _WIN32
#ifdef OVTF_INSTALL_LIB_DIR
      lib_path = OVTF_INSTALL_LIB_DIR;
#endif
#ifdef TF_CONVERSION_EXTENSIONS_MODULE_NAME
      lib_path = lib_path + "/" + TF_CONVERSION_EXTENSIONS_MODULE_NAME + EXT;
#endif
#else
      // Dynamically try to determine the location of
      // libtf_conversion_extensions.so / .dylib
      // using the identifier dummy function ptr of tf_ce_dll_id_
      // This way of identifying the library path works only on Linux and OSX
      // and *is required* whenever OVTF is used through the TF C++ API.
      // For the Python API, it is handled through an explicit SetLibPath call
      // during `import openvino_tensorflow`
      Dl_info dl_info;
      dladdr((void*)ov::frontend::tensorflow::op::tf_ce_dll_id_, &dl_info);
      lib_path = (std::string)dl_info.dli_fname;
#endif
      // If library is not found during add_extension(), dlopen throws an error
      SetLibPath(lib_path);
    }
    m_frontend_ptr->add_extension(m_tf_conversion_extensions_lib_path);
    once = false;
  }

  // _Arg implementation
  m_frontend_ptr->add_extension(
      std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
          "_Arg", [&indexed_shape](const ov::frontend::NodeContext& node)
                      -> ov::OutputVector {
            ov::Output<ov::Node> res;
            auto index = node.get_attribute<int64_t>("index");
            auto prov_tag = node.get_attribute<std::string>("_prov_tag");
            auto is_static_input = node.get_attribute<bool>("_static_input");
            if (is_static_input) {
              auto tensor = node.get_attribute<ov::Tensor>("_static_value");
              auto dtype =
                  node.get_attribute<ov::element::Type>("_static_dtype");
              FRONT_END_GENERAL_CHECK(dtype == tensor.get_element_type(),
                                      "_Arg has tensor with type different "
                                      "from _static_dtype attribute.");
              res = std::make_shared<ov::opset8::Constant>(
                  tensor.get_element_type(), tensor.get_shape(), tensor.data());
            } else {
              auto element_type = node.get_attribute<ov::element::Type>("T");
              auto shape = indexed_shape.at(index);
              if (BackendManager::DynamicShapesEnabled()) {
                auto dynamic_shape_support = node.get_attribute<bool>("_dynamic_shape");
                if (dynamic_shape_support) {
                  for (int d=0; d<shape.size(); d++) {
                      shape[d] = -1;
                  }
                }
              }
              res =
                  std::make_shared<ov::opset8::Parameter>(element_type, shape);
            }
            res.get_node_shared_ptr()->get_rt_info().insert(
                {"index", ov::Any(index)});
            res.get_node_shared_ptr()->get_rt_info().insert(
                {"_prov_tag", ov::Any(prov_tag)});
            return {res};
          }));

  // _Retval implementation
  m_frontend_ptr->add_extension(
      std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
          "_Retval", [&indexed_shape](const ov::frontend::NodeContext& node)
                         -> ov::OutputVector {
            if (node.get_input_size() != 1) {
              FRONT_END_GENERAL_CHECK(
                  false, "_Retval has " + to_string(node.get_input_size()) +
                             " inputs, should have 1");
            }

            auto index = node.get_attribute<int64_t>("index");
            auto res = make_shared<ov::op::v0::Result>(node.get_input(0));
            if (res == nullptr) {
              throw errors::Internal("Failed while converting op: _Retval");
            }
            res->get_rt_info().insert({"index", ov::Any(index)});
            return res->outputs();
          }));

  try {
    ov::frontend::InputModel::Ptr input_model = m_frontend_ptr->load(gany);
    ng_function = m_frontend_ptr->convert(input_model);
  } catch (const ov::NodeValidationFailure& exp) {
    // Treat NODE_VALIDATION_CHECK errors as InvalidArgument errors for proper
    // handling at TF
    // Workaround required for SplitVOp Tests
    return errors::InvalidArgument(
        "Frontend conversion error: NodeValidationFailure: " +
        string(exp.what()));
  } catch (const std::exception& exp) {
    return errors::Internal("Frontend conversion error: " + string(exp.what()));
  } catch (...) {
    return errors::Internal("Frontend conversion error");
  }

  // Get the result nodes with valid dim values
  auto ng_result_list = ng_function->get_results();
  auto result_dim_check = [ng_result_list](int i) {
    auto res_shape_list = ng_result_list[i]->get_shape();
    for (auto dim : res_shape_list) {
      if (dim == 0) return true;
    }
    return false;
  };

  ov::ResultVector ng_func_result_list;
  for (int i = 0; i < ng_result_list.size(); i++) {
    if (!(ng_result_list[i]->is_dynamic() ||
        !(ng_result_list[i]->get_shape().size() > 0 && result_dim_check(i)))) {
      zero_dim_outputs.push_back(ng_result_list[i]);
    }
  }

  ng_function->set_friendly_name(name);

  //
  // Apply additional passes on the nGraph function here.
  //
  {
    ov::pass::Manager passes;
    if (util::GetEnv("OPENVINO_TF_CONSTANT_FOLDING") == "1") {
      passes.register_pass<ov::pass::ConstantFolding>();
    }
    if (util::GetEnv("OPENVINO_TF_TRANSPOSE_SINKING") == "1") {
      passes.register_pass<pass::TransposeSinking>();
    }
    passes.run_passes(ng_function);
  }
  OVTF_VLOG(5) << "Done with passes";
  //
  // Request row-major layout on results.
  //
  for (auto result : ng_function->get_results()) {
    result->set_needs_default_layout(true);
  }
  OVTF_VLOG(5) << "Done with translations";

  return Status::OK();
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
