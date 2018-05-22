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
#include "ngraph_utils.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb_text.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ngraph_bridge {

// Helper for Builder::TranslateGraph ("Const" op)
template <typename T>
tf::Status MakeConstOp(tf::Node* op, ng::element::Type et,
                       std::shared_ptr<ng::Node>* ng_node) {
  vector<T> const_values;
  tf::TensorShapeProto shape_proto;

  TF_RETURN_IF_ERROR(
      ValuesFromConstNode<T>(op->def(), &shape_proto, &const_values));

  tf::TensorShape const_shape(shape_proto);
  ng::Shape ng_shape;
  TF_RETURN_IF_ERROR(TFTensorShapeToNGraphShape(const_shape, &ng_shape));

  *ng_node = make_shared<ng::op::Constant>(et, ng_shape, const_values);
  return tf::Status::OK();
}

template <typename T>
tf::Status TranslateBinOp(tf::Node* op, Builder::OpMap& ng_op_map) {
  if (op->num_inputs() != 2) {
    return tf::errors::InvalidArgument(
        "Number of inputs is not 2 for elementwise op");
  }
  tf::Node* tf_lhs;
  if (op->input_node(0, &tf_lhs) != tf::Status::OK()) {
    return tf::errors::InvalidArgument("Cannot get the input node 0");
  }

  tf::Node* tf_rhs;
  if (op->input_node(1, &tf_rhs) != tf::Status::OK()) {
    return tf::errors::InvalidArgument("Cannot get the input node 1");
  }

  auto ng_lhs = ng_op_map.find(tf_lhs->name())->second;
  auto ng_rhs = ng_op_map.find(tf_rhs->name())->second;

  // FIXME(amprocte): super-duper specific kludge to get this going...
  if (ng_rhs->get_shape().size() == 1 && ng_lhs->get_shape().size() == 2 &&
      ng_rhs->get_shape()[0] == ng_lhs->get_shape()[1]) {
    ng_rhs = make_shared<ngraph::op::Broadcast>(ng_rhs, ng_lhs->get_shape(),
                                                ngraph::AxisSet{0});
  }

  shared_ptr<ng::Node> ng_op;

  ng_op = make_shared<T>(ng_lhs, ng_rhs);

  // Add to the map
  ng_op_map[op->name()] = ng_op;
  return tf::Status::OK();
}

tf::Status Builder::TranslateGraph(const std::vector<tf::TensorShape>& inputs,
                                   const tf::Graph* input_graph,
                                   shared_ptr<ng::Function>& ng_function) {
  // Do a topological sort
  // GetReversePostOrder will give us topological sort.
  vector<tf::Node*> ordered;
  tf::GetReversePostOrder(*input_graph, &ordered);

  // Now start building the nGraph.
  vector<tf::Node*> tf_params;
  vector<tf::Node*> tf_consts;
  vector<tf::Node*> tf_ret_vals;
  vector<tf::Node*> tf_ops;

  for (auto n : ordered) {
    if (n->IsSink() || n->IsSource() || n->IsControlFlow()) {
      continue;
    }

    if (n->type_string() == "_Arg") {
      tf_params.push_back(n);
    } else if (n->type_string() == "_Retval") {
      tf_ret_vals.push_back(n);
    } else {
      tf_ops.push_back(n);
    }
  }

  Builder::OpMap ng_op_map;
  vector<shared_ptr<ng::op::Parameter>> ng_parameter_list(tf_params.size());
  for (auto parm : tf_params) {
    tf::DataType dtype;
    if (tf::GetNodeAttr(parm->attrs(), "T", &dtype) != tf::Status::OK()) {
      return tf::errors::InvalidArgument(
          "Error: No data type defined for _Arg!");
    }
    int index;
    if (tf::GetNodeAttr(parm->attrs(), "index", &index) != tf::Status::OK()) {
      return tf::errors::InvalidArgument("Error: No index defined for _Arg!");
    }

    const tf::TensorShape& tf_shape = inputs[index];
    // TODO: Convert tf::Type to ng::type
    ng::Shape ng_shape(tf_shape.dims());
    for (int i = 0; i < tf_shape.dims(); ++i) {
      ng_shape[i] = tf_shape.dim_size(i);
    }

    auto ng_param = make_shared<ng::op::Parameter>(ng::element::f32, ng_shape);
    ng_op_map[parm->name()] = ng_param;
    ng_parameter_list[index] = ng_param;
  }

  for (auto op : tf_ops) {
    //
    // Now create the ng ops from tf ops.
    //

    // -----
    // Const
    // -----
    if (op->type_string() == "Const") {
      tf::DataType dtype;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "dtype", &dtype));

      std::shared_ptr<ng::Node> ng_node;

      switch (dtype) {
        case tf::DataType::DT_FLOAT:
          TF_RETURN_IF_ERROR(
              MakeConstOp<float>(op, ng::element::f32, &ng_node));
          break;
        case tf::DataType::DT_DOUBLE:
          TF_RETURN_IF_ERROR(
              MakeConstOp<double>(op, ng::element::f64, &ng_node));
          break;
        case tf::DataType::DT_INT8:
          TF_RETURN_IF_ERROR(
              MakeConstOp<tf::int8>(op, ng::element::i8, &ng_node));
          break;
        case tf::DataType::DT_INT16:
          TF_RETURN_IF_ERROR(
              MakeConstOp<tf::int16>(op, ng::element::i16, &ng_node));
          break;
        case tf::DataType::DT_INT32:
          TF_RETURN_IF_ERROR(
              MakeConstOp<tf::int32>(op, ng::element::i32, &ng_node));
          break;
        case tf::DataType::DT_INT64:
          TF_RETURN_IF_ERROR(
              MakeConstOp<tf::int64>(op, ng::element::i64, &ng_node));
          break;
        case tf::DataType::DT_UINT8:
          TF_RETURN_IF_ERROR(
              MakeConstOp<tf::uint8>(op, ng::element::u8, &ng_node));
          break;
        case tf::DataType::DT_UINT16:
          TF_RETURN_IF_ERROR(
              MakeConstOp<tf::uint16>(op, ng::element::u16, &ng_node));
          break;
        // For some reason the following do not work (no specialization of
        // tensorflow::checkpoint::SavedTypeTraits...)
        // case tf::DataType::DT_UINT32:
        //   TF_RETURN_IF_ERROR(MakeConstOp<tf::uint32>(op, ng::element::u32,
        //   &ng_node));
        //   break;
        // case tf::DataType::DT_UINT64:
        //   TF_RETURN_IF_ERROR(MakeConstOp<tf::uint64>(op, ng::element::u64,
        //   &ng_node));
        //   break;
        default:
          return tf::errors::Unimplemented("Unsupported TensorFlow data type: ",
                                           tf::DataType_Name(dtype));
      }

      ng_op_map[op->name()] = ng_node;
    } else if (op->type_string() == "Mul") {
      TF_RETURN_IF_ERROR(TranslateBinOp<ngraph::op::Multiply>(op, ng_op_map));
    } else if (op->type_string() == "Add") {
      TF_RETURN_IF_ERROR(TranslateBinOp<ngraph::op::Add>(op, ng_op_map));
    } else if (op->type_string() == "NoOp") {
      // Do nothing!
    } else if (op->type_string() == "MatMul") {
      // TODO(amprocte): need to handle transpose_a and transpose_b.
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 2 for elementwise op");
      }
      tf::Node* tf_lhs;
      if (op->input_node(0, &tf_lhs) != tf::Status::OK()) {
        return tf::errors::InvalidArgument("Cannot get the input node 0");
      }

      tf::Node* tf_rhs;
      if (op->input_node(1, &tf_rhs) != tf::Status::OK()) {
        return tf::errors::InvalidArgument("Cannot get the input node 1");
      }

      auto ng_lhs = ng_op_map.find(tf_lhs->name())->second;
      auto ng_rhs = ng_op_map.find(tf_rhs->name())->second;

      ng_op_map[op->name()] = make_shared<ngraph::op::Dot>(ng_lhs, ng_rhs);
    } else {
      VLOG(0) << "Unsupported Op: " << op->name() << " (" << op->type_string()
              << ")";
      return tf::errors::InvalidArgument("Unsupported Op: ", op->name(), " (",
                                         op->type_string(), ")");
    }
  }

  vector<shared_ptr<ng::Node>> ng_node_list;
  for (auto n : tf_ret_vals) {
    // Make sure that this _RetVal ONLY has one input node
    if (n->num_inputs() != 1) {
      return tf::errors::InvalidArgument(
          "ERROR: _RetVal number of inputs wrong: ", n->num_inputs());
    }

    tf::Node* tf_input_node;
    if (n->input_node(0, &tf_input_node) != tf::Status::OK()) {
      return tf::errors::InvalidArgument(
          "Error: Cannot find the source of the return node!");
    }

    // Get the corresponding nGraph node
    auto item = ng_op_map.find(tf_input_node->name());
    if (item != ng_op_map.end()) {
      ng_node_list.push_back(item->second);
    } else {
      return tf::errors::InvalidArgument("Error: Cannot find return node! ",
                                         tf_input_node->name());
    }
  }

  // Now create the nGraph function
  ng_function = make_shared<ng::Function>(ng_node_list, ng_parameter_list);
  return tf::Status::OK();
}

}  // namespace ngraph_bridge
