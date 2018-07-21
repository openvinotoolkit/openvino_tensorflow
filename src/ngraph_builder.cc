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
#include "ngraph_log.h"
#include "ngraph_utils.h"

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/numpy_transpose.hpp"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb_text.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"

namespace ngraph_bridge {

const static std::map<tf::DataType, ngraph::element::Type> TF_NGRAPH_TYPE_MAP =
    {{tf::DataType::DT_FLOAT, ng::element::f32},
     {tf::DataType::DT_DOUBLE, ng::element::f64},
     {tf::DataType::DT_INT8, ng::element::i8},
     {tf::DataType::DT_INT16, ng::element::i16},
     {tf::DataType::DT_INT32, ng::element::i32},
     {tf::DataType::DT_INT64, ng::element::i64},
     {tf::DataType::DT_UINT8, ng::element::u8},
     {tf::DataType::DT_UINT16, ng::element::u16},
     {tf::DataType::DT_BOOL, ng::element::boolean}};

static tf::Status ValidateInputCount(const tf::Node* op, size_t count) {
  if (op->num_inputs() != count) {
    return tf::errors::InvalidArgument(
        "\"", op->name(), "\" requires ", count, " input(s), got ",
        op->num_inputs(), " instead");
  }
  return tf::Status::OK();
}

static tf::Status ValidateInputCountMin(const tf::Node* op, size_t count) {
  if (op->num_inputs() < count) {
    return tf::errors::InvalidArgument(
        "\"", op->name(), "\" requires at least ", count, " input(s), got ",
        op->num_inputs(), " instead");
  }
  return tf::Status::OK();
}

// Helper for Builder::TranslateGraph ("Const" op)
template <typename T, typename VecT = T>
static tf::Status MakeConstOp(tf::Node* op, ng::element::Type et,
                              std::shared_ptr<ng::Node>* ng_node) {
  vector<VecT> const_values;
  tf::TensorShapeProto shape_proto;

  TF_RETURN_IF_ERROR(
      ValuesFromConstNode<T, VecT>(op->def(), &shape_proto, &const_values));

  tf::TensorShape const_shape(shape_proto);
  ng::Shape ng_shape;
  TF_RETURN_IF_ERROR(TFTensorShapeToNGraphShape(const_shape, &ng_shape));

  *ng_node = make_shared<ng::op::Constant>(et, ng_shape, const_values);
  return tf::Status::OK();
}

// Helper function to translate a unary op.
//
// Parameters:
//
//    tf::Node* op               - TF op being translated. Must have one input.
//    Builder::OpMap& ng_op_map  - The TF-to-nGraph op map.
//
//    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>>
//      build_graph              - Function to construct the graph implementing
//                                 the unary op, given the input to the unop
//                                 as an argument.
//
// Example usage:
//
//  if (n->type_string == "Square") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp(n, ng_op_map,
//                       [] (std::shared_ptr<ng::Node> n) {
//                           return (std::make_shared<ng::op::Multiply>(n,n));
//                       });
//  }
static tf::Status TranslateUnaryOp(
    tf::Node* op, Builder::OpMap& ng_op_map,
    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>)>
        build_graph) {
  TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

  tf::Node* tf_input;
  TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

  std::shared_ptr<ng::Node> ng_input;

  try {
    ng_input = ng_op_map.at(tf_input->name());
  } catch (const std::out_of_range&) {
    return tf::errors::NotFound("Input to unary op not found: %s",
                                tf_input->name());
  }

  ng_op_map[op->name()] = build_graph(ng_input);

  return tf::Status::OK();
}

// Helper function to translate a unary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Abs") {
//    TF_RETURN_IF_ERROR(TranslateUnaryOp<ng::op::Abs>(n, ng_op_map));
//  }
//
template <typename T>
static tf::Status TranslateUnaryOp(tf::Node* op, Builder::OpMap& ng_op_map) {
  return TranslateUnaryOp(op, ng_op_map, [](std::shared_ptr<ng::Node> n) { return make_shared<T>(n); });
}

// Helper for Builder::TranslateGraph (elementwise binops)
template <typename T>
static tf::Status TranslateBinaryOp(tf::Node* op, Builder::OpMap& ng_op_map) {
  TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

  tf::Node* tf_lhs;
  tf::Node* tf_rhs;
  TF_RETURN_IF_ERROR(op->input_node(0, &tf_lhs));
  TF_RETURN_IF_ERROR(op->input_node(1, &tf_rhs));

  auto ng_lhs = ng_op_map.at(tf_lhs->name());
  auto ng_rhs = ng_op_map.at(tf_rhs->name());
  std::tie(ng_lhs, ng_rhs) =
      ng::builder::numpy_broadcast(std::make_pair(ng_lhs, ng_rhs));

  ng_op_map[op->name()] = make_shared<T>(ng_lhs, ng_rhs);

  return tf::Status::OK();
}

tf::Status Builder::TranslateGraph(const std::vector<tf::TensorShape>& inputs,
                                   const tf::Graph* input_graph,
                                   shared_ptr<ng::Function>& ng_function) {
  //
  // We will visit ops in topological order.
  //
  vector<tf::Node*> ordered;
  tf::GetReversePostOrder(*input_graph, &ordered);

  //
  // Split ops into params, retvals, and all others.
  //
  vector<tf::Node*> tf_params;
  vector<tf::Node*> tf_ret_vals;
  vector<tf::Node*> tf_ops;

  for (auto n : ordered) {
    if (n->IsSink() || n->IsSource()) {
      continue;
    }

    if (n->IsControlFlow()) {
      return tf::errors::Unimplemented(
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
  // generated nGraph nodes.
  //
  // TODO(amprocte): at some point it may become necessary to extend this
  // to vectors of nGraph nodes, for cases where a TF node has multiple
  // outputs that will have to be served by different nGraph nodes.
  //
  Builder::OpMap ng_op_map;

  //
  // Populate the parameter list, and also put parameters into the op map.
  //
  vector<shared_ptr<ng::op::Parameter>> ng_parameter_list(tf_params.size());

  for (auto parm : tf_params) {
    tf::DataType dtype;
    if (tf::GetNodeAttr(parm->attrs(), "T", &dtype) != tf::Status::OK()) {
      return tf::errors::InvalidArgument("No data type defined for _Arg");
    }
    int index;
    if (tf::GetNodeAttr(parm->attrs(), "index", &index) != tf::Status::OK()) {
      return tf::errors::InvalidArgument("No index defined for _Arg");
    }

    ng::element::Type ng_et;
    TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

    ng::Shape ng_shape;
    TF_RETURN_IF_ERROR(TFTensorShapeToNGraphShape(inputs[index], &ng_shape));

    auto ng_param = make_shared<ng::op::Parameter>(ng_et, ng_shape);
    ng_op_map[parm->name()] = ng_param;
    ng_parameter_list[index] = ng_param;
  }

  //
  // Now create the nGraph ops from TensorFlow ops.
  //
  for (auto op : tf_ops) {
    NGRAPH_VLOG(2) << "Constructing op " << op->name() << " which is " << op->type_string();

    // NOTE: The following cases should be kept in alphabetical order.

    // ---
    // Abs
    // ---
    if (op->type_string() == "Abs") {
      TF_RETURN_IF_ERROR(TranslateUnaryOp<ngraph::op::Abs>(op, ng_op_map));
    }
    // ---
    // Add
    // ---
    else if (op->type_string() == "Add") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::Add>(op, ng_op_map));
    }
    // -------
    // AvgPool
    // -------
    else if (op->type_string() == "AvgPool") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

      auto ng_input = ng_op_map.at(tf_input->name());

      std::vector<tf::int32> tf_strides;
      std::vector<tf::int32> tf_ksize;
      std::string tf_padding_type;
      std::string tf_data_format;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "strides", &tf_strides));
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

      if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        return tf::errors::InvalidArgument(
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

      if (is_nhwc) {
        auto& s = ng_input->get_shape();
        ng::Shape reshaped_shape{s[0], s[3], s[1], s[2]};

        NGRAPH_VLOG(3) << "reshaped_shape: " << ng::join(reshaped_shape);

        ng_input = make_shared<ng::op::Reshape>(
            ng_input, ng::AxisVector{0, 3, 1, 2}, reshaped_shape);

        ng_strides[0] = tf_strides[1];
        ng_strides[1] = tf_strides[2];

        ng_image_shape[0] = s[1];
        ng_image_shape[1] = s[2];

        ng_kernel_shape[0] = tf_ksize[1];
        ng_kernel_shape[1] = tf_ksize[2];
      } else {
        auto& s = ng_input->get_shape();

        ng_strides[0] = tf_strides[2];
        ng_strides[1] = tf_strides[3];

        ng_image_shape[0] = s[2];
        ng_image_shape[1] = s[3];

        ng_kernel_shape[0] = tf_ksize[2];
        ng_kernel_shape[1] = tf_ksize[3];
      }

      NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
      NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
      NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

      // TODO: change this once nGraph supports negative padding
      // (CoordinateDiff) for AvgPool
      // ng::CoordinateDiff ng_padding_below{0,0};
      // ng::CoordinateDiff ng_padding_above{0,0};
      ng::Shape ng_padding_below{0, 0};
      ng::Shape ng_padding_above{0, 0};

      if (tf_padding_type == "SAME") {
        for (size_t i = 0; i < 2; i++) {
          size_t image_size = ng_image_shape[i];
          size_t filter_shape = ng_kernel_shape[i];
          size_t filter_stride = ng_strides[i];

          tf::int64 padding_needed;
          if (image_size % filter_stride == 0) {
            padding_needed = filter_shape - filter_stride;
          } else {
            padding_needed = filter_shape - (image_size % filter_stride);
          }
          if (padding_needed < 0) {
            padding_needed = 0;
          }

          size_t padding_lhs = padding_needed / 2;
          size_t padding_rhs = padding_needed - padding_lhs;
          ng_padding_below[i] = padding_lhs;
          ng_padding_above[i] = padding_rhs;
        }
      }

      NGRAPH_VLOG(3) << "ng_padding_below: " << ng::join(ng_padding_below);
      NGRAPH_VLOG(3) << "ng_padding_above: " << ng::join(ng_padding_above);

      std::shared_ptr<ng::Node> ng_avgpool = make_shared<ng::op::AvgPool>(
          ng_input, ng_kernel_shape, ng_strides, ng_padding_below,
          ng_padding_above, false);

      if (is_nhwc) {
        auto& s = ng_avgpool->get_shape();
        ng::Shape reshaped_shape{s[0], s[2], s[3], s[1]};

        ng_avgpool = make_shared<ng::op::Reshape>(
            ng_avgpool, ng::AxisVector{0, 2, 3, 1}, reshaped_shape);
      }

      NGRAPH_VLOG(3) << "avgpool outshape: {"
                     << ng::join(ng_avgpool->get_shape()) << "}";

      ng_op_map[op->name()] = ng_avgpool;
    }
    // -------
    // BatchMatMul
    // -------
    else if (op->type_string() == "BatchMatMul") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));
 
      tf::Node* tf_lhs;
      tf::Node* tf_rhs;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_lhs));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_rhs));
  
      try {
        ng_op_map.at(tf_lhs->name());
      } catch(const std::out_of_range&) {
          return tf::errors::NotFound(tf_lhs->name(),"is not found in ng_op_map");
      }

      try {
        ng_op_map.at(tf_rhs->name());
      } catch(const std::out_of_range&) {
          return tf::errors::NotFound(tf_rhs->name(),"is not found in ng_op_map");
      }

      auto ng_lhs = ng_op_map.at(tf_lhs->name()); 
      auto ng_rhs = ng_op_map.at(tf_rhs->name()); 
      auto ng_lhs_shape = ng_lhs->get_shape(); 
      auto ng_rhs_shape = ng_rhs->get_shape();

      if(ng_lhs_shape.size() != ng_rhs_shape.size()) {
        return tf::errors::InvalidArgument(
            "Dimensions of two input args are not the same for BatchMatMul");
      }
      size_t n_dims = ng_lhs_shape.size();
      if(n_dims < 2) {
        return tf::errors::InvalidArgument(
            "Dimensions of input args for BatchMatMul must be >=2", n_dims);
      }

      ng::AxisVector out_axes;
      for (size_t i = 0; i < n_dims - 2; ++i) {
        if(ng_lhs_shape[i] != ng_rhs_shape[i]){
          return tf::errors::InvalidArgument(
              "ng_lhs_shape and ng_rhs_shape must be the same for BatchMatMul for each dimension",
               i);
        }
        out_axes.push_back(i);
      }

      bool tf_adj_x = false;
      bool tf_adj_y = false;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "adj_x", &tf_adj_x));
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "adj_y", &tf_adj_y));
      
      auto ng_lhs_axes = out_axes;
      auto ng_rhs_axes = out_axes;
      if (tf_adj_x) {
        ng_lhs_axes.push_back(n_dims-1);
        ng_lhs_axes.push_back(n_dims-2);
        ng_lhs = ng::builder::numpy_transpose(ng_lhs, ng_lhs_axes);
      }
      if (tf_adj_y) {
        ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims-2); 
        ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims-1);
        ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng_rhs_axes);
      } else {
        ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims-1);      
        ng_rhs_axes.insert(ng_rhs_axes.begin(), n_dims-2);
        ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng_rhs_axes);
      }

      ng_lhs_shape = ng_lhs->get_shape();
      ng_rhs_shape = ng_rhs->get_shape();
 
      if(ng_lhs_shape[n_dims-1] != ng_rhs_shape[0]) {
        return tf::errors::InvalidArgument(
            "The last dimension of ng_lhs and the first dimension of ng_rhs should have the same size"
            );
      }
      if (n_dims == 2) {
        ng_op_map[op->name()] = make_shared<ngraph::op::Dot>(ng_lhs, ng_rhs);
      } else {
        auto output_shape = ng_lhs_shape;
        output_shape[n_dims-1] = ng_rhs_shape[1];
        auto dot_output = make_shared<ngraph::op::Dot>(ng_lhs, ng_rhs);
        size_t compound_size = 1;
        for (int i=0; i<out_axes.size(); i++) {
          compound_size *= output_shape[i];
        }
        auto dot_axes = out_axes;
        dot_axes.push_back(n_dims-2);
        dot_axes.push_back(n_dims-1);
        for (int i=0; i<out_axes.size(); i++) {
          dot_axes.push_back(n_dims+i);
        }
        ng::Shape dot_shape = {compound_size, ng_lhs_shape[n_dims-2], ng_rhs_shape[1], compound_size};
        std::shared_ptr<ng::Node> dot_reshape;
        if (n_dims == 3) {
          dot_reshape = dot_output;
        } else {
          dot_reshape = make_shared<ngraph::op::Reshape>(dot_output, dot_axes, dot_shape); 
        }
        ng::Shape tmp_shape = {1, ng_lhs_shape[n_dims-2], ng_rhs_shape[1]};
        vector<shared_ptr<ngraph::Node>> tmp_tensors;
        for (size_t i = 0; i < dot_shape[0]; i++) { 
          const std::vector<size_t> lower_bound{i, 0, 0, i};
          const std::vector<size_t> upper_bound{i+1, dot_shape[1], dot_shape[2], i+1};
          auto slice_out = make_shared<ngraph::op::Slice>(dot_reshape, lower_bound, upper_bound);
          auto reshape_out = make_shared<ngraph::op::Reshape>(slice_out, ng::AxisVector{0, 1, 2, 3}, tmp_shape);
          tmp_tensors.push_back(reshape_out);
        }
        auto concat_op = make_shared<ngraph::op::Concat>(tmp_tensors, 0);
        if (n_dims == 3) {
          ng_op_map[op->name()] = concat_op;
        } else {
          ng_op_map[op->name()] = make_shared<ngraph::op::Reshape>(concat_op, ng::AxisVector{0, 1, 2 }, output_shape);
        }
      }
    }
    // -------
    // BiasAdd
    // -------
    else if (op->type_string() == "BiasAdd") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_input;
      tf::Node* tf_bias;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_bias));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_bias = ng_op_map.at(tf_bias->name());

      std::string tf_data_format;
      if (tf::GetNodeAttr(op->attrs(), "data_format", &tf_data_format) !=
          tf::Status::OK()) {
        tf_data_format = "NHWC";
      }

      if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        return tf::errors::InvalidArgument(
            "BiasAdd data format is neither NHWC nor NCHW");
      }

      auto ng_input_shape = ng_input->get_shape();
      auto ng_bias_shape = ng_bias->get_shape();
      if (ng_bias_shape.size() != 1) {
        return tf::errors::InvalidArgument(
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

      ng_op_map[op->name()] = ng_add;
    }
    // --------
    // Cast
    // --------
    else if (op->type_string() == "Cast") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      try {
        auto ng_input = ng_op_map.at(tf_input->name());
        tf::DataType dtype;
        TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "DstT", &dtype));

        try {
          ng_op_map[op->name()] = make_shared<ng::op::Convert>(
              ng_input, TF_NGRAPH_TYPE_MAP.at(dtype));
        } catch (const std::out_of_range&) {
          return tf::errors::Unimplemented("Unsupported TensorFlow data type: ",
                                           tf::DataType_Name(dtype));
        }
      } catch (const std::out_of_range&) {
        return tf::errors::NotFound("Input not found: ", tf_input->name());
      }

    }
    // --------
    // ConcatV2
    // --------
    else if (op->type_string() == "ConcatV2") {
      TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 2));

      tf::Node* tf_axis_node;
      TF_RETURN_IF_ERROR(op->input_node(op->num_inputs() - 1, &tf_axis_node));

      auto ng_axis_op = ng_op_map.at(tf_axis_node->name());

      tf::int64 concat_axis;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(
          op->attrs(), "_ngraph_concat_static_axis", &concat_axis));

      tf::Node* tf_first_arg;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_first_arg));

      if (concat_axis < 0) {
        concat_axis +=
            tf::int64(ng_op_map[tf_first_arg->name()]->get_shape().size());
      }

      ng::NodeVector ng_args;

      for (int i = 0; i < op->num_inputs() - 1; i++) {
        tf::Node* tf_arg;
        TF_RETURN_IF_ERROR(op->input_node(i, &tf_arg));
        ng_args.push_back(ng_op_map[tf_arg->name()]);
      }

      ng_op_map[op->name()] =
          make_shared<ng::op::Concat>(ng_args, size_t(concat_axis));
    }
    // -----
    // Const
    // -----
    else if (op->type_string() == "Const") {
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
        case tf::DataType::DT_BOOL:
          TF_RETURN_IF_ERROR(
              MakeConstOp<bool, char>(op, ng::element::boolean, &ng_node));
          break;
        default:
          return tf::errors::Unimplemented("Unsupported TensorFlow data type: ",
                                           tf::DataType_Name(dtype));
      }

      ng_op_map[op->name()] = ng_node;
    }
    // ------
    // Conv2D
    // ------
    else if (op->type_string() == "Conv2D") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_input;
      tf::Node* tf_filter;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_filter));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_filter = ng_op_map.at(tf_filter->name());

      std::vector<tf::int32> tf_strides;
      std::vector<tf::int32> tf_dilations;
      std::string tf_padding_type;
      std::string tf_data_format;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "strides", &tf_strides));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

      if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        return tf::errors::InvalidArgument(
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

      if (is_nhwc) {
        auto& s = ng_input->get_shape();
        ng::Shape reshaped_shape{s[0], s[3], s[1], s[2]};

        NGRAPH_VLOG(3) << "reshaped_shape: " << ng::join(reshaped_shape);

        ng_input = make_shared<ng::op::Reshape>(
            ng_input, ng::AxisVector{0, 3, 1, 2}, reshaped_shape);

        ng_strides[0] = tf_strides[1];
        ng_strides[1] = tf_strides[2];

        ng_dilations[0] = tf_dilations[1];
        ng_dilations[1] = tf_dilations[2];

        ng_image_shape[0] = s[1];
        ng_image_shape[1] = s[2];
      } else {
        auto& s = ng_input->get_shape();

        ng_strides[0] = tf_strides[2];
        ng_strides[1] = tf_strides[3];

        ng_dilations[0] = tf_dilations[2];
        ng_dilations[1] = tf_dilations[3];

        ng_image_shape[0] = s[2];
        ng_image_shape[1] = s[3];
      }

      NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
      NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
      NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

      {
        auto& s = ng_filter->get_shape();
        ng::Shape reshaped_shape{s[3], s[2], s[0], s[1]};
        ng_filter = make_shared<ng::op::Reshape>(
            ng_filter, ng::AxisVector{3, 2, 0, 1}, reshaped_shape);

        ng_kernel_shape[0] = s[0];
        ng_kernel_shape[1] = s[1];
      }

      NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

      ng::CoordinateDiff ng_padding_below{0, 0};
      ng::CoordinateDiff ng_padding_above{0, 0};

      if (tf_padding_type == "SAME") {
        for (size_t i = 0; i < 2; i++) {
          size_t image_size = ng_image_shape[i];
          size_t filter_shape = (ng_kernel_shape[i] - 1) * ng_dilations[i] + 1;
          size_t filter_stride = ng_strides[i];

          tf::int64 padding_needed;
          if (image_size % filter_stride == 0) {
            padding_needed = filter_shape - filter_stride;
          } else {
            padding_needed = filter_shape - (image_size % filter_stride);
          }
          if (padding_needed < 0) {
            padding_needed = 0;
          }

          size_t padding_lhs = padding_needed / 2;
          size_t padding_rhs = padding_needed - padding_lhs;
          ng_padding_below[i] = padding_lhs;
          ng_padding_above[i] = padding_rhs;
        }
      }

      NGRAPH_VLOG(3) << "ng_padding_below: " << ng::join(ng_padding_below);
      NGRAPH_VLOG(3) << "ng_padding_above: " << ng::join(ng_padding_above);

      std::shared_ptr<ng::Node> ng_conv = make_shared<ng::op::Convolution>(
          ng_input, ng_filter, ng_strides, ng_dilations, ng_padding_below,
          ng_padding_above);

      if (is_nhwc) {
        auto& s = ng_conv->get_shape();
        ng::Shape reshaped_shape{s[0], s[2], s[3], s[1]};

        ng_conv = make_shared<ng::op::Reshape>(
            ng_conv, ng::AxisVector{0, 2, 3, 1}, reshaped_shape);
      }

      ng_op_map[op->name()] = ng_conv;
    }
    // ------
    // Conv2DBackpropInput
    // ------
    else if (op->type_string() == "Conv2DBackpropInput") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 3));

      tf::Node *tf_filter, *tf_out_backprop;
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_filter));
      TF_RETURN_IF_ERROR(op->input_node(2, &tf_out_backprop));
      shared_ptr<ng::Node> ng_filter, ng_out_backprop;
      try {
        ng_filter = ng_op_map.at(tf_filter->name());
      } catch (const std::out_of_range&) {
        return tf::errors::NotFound("Filter not found: %s", tf_filter->name());
      }
      try {
        ng_out_backprop = ng_op_map.at(tf_out_backprop->name());
      } catch (const std::out_of_range&) {
        return tf::errors::NotFound("Out Backprop not found: %s",
                                    tf_out_backprop->name());
      }

      // TODO: refactor me to be less redundant with other convolution ops
      std::vector<tf::int32> tf_strides;
      std::vector<tf::int32> tf_dilations;
      std::string tf_padding_type;
      std::string tf_data_format;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "strides", &tf_strides));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

      if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        return tf::errors::InvalidArgument(
            "Conv2DBackpropInput data format is neither NHWC nor NCHW: %s",
            tf_data_format);
      }

      std::vector<tf::int64> tf_input_sizes;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(),
            "_ngraph_static_input_sizes", &tf_input_sizes));
      if (std::any_of(tf_input_sizes.begin(), tf_input_sizes.end(),
            [](tf::int32 size) { return size <= 0; })) {
        return tf::errors::InvalidArgument(
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

      if (is_nhwc) {
        ng_strides[0] = tf_strides[1];
        ng_strides[1] = tf_strides[2];
        ng_dilations[0] = tf_dilations[1];
        ng_dilations[1] = tf_dilations[2];
        ng_image_shape[0] = tf_input_sizes[1];
        ng_image_shape[1] = tf_input_sizes[2];
        ng_batch_shape = { static_cast<unsigned long>(tf_input_sizes[0]),
                           static_cast<unsigned long>(tf_input_sizes[3]),
                           static_cast<unsigned long>(tf_input_sizes[1]),
                           static_cast<unsigned long>(tf_input_sizes[2]) };
        auto& s = ng_out_backprop->get_shape();
        ng::Shape reshaped{s[0], s[3], s[1], s[2]};
        ng_out_backprop = make_shared<ng::op::Reshape>(
            ng_out_backprop, ng::AxisVector{0, 3, 1, 2}, reshaped);
      } else {
        ng_strides[0] = tf_strides[2];
        ng_strides[1] = tf_strides[3];
        ng_dilations[0] = tf_dilations[2];
        ng_dilations[1] = tf_dilations[3];
        ng_image_shape[0] = tf_input_sizes[2];
        ng_image_shape[1] = tf_input_sizes[3];
        ng_batch_shape = { static_cast<unsigned long>(tf_input_sizes[0]),
                           static_cast<unsigned long>(tf_input_sizes[1]),
                           static_cast<unsigned long>(tf_input_sizes[2]),
                           static_cast<unsigned long>(tf_input_sizes[3]) };
      }

      NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
      NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
      NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

      {
        auto& s = ng_filter->get_shape();
        ng::Shape reshaped_shape{s[3], s[2], s[0], s[1]};
        ng_filter = make_shared<ng::op::Reshape>(
            ng_filter, ng::AxisVector{3, 2, 0, 1}, reshaped_shape);

        ng_kernel_shape[0] = s[0];
        ng_kernel_shape[1] = s[1];
      }

      NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

      ng::CoordinateDiff ng_padding_below{0, 0};
      ng::CoordinateDiff ng_padding_above{0, 0};

      if (tf_padding_type == "SAME") {
        for (size_t i = 0; i < 2; i++) {
          size_t image_size = ng_image_shape[i];
          size_t filter_shape = ng_kernel_shape[i];
          size_t filter_stride = ng_strides[i];

          tf::int64 padding_needed;
          if (image_size % filter_stride == 0) {
            padding_needed = filter_shape - filter_stride;
          } else {
            padding_needed = filter_shape - (image_size % filter_stride);
          }
          if (padding_needed < 0) {
            padding_needed = 0;
          }

          size_t padding_lhs = padding_needed / 2;
          size_t padding_rhs = padding_needed - padding_lhs;
          ng_padding_below[i] = padding_lhs;
          ng_padding_above[i] = padding_rhs;
        }
      }

      NGRAPH_VLOG(3) << "ng_padding_below: " << ng::join(ng_padding_below);
      NGRAPH_VLOG(3) << "ng_padding_above: " << ng::join(ng_padding_above);

      std::shared_ptr<ng::Node> ng_data =
        make_shared<ng::op::ConvolutionBackpropData>(
            ng_batch_shape, ng_filter, ng_out_backprop, ng_strides,
            ng_dilations, ng_padding_below, ng_padding_above,
            ng::Strides(ng_batch_shape.size() - 2, 1));

      if (is_nhwc) {
        auto& s = ng_data->get_shape();
        ng::Shape reshaped{s[0], s[2], s[3], s[1]};
        ng_data = make_shared<ng::op::Reshape>(
            ng_data, ng::AxisVector{0, 2, 3, 1}, reshaped);
      }

      ng_op_map[op->name()] = ng_data;
    }

    // -----
    // DepthwiseConv2dNative
    // -----
    else if (op->type_string() == "DepthwiseConv2dNative") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_input;
      tf::Node* tf_filter;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_filter));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_filter = ng_op_map.at(tf_filter->name());

      std::vector<tf::int32> tf_strides;
      std::vector<tf::int32> tf_dilations;
      std::string tf_padding_type;
      std::string tf_data_format;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "strides", &tf_strides));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "dilations", &tf_dilations));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

      if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        return tf::errors::InvalidArgument(
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

      if (is_nhwc) {
        auto& s = ng_input->get_shape();
        ng::Shape reshaped_shape{s[0], s[3], s[1], s[2]};

        NGRAPH_VLOG(3) << "reshaped_shape: " << ng::join(reshaped_shape);

        ng_input = make_shared<ng::op::Reshape>(
            ng_input, ng::AxisVector{0, 3, 1, 2}, reshaped_shape);

        ng_strides[0] = tf_strides[1];
        ng_strides[1] = tf_strides[2];

        ng_dilations[0] = tf_dilations[0];
        ng_dilations[1] = tf_dilations[1];

        ng_image_shape[0] = s[1];
        ng_image_shape[1] = s[2];
      } else {
        auto& s = ng_input->get_shape();

        ng_strides[0] = tf_strides[1];
        ng_strides[1] = tf_strides[2];

        ng_dilations[0] = tf_dilations[0];
        ng_dilations[1] = tf_dilations[1];

        ng_image_shape[0] = s[2];
        ng_image_shape[1] = s[3];
      }

      NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
      NGRAPH_VLOG(3) << "ng_dilations: " << ng::join(ng_dilations);
      NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);

      {
        auto& s = ng_filter->get_shape();
        ng::Shape reshaped_shape{s[3], s[2], s[0], s[1]};
        ng_filter = make_shared<ng::op::Reshape>(
            ng_filter, ng::AxisVector{3, 2, 0, 1}, reshaped_shape);

        ng_kernel_shape[0] = s[0];
        ng_kernel_shape[1] = s[1];
      }

      NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

      ng::CoordinateDiff ng_padding_below{0, 0};
      ng::CoordinateDiff ng_padding_above{0, 0};

      if (tf_padding_type == "SAME") {
        for (size_t i = 0; i < 2; i++) {
          size_t image_size = ng_image_shape[i];
          size_t filter_shape = (ng_kernel_shape[i] - 1) * ng_dilations[i] + 1;
          size_t filter_stride = ng_strides[i];

          tf::int64 padding_needed;
          if (image_size % filter_stride == 0) {
            padding_needed = filter_shape - filter_stride;
          } else {
            padding_needed = filter_shape - (image_size % filter_stride);
          }
          if (padding_needed < 0) {
            padding_needed = 0;
          }

          size_t padding_lhs = padding_needed / 2;
          size_t padding_rhs = padding_needed - padding_lhs;
          ng_padding_below[i] = padding_lhs;
          ng_padding_above[i] = padding_rhs;
        }
      }

      NGRAPH_VLOG(3) << "ng_padding_below: " << ng::join(ng_padding_below);
      NGRAPH_VLOG(3) << "ng_padding_above: " << ng::join(ng_padding_above);

      // ng input shape is NCHW
      auto& input_shape = ng_input->get_shape();
      // ng filter shape is OIHW
      auto& filter_shape = ng_filter->get_shape();
      ng::NodeVector ng_args;

      for (size_t i = 0; i < input_shape[1]; i++) {
        const std::vector<size_t> lower_bound{0, i, 0, 0};
        const std::vector<size_t> upper_bound{input_shape[0], i + 1,
                                              input_shape[2], input_shape[3]};
        auto ng_sliced_input =
            make_shared<ng::op::Slice>(ng_input, lower_bound, upper_bound);

        const std::vector<size_t> f_lower_bound{0, i, 0, 0};
        const std::vector<size_t> f_upper_bound{
            filter_shape[0], i + 1, filter_shape[2], filter_shape[3]};
        auto ng_sliced_filter =
            make_shared<ng::op::Slice>(ng_filter, f_lower_bound, f_upper_bound);

        NGRAPH_VLOG(3) << "depthwise conv 2d.";
        NGRAPH_VLOG(3) << "sliced shape "
                       << ng::join(ng_sliced_input->get_shape());
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

      if (is_nhwc) {
        auto& s = ng_concat->get_shape();
        ng::Shape reshaped_shape{s[0], s[2], s[3], s[1]};
        ng_concat = make_shared<ng::op::Reshape>(
            ng_concat, ng::AxisVector{0, 2, 3, 1}, reshaped_shape);
      }

      ng_op_map[op->name()] = ng_concat;
    }
    // -----
    // Equal
    // -----
    else if (op->type_string() == "Equal") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::Equal>(op, ng_op_map));
    }
    // -----
    // Exp
    // -----
    else if (op->type_string() == "Exp") {
      TF_RETURN_IF_ERROR(TranslateUnaryOp<ngraph::op::Exp>(op, ng_op_map));
    }
    // --------
    // ExpandDims
    // --------
    else if (op->type_string() == "ExpandDims") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_input;
      tf::Node* tf_dim;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_dim));

      auto ng_input = ng_op_map.find(tf_input->name());
      if (ng_input == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " +
                                           tf_input->name());
      }
      auto ng_dim = ng_op_map.find(tf_dim->name());
      if (ng_dim == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " + tf_dim->name());
      }

      auto ng_dim_const =
          std::dynamic_pointer_cast<ng::op::Constant>(ng_dim->second);
      if (ng_dim_const == nullptr) {
        return tf::errors::InvalidArgument(
            "The argument dim is null for ExpandDims");
      }
      auto dim_vec = ng_dim_const->get_vector<int>();
      if (dim_vec.size() != 1) {
        return tf::errors::InvalidArgument(
            "The size of argument dim is not 1 for ExpandDims");
      }

      auto& shape = ng_input->second->get_shape();
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
      std::shared_ptr<ng::Node> ng_expand_dim = make_shared<ng::op::Reshape>(
          ng_input->second, shape_dimensions, out_shape);

      ng_op_map[op->name()] = ng_expand_dim;
    }
    // --------
    // Fill
    // --------
    else if (op->type_string() == "Fill") {

      tf::Node* tf_value;
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_value));

      shared_ptr<ng::Node> ng_value;
      try {
        ng_value = ng_op_map.at(tf_value->name()); 
      } catch (const std::out_of_range&) {
        return tf::errors::InvalidArgument("Missing input: " + tf_value->name());
      }
      ng_value = ng_op_map.at(tf_value->name());
     
      std::vector<tf::int64> dims_vec;
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "_ngraph_fill_static_dims", &dims_vec));

      ng::Shape ng_output_shape(dims_vec.size());
      ng::AxisSet ng_axis_set;
      for (size_t i = 0; i < dims_vec.size(); ++i) {
        ng_output_shape[i] = dims_vec[i];
        ng_axis_set.insert(i);
      }
      ng_op_map[op->name()] = make_shared<ng::op::Broadcast>(ng_value,
        ng_output_shape, ng_axis_set);
    }
    // --------
    // Floor
    // --------
    else if (op->type_string() == "Floor") {
      TF_RETURN_IF_ERROR(TranslateUnaryOp<ngraph::op::Floor>(op, ng_op_map));
    }
    // --------------
    // FusedBatchNorm
    // --------------
    else if (op->type_string() == "FusedBatchNorm") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 5));

      bool tf_is_training;
      if (tf::GetNodeAttr(op->attrs(), "is_training", &tf_is_training) !=
          tf::Status::OK()) {
        NGRAPH_VLOG(3) << "is_training attribute not present, setting to true";
        tf_is_training = true;
      }

      NGRAPH_VLOG(3) << "is_training: " << tf_is_training;

      tf::Node* tf_input;
      tf::Node* tf_scale;
      tf::Node* tf_offset;
      tf::Node* tf_mean;
      tf::Node* tf_variance;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_scale));
      TF_RETURN_IF_ERROR(op->input_node(2, &tf_offset));
      TF_RETURN_IF_ERROR(op->input_node(3, &tf_mean));
      TF_RETURN_IF_ERROR(op->input_node(4, &tf_variance));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_scale = ng_op_map.at(tf_scale->name());
      auto ng_offset = ng_op_map.at(tf_offset->name());
      auto ng_mean = ng_op_map.at(tf_mean->name());
      auto ng_variance = ng_op_map.at(tf_variance->name());

      std::string tf_data_format;
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

      if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        return tf::errors::InvalidArgument(
            "Conv2D data format is neither NHWC nor NCHW");
      }

      bool is_nhwc = (tf_data_format == "NHWC");

      NGRAPH_VLOG(3) << "data_format: " << tf_data_format;

      float tf_epsilon;
      if (tf::GetNodeAttr(op->attrs(), "epsilon", &tf_epsilon) !=
          tf::Status::OK()) {
        NGRAPH_VLOG(3) << "epsilon attribute not present, setting to zero";
        tf_epsilon = 0;  // FIXME(amprocte): is this the right default?
      }

      NGRAPH_VLOG(3) << "epsilon: " << tf_epsilon;

      if (is_nhwc) {
        auto& s = ng_input->get_shape();
        ng::Shape reshaped_shape{s[0], s[3], s[1], s[2]};

        NGRAPH_VLOG(3) << "reshaped_shape: " << ng::join(reshaped_shape);

        ng_input = make_shared<ng::op::Reshape>(
            ng_input, ng::AxisVector{0, 3, 1, 2}, reshaped_shape);
      }

      std::shared_ptr<ng::Node> ng_batch_norm;

      ng_batch_norm = make_shared<ng::op::BatchNorm>(
          tf_epsilon, ng_scale, ng_offset, ng_input, ng_mean, ng_variance,
          tf_is_training);

      if (is_nhwc) {
        auto& s = ng_batch_norm->get_shape();
        ng::Shape reshaped_shape{s[0], s[2], s[3], s[1]};

        ng_batch_norm = make_shared<ng::op::Reshape>(
            ng_batch_norm, ng::AxisVector{0, 2, 3, 1}, reshaped_shape);
      }

      ng_op_map[op->name()] = ng_batch_norm;
    }
    // -----
    // Greater
    // -----
    else if (op->type_string() == "Greater") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::Greater>(op, ng_op_map));
    }
    // -----
    // GreaterEqual
    // -----
    else if (op->type_string() == "GreaterEqual") {
      TF_RETURN_IF_ERROR(
          TranslateBinaryOp<ngraph::op::GreaterEq>(op, ng_op_map));
    }
    // --------
    // Identity
    // --------
    else if (op->type_string() == "Identity") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_arg;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_arg));
      ng_op_map[op->name()] = ng_op_map.at(tf_arg->name());
    }
    // -----
    // Less
    // -----
    else if (op->type_string() == "Less") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::Less>(op, ng_op_map));
    }
    // -----
    // LessEqual
    // -----
    else if (op->type_string() == "LessEqual") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::LessEq>(op, ng_op_map));
    }
    // ---
    // Log
    // ---
    else if (op->type_string() == "Log") {
      TF_RETURN_IF_ERROR(TranslateUnaryOp<ngraph::op::Log>(op, ng_op_map));
    }
    // -----
    // LogicalAnd
    // -----
    else if (op->type_string() == "LogicalAnd") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::And>(op, ng_op_map));
    }
    // ------
    // MatMul
    // ------
    else if (op->type_string() == "MatMul") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_lhs;
      tf::Node* tf_rhs;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_lhs));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_rhs));

      auto ng_lhs = ng_op_map.at(tf_lhs->name());
      auto ng_rhs = ng_op_map.at(tf_rhs->name());

      // Transpose arguments if requested.
      bool transpose_a = false;
      bool transpose_b = false;

      if (tf::GetNodeAttr(op->attrs(), "transpose_a", &transpose_a) ==
              tf::Status::OK() &&
          transpose_a) {
        ng_lhs = ng::builder::numpy_transpose(ng_lhs, ng::AxisVector{1, 0});
      }
      if (tf::GetNodeAttr(op->attrs(), "transpose_b", &transpose_b) ==
              tf::Status::OK() &&
          transpose_b) {
        ng_rhs = ng::builder::numpy_transpose(ng_rhs, ng::AxisVector{1, 0});
      }

      // The default axis count for nGraph's Dot op is 1, which is just what
      // we need here.
      ng_op_map[op->name()] = make_shared<ngraph::op::Dot>(ng_lhs, ng_rhs);
    }
    // -----
    // Maximum
    // -----
    else if (op->type_string() == "Maximum") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::Maximum>(op, ng_op_map));
    }
    // -------
    // MaxPool
    // -------
    else if (op->type_string() == "MaxPool") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

      auto ng_input = ng_op_map.at(tf_input->name());

      std::vector<tf::int32> tf_strides;
      std::vector<tf::int32> tf_ksize;
      std::string tf_padding_type;
      std::string tf_data_format;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "strides", &tf_strides));
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

      if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
        return tf::errors::InvalidArgument(
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

      if (is_nhwc) {
        auto& s = ng_input->get_shape();
        ng::Shape reshaped_shape{s[0], s[3], s[1], s[2]};

        NGRAPH_VLOG(3) << "reshaped_shape: " << ng::join(reshaped_shape);

        ng_input = make_shared<ng::op::Reshape>(
            ng_input, ng::AxisVector{0, 3, 1, 2}, reshaped_shape);

        ng_strides[0] = tf_strides[1];
        ng_strides[1] = tf_strides[2];

        ng_image_shape[0] = s[1];
        ng_image_shape[1] = s[2];

        ng_kernel_shape[0] = tf_ksize[1];
        ng_kernel_shape[1] = tf_ksize[2];
      } else {
        auto& s = ng_input->get_shape();

        ng_strides[0] = tf_strides[2];
        ng_strides[1] = tf_strides[3];

        ng_image_shape[0] = s[2];
        ng_image_shape[1] = s[3];

        ng_kernel_shape[0] = tf_ksize[2];
        ng_kernel_shape[1] = tf_ksize[3];
      }

      NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
      NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
      NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

      // TODO: change this once nGraph supports negative padding
      // (CoordinateDiff) for MaxPool
      // ng::CoordinateDiff ng_padding_below{0,0};
      // ng::CoordinateDiff ng_padding_above{0,0};
      ng::Shape ng_padding_below{0, 0};
      ng::Shape ng_padding_above{0, 0};

      if (tf_padding_type == "SAME") {
        for (size_t i = 0; i < 2; i++) {
          size_t image_size = ng_image_shape[i];
          size_t filter_shape = ng_kernel_shape[i];
          size_t filter_stride = ng_strides[i];

          tf::int64 padding_needed;
          if (image_size % filter_stride == 0) {
            padding_needed = filter_shape - filter_stride;
          } else {
            padding_needed = filter_shape - (image_size % filter_stride);
          }
          if (padding_needed < 0) {
            padding_needed = 0;
          }

          size_t padding_lhs = padding_needed / 2;
          size_t padding_rhs = padding_needed - padding_lhs;
          ng_padding_below[i] = padding_lhs;
          ng_padding_above[i] = padding_rhs;
        }
      }

      NGRAPH_VLOG(3) << "ng_padding_below: " << ng::join(ng_padding_below);
      NGRAPH_VLOG(3) << "ng_padding_above: " << ng::join(ng_padding_above);

      std::shared_ptr<ng::Node> ng_maxpool =
          make_shared<ng::op::MaxPool>(ng_input, ng_kernel_shape, ng_strides,
                                       ng_padding_below, ng_padding_above);

      if (is_nhwc) {
        auto& s = ng_maxpool->get_shape();
        ng::Shape reshaped_shape{s[0], s[2], s[3], s[1]};

        ng_maxpool = make_shared<ng::op::Reshape>(
            ng_maxpool, ng::AxisVector{0, 2, 3, 1}, reshaped_shape);
      }

      NGRAPH_VLOG(3) << "maxpool outshape: {"
                     << ng::join(ng_maxpool->get_shape()) << "}";

      ng_op_map[op->name()] = ng_maxpool;
    }
    // ----
    // Mean
    // ----
    else if (op->type_string() == "Mean") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_input;
      tf::Node* tf_axes_node;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_axes_node));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_axes_op = ng_op_map.at(tf_axes_node->name());

      bool tf_keep_dims;
      if (tf::GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) !=
          tf::Status::OK()) {
        tf_keep_dims = false;
      }

      std::vector<tf::int64> mean_axes;
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "_ngraph_mean_static_axes", &mean_axes));

      ng::Shape input_shape = ng_input->get_shape();
      size_t input_rank = ng_input->get_shape().size();

      ng::AxisSet ng_reduction_axes;

      for (auto i : mean_axes) {
        if (i < 0) {
          ng_reduction_axes.insert(input_rank + i);
        } else {
          ng_reduction_axes.insert(i);
        }
      }

      std::shared_ptr<ng::Node> ng_mean = ng::builder::mean(ng_input, ng_reduction_axes);

      // If keep_dims is specified we need to reshape to put back the reduced
      // axes, with length 1.
      if (tf_keep_dims) {
        ng::Shape ng_result_shape_with_keep(input_rank);

        for (size_t i = 0; i < input_rank; i++) {
          if (ng_reduction_axes.count(i) == 0) {
            ng_result_shape_with_keep[i] = input_shape[i];
          }
          else {
            ng_result_shape_with_keep[i] = 1;
          }
        }

        ng::AxisVector ng_axis_order(ng_mean->get_shape().size());

        for (size_t i = 0; i < ng_mean->get_shape().size(); i++) {
          ng_axis_order[i] = i;
        }

        ng_mean = make_shared<ng::op::Reshape>(ng_mean, ng_axis_order, ng_result_shape_with_keep);
      }

      ng_op_map[op->name()] = ng_mean;
    }
    // -----
    // Minimum
    // -----
    else if (op->type_string() == "Minimum") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::Minimum>(op, ng_op_map));
    }
    // ---
    // Mul
    // ---
    else if (op->type_string() == "Mul") {
      TF_RETURN_IF_ERROR(
          TranslateBinaryOp<ngraph::op::Multiply>(op, ng_op_map));
    }
    // ----
    // NoOp
    // ----
    else if (op->type_string() == "NoOp") {
      // Do nothing! NoOps sometimes get placed on nGraph for bureaucratic
      // reasons, but they have no data flow inputs or outputs.
    }
    // -------
    // Pack
    // -------
    else if (op->type_string() == "Pack") {
      TF_RETURN_IF_ERROR(ValidateInputCountMin(op, 1));

      ng::NodeVector ng_concat_inputs;

      for (size_t i = 0;i < op->num_inputs();++i) {  
        tf::Node* tf_input;
        TF_RETURN_IF_ERROR(op->input_node(i, &tf_input));
        shared_ptr<ng::Node> ng_input;
        try {
          ng_input = ng_op_map.at(tf_input->name());
        } catch (const std::out_of_range&) {
          return tf::errors::InvalidArgument("Missing input: " + tf_input->name());
        }
        ng_concat_inputs.push_back(ng_input);
      }

      tf::int32 tf_axis;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "axis", &tf_axis));
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
      for (size_t i = 0; i < input_rank; i++) {
        ng_axis_order[i] = i;
      }

      if (concat_axis == input_rank) {
        // need to add extra dimension before we concatenate
        // along it
        ng::Shape extended_shape = input_shape;
        extended_shape.push_back( 1 );
        for (size_t i = 0;i < ng_concat_inputs.size(); ++i) {
          ng_concat_inputs[i] = make_shared<ng::op::Reshape>(
            ng_concat_inputs[i], ng_axis_order, extended_shape); 
        }
        ng_axis_order.push_back(input_rank);
      }

      auto concat = make_shared<ng::op::Concat>(ng_concat_inputs, concat_axis);
      ng_op_map[op->name()] =
          make_shared<ng::op::Reshape>(concat, ng_axis_order, output_shape);
    }

    // ---
    // Pad
    // ---
    else if (op->type_string() == "Pad") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_input;
      tf::Node* tf_paddings_node;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_paddings_node));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_paddings_op = ng_op_map.at(tf_paddings_node->name());

      std::vector<tf::int64> paddings;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(
          op->attrs(), "_ngraph_pad_static_paddings", &paddings));

      NGRAPH_VLOG(3) << "{" << ng::join(paddings) << "}";

      if (paddings.size() % 2 != 0) {
        return tf::errors::InvalidArgument(
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
          ng_input->get_element_type(), ng::Shape{},
          std::vector<std::string>{"0"});
      auto pad_op = make_shared<ng::op::Pad>(
          ng_input, pad_val_op, padding_below, padding_above, padding_interior);

      ng_op_map[op->name()] = pad_op;
    }
    // ---
    // Pow
    // ---
    else if (op->type_string() == "Pow") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ng::op::Power>(op, ng_op_map));
    }
    // ---
    // Prod
    // ---
    else if (op->type_string() == "Prod") {
      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

      shared_ptr<ng::Node> ng_input;
      try {
        ng_input = ng_op_map.at(tf_input->name()); 
      } catch (const std::out_of_range&) {
        return tf::errors::InvalidArgument("Missing input: " + tf_input->name());
      }

      ng_input = ng_op_map.at(tf_input->name());

      ng::AxisSet ng_axis_set;
      if (op->num_inputs() == 2) {
        tf::Node* tf_axis;
        TF_RETURN_IF_ERROR(op->input_node(1, &tf_axis));
        auto ng_axis = ng_op_map.find(tf_axis->name());

        if (ng_axis == ng_op_map.end()) {
          return tf::errors::InvalidArgument("Missing input: " + tf_axis->name());
        }
        
        auto ng_axis_const = std::dynamic_pointer_cast<ng::op::Constant>(ng_axis->second);
        if (ng_axis_const == nullptr) {
          for (size_t i = 0; i < ng_input->get_shape().size(); i++) {
            ng_axis_set.insert(i);
          }
        } 
        else {
          auto axis_vec = ng_axis_const->get_vector<int>();
          for (size_t i = 0; i < axis_vec.size(); ++i) {
            if (axis_vec[i] >= 0) {
              ng_axis_set.insert(axis_vec[i]);       
            }
            else {
              // ng_axis_set has unsigned type, converting negative axis 
              ng_axis_set.insert(ng_input->get_shape().size() + axis_vec[i]);
            }
	  }
	}
      }
      else {
        return tf::errors::InvalidArgument("Prod operation requires 2 inputs");
      }

      bool tf_keep_dims;
      if (tf::GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) !=
          tf::Status::OK()) {
        tf_keep_dims = false;
      }

      if (tf_keep_dims) {
        return tf::errors::Unimplemented(
            "keep_dims is not implemented for Prod");
      }

      ng_op_map[op->name()] = make_shared<ng::op::Product>(ng_input, ng_axis_set);
    }
    // ----
    // RealDiv
    // ----
    else if (op->type_string() == "RealDiv") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::Divide>(op, ng_op_map));
    }
    // ----
    // Relu
    // ----
    else if (op->type_string() == "Relu") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

      auto ng_input = ng_op_map.at(tf_input->name());

      ng_op_map[op->name()] = make_shared<ng::op::Relu>(ng_input);
    }
    // ----
    // Relu6
    // ----
    else if (op->type_string() == "Relu6") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto constant_6 = make_shared<ng::op::Constant>(
          ng_input->get_element_type(), ng_input->get_shape(),
          std::vector<std::string>(ng::shape_size(ng_input->get_shape()), "6"));
      auto relu6_op = make_shared<ng::op::Minimum>(
          make_shared<ng::op::Relu>(ng_input), constant_6);

      ng_op_map[op->name()] = relu6_op;
    }
    // -------
    // Reshape
    // -------
    else if (op->type_string() == "Reshape") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_input;
      tf::Node* tf_shape_node;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_shape_node));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_shape_op = ng_op_map.at(tf_shape_node->name());

      NGRAPH_VLOG(3) << "Input shape: " << ng::join(ng_input->get_shape());

      std::vector<tf::int64> shape;
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "_ngraph_reshape_static_shape", &shape));

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
            return tf::errors::InvalidArgument(
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
          NGRAPH_VLOG(3) << tf_input->name();
          NGRAPH_VLOG(3) << "{" << ng::join(ng_input->get_shape()) << "}";
          NGRAPH_VLOG(3) << "{" << ng::join(shape) << "}";
          return tf::errors::InvalidArgument(
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
      for (size_t i = 0; i < ng_input->get_shape().size(); i++) {
        ng_axis_order[i] = i;
      }

      ng_op_map[op->name()] =
          make_shared<ng::op::Reshape>(ng_input, ng_axis_order, ng_shape);
    }
    // -----
    // Rsqrt
    // -----
    else if (op->type_string() == "Rsqrt") {
      TF_RETURN_IF_ERROR(TranslateUnaryOp(op, ng_op_map,
         [](std::shared_ptr<ng::Node> n) {
           // Create a constant tensor populated with the value -1/2. (1/sqrt(x) = x^(-1/2))
           auto et = n->get_element_type();
           auto shape = n->get_shape();
           std::vector<std::string> constant_values(ng::shape_size(shape),"-0.5");
           auto ng_exponent = std::make_shared<ng::op::Constant>(et,shape,constant_values);

           // Raise each element of the input to the power -0.5.
           return std::make_shared<ng::op::Power>(n,ng_exponent);
         }));
    }
    // ---------
    // Sigmoid
    // ---------
    else if (op->type_string() == "Sigmoid") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto exp_op =
          make_shared<ng::op::Exp>(make_shared<ng::op::Negative>(ng_input));
      auto constant_1 = make_shared<ng::op::Constant>(
          ng_input->get_element_type(), ng_input->get_shape(),
          std::vector<std::string>(ng::shape_size(ng_input->get_shape()), "1"));

      auto denominator_op = make_shared<ng::op::Add>(constant_1, exp_op);

      ng_op_map[op->name()] =
          make_shared<ng::op::Divide>(constant_1, denominator_op);
    }
    // ---
    // Sign
    // ---
    else if (op->type_string() == "Sign") {
      TF_RETURN_IF_ERROR(TranslateUnaryOp<ngraph::op::Sign>(op, ng_op_map));
    }
    // --------
    // Slice
    // --------
    else if (op->type_string() == "Slice") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 3));

      tf::Node* tf_input;
      tf::Node* tf_begin;
      tf::Node* tf_size;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_begin));
      TF_RETURN_IF_ERROR(op->input_node(2, &tf_size));

      auto ng_input = ng_op_map.find(tf_input->name());
      if (ng_input == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " +
                                           tf_input->name());
      }
      auto ng_begin = ng_op_map.find(tf_begin->name());
      if (ng_begin == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " +
                                           tf_begin->name());
      }
      auto ng_size = ng_op_map.find(tf_size->name());
      if (ng_size == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " + tf_size->name());
      }

      auto ng_begin_const =
          std::dynamic_pointer_cast<ng::op::Constant>(ng_begin->second);
      if (ng_begin_const == nullptr) {
        return tf::errors::InvalidArgument(
            "The argument begin is null for Slice");
      }
      auto lower_vec = ng_begin_const->get_vector<int>();

      auto ng_size_const =
          std::dynamic_pointer_cast<ng::op::Constant>(ng_size->second);
      if (ng_size_const == nullptr) {
        return tf::errors::InvalidArgument(
            "The argument size is null for Slice");
      }
      auto size_vec = ng_size_const->get_vector<int>();

      auto& input_shape = ng_input->second->get_shape();
      NGRAPH_VLOG(3) << "Begin input for Slice: " << ng::join(lower_vec);
      NGRAPH_VLOG(3) << "Size input for Slice: " << ng::join(size_vec);
      if (std::any_of(size_vec.begin(), size_vec.end(), [](int i){ return i <= 0; })) {
        std::transform(size_vec.begin(), size_vec.end(), input_shape.begin(),
                       size_vec.begin(), [](int first, int second) {
                         if (first < 0) {
                           return second + first + 1;
                         } else if (first == 0) {
                           return second;
                         } else {
                           return first;
                         }
                       });
        NGRAPH_VLOG(3) << "Size input for Slice (if less than 0): " << ng::join(size_vec);
      }

      std::vector<int> upper_vec(lower_vec.size());
      std::transform(lower_vec.begin(), lower_vec.end(), size_vec.begin(),
                     upper_vec.begin(), std::plus<int>());

      std::vector<size_t> l(lower_vec.begin(), lower_vec.end());
      std::vector<size_t> u(upper_vec.begin(), upper_vec.end());
      auto ng_slice = make_shared<ng::op::Slice>(ng_input->second, l, u);
      ng_op_map[op->name()] = ng_slice;

    }
    // --------
    // Snapshot
    // --------
    else if (op->type_string() == "Snapshot") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_arg;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_arg));
      ng_op_map[op->name()] = ng_op_map.at(tf_arg->name());
    }
    // ---------
    // Softmax
    // ---------
    else if (op->type_string() == "Softmax") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

      shared_ptr<ng::Node> ng_input;
      try {
        ng_input = ng_op_map.at(tf_input->name());
      } catch (const std::out_of_range&) {
        return tf::errors::NotFound(tf_input->name(),
                                    " is not found in the ng_op_map");
      }
      auto ng_input_shape = ng_input->get_shape();

      // We apply softmax on the 2nd dimension by following TF
      // And we restrict the softmax input argument to be 2D for now
      ng::AxisSet ng_axes_softmax;
      auto shape_size = ng_input_shape.size();

      if (shape_size != 2) {
        return tf::errors::InvalidArgument(
            "TF Softmax logits must be 2-dimensional");
      }

      ng_axes_softmax.insert(1);

      ng_op_map[op->name()] =
          make_shared<ng::op::Softmax>(ng_input, ng_axes_softmax);
    }
    // ------
    // Square
    // ------
    else if (op->type_string() == "Square") {
      TF_RETURN_IF_ERROR(TranslateUnaryOp(op, ng_op_map,
        [](std::shared_ptr<ng::Node> n)
        {
          return std::make_shared<ng::op::Multiply>(n,n);
        }));
    }
    // -------
    // Squeeze
    // -------
    else if (op->type_string() == "Squeeze") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      auto ng_input = ng_op_map.at(tf_input->name());

      std::vector<tf::int32> tf_axis;
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "squeeze_dims", &tf_axis));
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
              throw tensorflow::errors::InvalidArgument(
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
      for (size_t i = 0; i < ng_input->get_shape().size(); i++) {
        ng_axis_order[i] = i;
      }

      ng_op_map[op->name()] =
          make_shared<ng::op::Reshape>(ng_input, ng_axis_order, output_shape);
    }
    // --------
    // StridedSlice
    // --------
    else if (op->type_string() == "StridedSlice") {
      // TODO refactor StrideSlice with Slice op
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 4));

      tf::Node* tf_input;
      tf::Node* tf_begin;
      tf::Node* tf_size;
      tf::Node* tf_stride;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_begin));
      TF_RETURN_IF_ERROR(op->input_node(2, &tf_size));
      TF_RETURN_IF_ERROR(op->input_node(3, &tf_stride));

      auto ng_input = ng_op_map.find(tf_input->name());
      if (ng_input == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " +
                                           tf_input->name());
      }
      auto ng_begin = ng_op_map.find(tf_begin->name());
      if (ng_begin == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " +
                                           tf_begin->name());
      }
      auto ng_size = ng_op_map.find(tf_size->name());
      if (ng_size == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " + tf_size->name());
      }
      auto ng_stride = ng_op_map.find(tf_stride->name());
      if (ng_stride == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " +
                                           tf_stride->name());
      }

      auto ng_begin_const =
          std::dynamic_pointer_cast<ng::op::Constant>(ng_begin->second);
      if (ng_begin_const == nullptr) {
        return tf::errors::InvalidArgument(
            "The argument begin is null for StridedSlice");
      }
      auto lower_vec = ng_begin_const->get_vector<int>();

      auto ng_size_const =
          std::dynamic_pointer_cast<ng::op::Constant>(ng_size->second);
      if (ng_size_const == nullptr) {
        return tf::errors::InvalidArgument(
            "The argument size is null for StridedSlice");
      }
      auto size_vec = ng_size_const->get_vector<int>();

      auto& input_shape = ng_input->second->get_shape();
      NGRAPH_VLOG(3) << "Begin input for StridedSlice: " << ng::join(lower_vec);
      NGRAPH_VLOG(3) << "Size input for StridedSlice: " << ng::join(size_vec);
      if (std::any_of(size_vec.begin(), size_vec.end(), [](int i){ return i <= 0; })) {
        std::transform(size_vec.begin(), size_vec.end(), input_shape.begin(),
                       size_vec.begin(), [](int first, int second) {
                         if (first < 0) {
                           return second + first + 1;
                         } else if (first == 0) {
                           return second;
                         } else {
                           return first;
                         }
                       });
        NGRAPH_VLOG(3) << "Transform size input for StridedSlice: " << ng::join(size_vec);
      }

      std::vector<int> upper_vec(lower_vec.size());
      std::transform(lower_vec.begin(), lower_vec.end(), size_vec.begin(),
                     upper_vec.begin(), std::plus<int>());

      auto ng_stride_const =
          std::dynamic_pointer_cast<ng::op::Constant>(ng_stride->second);
      if (ng_stride_const == nullptr) {
        return tf::errors::InvalidArgument(
            "The argument stride is null for StridedSlice");
      }
      auto stride_vec = ng_stride_const->get_vector<int>();

      std::vector<size_t> l(lower_vec.begin(), lower_vec.end());
      std::vector<size_t> u(upper_vec.begin(), upper_vec.end());
      std::vector<size_t> s(stride_vec.begin(), stride_vec.end());
      auto ng_strided_slice =
          make_shared<ng::op::Slice>(ng_input->second, l, u, s);
      ng_op_map[op->name()] = ng_strided_slice;

    }
    // ---
    // Subtract
    // ---
    else if (op->type_string() == "Sub") {
      TF_RETURN_IF_ERROR(
          TranslateBinaryOp<ngraph::op::Subtract>(op, ng_op_map));
    }
    // ---
    // Sum
    // ---
    else if (op->type_string() == "Sum") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_input;
      tf::Node* tf_axes_node;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_axes_node));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_axes_op = ng_op_map.at(tf_axes_node->name());

      bool tf_keep_dims;
      if (tf::GetNodeAttr(op->attrs(), "keep_dims", &tf_keep_dims) !=
          tf::Status::OK()) {
        tf_keep_dims = false;
      }

      std::vector<tf::int64> sum_axes;
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "_ngraph_sum_static_axes", &sum_axes));

      ng::Shape input_shape = ng_input->get_shape();
      size_t input_rank = input_shape.size();

      ng::AxisSet ng_reduction_axes;

      for (auto i : sum_axes) {
        if (i < 0) {
          ng_reduction_axes.insert(input_rank + i);
        } else {
          ng_reduction_axes.insert(i);
        }
      }

      std::shared_ptr<ng::Node> ng_sum = make_shared<ng::op::Sum>(ng_input, ng_reduction_axes);

      // If keep_dims is specified we need to reshape to put back the reduced
      // axes, with length 1.
      if (tf_keep_dims) {
        ng::Shape ng_result_shape_with_keep(input_rank);

        for (size_t i = 0; i < input_rank; i++) {
          if (ng_reduction_axes.count(i) == 0) {
            ng_result_shape_with_keep[i] = input_shape[i];
          }
          else {
            ng_result_shape_with_keep[i] = 1;
          }
        }

        ng::AxisVector ng_axis_order(ng_sum->get_shape().size());

        for (size_t i = 0; i < ng_sum->get_shape().size(); i++) {
          ng_axis_order[i] = i;
        }

        ng_sum = make_shared<ng::op::Reshape>(ng_sum, ng_axis_order, ng_result_shape_with_keep);
      }

      ng_op_map[op->name()] = ng_sum;
    }
    // ---------
    // Tanh
    // ---------
    else if (op->type_string() == "Tanh") {
      TF_RETURN_IF_ERROR(TranslateUnaryOp<ngraph::op::Tanh>(op, ng_op_map));
    }
    // ---------
    // Tile
    // ---------
    else if (op->type_string() == "Tile") {
      if (op->num_inputs() != 2) { 
        return tf::errors::InvalidArgument( 
            "Number of inputs is not 2 for Tile");
      }

      tf::Node* tf_input;
      tf::Node* tf_multiples;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_multiples));

      shared_ptr<ng::Node> ng_input;
      shared_ptr<ng::Node> ng_multiples;
      try {
        ng_input = ng_op_map.at(tf_input->name()); 
      } catch (const std::out_of_range&) {
        return tf::errors::NotFound("Input to tile op not found: %s",
                                tf_input->name());
      }
      try {
        ng_multiples = ng_op_map.at(tf_multiples->name()); 
      } catch (const std::out_of_range&) {
        return tf::errors::NotFound("Input to tile op not found: %s",
                                tf_multiples->name());
      }
      std::vector<tf::int64> multiples;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(
          op->attrs(), "_ngraph_tile_static_multiples", &multiples));
      auto ng_input_shape = ng_input->get_shape();
      if (ng_input_shape.size() != multiples.size()) {
        return tf::errors::InvalidArgument(
            "dimension of input does not match length of multiples");
      }
      std::shared_ptr<ng::Node> ng_output = ng_input;
      ng::Shape output_shape = ng_input_shape;
      bool is_empty = false;
      for (int i=0; i<ng_input_shape.size(); i++) {
        if (multiples[i] == 0) {
          is_empty = true;
        }
        output_shape[i] = ng_input_shape[i] * multiples[i];
      }
      if (is_empty) {
        ng_op_map[op->name()] = make_shared<ngraph::op::Constant>( 
                       ng_input->get_element_type(),
                       output_shape,
                       std::vector<std::string>(ng::shape_size(output_shape), "0"));
      } else {
        for (int i=0; i<ng_input_shape.size(); i++) {
          if (multiples[i] < 0) {
            return tf::errors::InvalidArgument("Expected multiples[", i, "] >= 0, but got ",
                                  multiples[i]); 
          }
          vector<shared_ptr<ng::Node>> tmp_tensors;
          for (int k=0; k<multiples[i]; k++) {
            tmp_tensors.push_back(ng_output);
          }
          auto ng_concat = make_shared<ngraph::op::Concat>(tmp_tensors, i);
          ng_output = ng_concat;
        }
        ng_op_map[op->name()] = ng_output;
      }
    }
    // ---------
    // Transpose
    // ---------
    else if (op->type_string() == "Transpose") {
      TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));

      tf::Node* tf_input;
      tf::Node* tf_permutation_node;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_permutation_node));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_permutation_op = ng_op_map.at(tf_permutation_node->name());

      std::vector<tf::int64> permutation;
      TF_RETURN_IF_ERROR(tf::GetNodeAttr(
          op->attrs(), "_ngraph_transpose_static_permutation", &permutation));

      ng::AxisVector ng_axis_order;
      ng_axis_order.reserve(permutation.size());

      NGRAPH_VLOG(3) << ng::join(permutation);

      for (auto i : permutation) {
        ng_axis_order.push_back(i);
      }

      NGRAPH_VLOG(3) << ng::join(ng_axis_order);

      ng_op_map[op->name()] =
          ng::builder::numpy_transpose(ng_input, ng_axis_order);
    }

    // -----------------------------
    // Catch-all for unsupported ops
    // -----------------------------
    else {
      NGRAPH_VLOG(3) << "Unsupported Op: " << op->name() << " ("
                     << op->type_string() << ")";
      NGRAPH_VLOG(3) << op->def().DebugString();
      return tf::errors::InvalidArgument("Unsupported Op: ", op->name(), " (",
                                         op->type_string(), ")");
    }
  }

  //
  // Populate the result list.
  //
  vector<shared_ptr<ng::Node>> ng_result_list(tf_ret_vals.size());

  for (auto n : tf_ret_vals) {
    // Make sure that this _Retval only has one input node.
    if (n->num_inputs() != 1) {
      return tf::errors::InvalidArgument("_Retval has ", n->num_inputs(),
                                         " inputs, should have 1");
    }

    tf::Node* tf_input_node;
    if (n->input_node(0, &tf_input_node) != tf::Status::OK()) {
      return tf::errors::InvalidArgument(
          "Cannot find the source of the return node");
    }

    int index;
    if (tf::GetNodeAttr(n->attrs(), "index", &index) != tf::Status::OK()) {
      return tf::errors::InvalidArgument("No index defined for _Retval");
    }

    auto item = ng_op_map.find(tf_input_node->name());
    if (item != ng_op_map.end()) {
      ng_result_list[index] = item->second;
    } else {
      return tf::errors::InvalidArgument("Cannot find return node: ",
                                         tf_input_node->name());
    }
  }

  //
  // Create the nGraph function.
  //
  ng_function = make_shared<ng::Function>(ng_result_list, ng_parameter_list);
  return tf::Status::OK();
}

}  // namespace ngraph_bridge
