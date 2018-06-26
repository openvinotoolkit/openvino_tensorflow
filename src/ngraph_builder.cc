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

const static std::map<tf::DataType, ngraph::element::Type> TF_NGRAPH_TYPE_MAP = {
    { tf::DataType::DT_FLOAT, ng::element::f32 },
    { tf::DataType::DT_DOUBLE, ng::element::f64 },
    { tf::DataType::DT_INT8, ng::element::i8 },
    { tf::DataType::DT_INT16, ng::element::i16 },
    { tf::DataType::DT_INT32, ng::element::i32 },
    { tf::DataType::DT_INT64, ng::element::i64 },
    { tf::DataType::DT_UINT8, ng::element::u8 },
    { tf::DataType::DT_UINT16, ng::element::u16 },
    { tf::DataType::DT_BOOL, ng::element::boolean }
};

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

template <typename T>
static tf::Status TranslateUnaryOp(tf::Node* op, Builder::OpMap& ng_op_map) {
  if (op->num_inputs() != 1) {
    return tf::errors::InvalidArgument(
        "Number of inputs is not 1 for unary op");
  }

  tf::Node* tf_input;
  TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
  auto ng_input = ng_op_map.at(tf_input->name());
  ng_op_map[op->name()] = make_shared<T>(ng_input);

  return tf::Status::OK();
}

// Helper for Builder::TranslateGraph (elementwise binops)
template <typename T>
static tf::Status TranslateBinaryOp(tf::Node* op, Builder::OpMap& ng_op_map) {
  if (op->num_inputs() != 2) {
    return tf::errors::InvalidArgument(
        "Number of inputs is not 2 for elementwise binary op");
  }

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
      NGRAPH_VLOG(3) << op->name();
      if (op->num_inputs() != 1) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 1 for AvgPool");
      }

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
    // BiasAdd
    // -------
    else if (op->type_string() == "BiasAdd") {
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 2 for BiasAdd");
      }

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
      if (op->num_inputs() != 1) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 1 for Cast");
      }

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      try {
          auto ng_input = ng_op_map.at(tf_input->name());
          tf::DataType dtype;
          TF_RETURN_IF_ERROR(tf::GetNodeAttr(op->attrs(), "DstT", &dtype));

          try {
              ng_op_map[op->name()] = make_shared<ng::op::Convert>(
                      ng_input, TF_NGRAPH_TYPE_MAP.at(dtype));
          } catch(const std::out_of_range&) {
              return tf::errors::Unimplemented(
                      "Unsupported TensorFlow data type: ",
                      tf::DataType_Name(dtype));
          }
      } catch(const std::out_of_range&) {
          return tf::errors::NotFound("Input not found: ", tf_input->name());
      }

    }
    // --------
    // ConcatV2
    // --------
    else if (op->type_string() == "ConcatV2") {
      if (op->num_inputs() < 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not at least 2 for ConcatV2");
      }

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
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 2 for Conv2D");
      }

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
    // -----
    // DepthwiseConv2D
    // -----
    else if (op->type_string() == "DepthwiseConv2dNative") {
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 2 for DepthwiseConv2d");
      }

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
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 2 for ExpandDims");
      }

      tf::Node* tf_input;
      tf::Node* tf_dim;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_dim));

      auto ng_input = ng_op_map.find(tf_input->name());
      if (ng_input == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " + tf_input->name());
      }
      auto ng_dim = ng_op_map.find(tf_dim->name());
      if (ng_dim == ng_op_map.end()) {
        return tf::errors::InvalidArgument("Missing input: " + tf_dim->name());
      }

      auto ng_dim_const = std::dynamic_pointer_cast<ng::op::Constant>(ng_dim->second);
      if (ng_dim_const == nullptr) {
        return tf::errors::InvalidArgument("The argument dim is null for ExpandDims");
      }
      auto dim_vec = ng_dim_const->get_vector<int>();
      if (dim_vec.size() != 1) {
        return tf::errors::InvalidArgument("The size of argument dim is not 1 for ExpandDims");
      }

      auto& shape = ng_input->second->get_shape();
      auto out_shape = shape;
      out_shape.insert(out_shape.begin() + size_t(dim_vec[0]), 1);
      std::vector<size_t> shape_dimensions(shape.size());
      std::iota(shape_dimensions.begin(), shape_dimensions.end(), 0);
      std::shared_ptr<ng::Node> ng_expand_dim =
          make_shared<ng::op::Reshape>(ng_input->second, shape_dimensions, out_shape);

      ng_op_map[op->name()] = ng_expand_dim;
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
      if (op->num_inputs() != 5) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 5 for FusedBatchNorm");
      }

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
      if (op->num_inputs() != 1) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 1 for Identity");
      }

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
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 2 for MatMul");
      }

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
      NGRAPH_VLOG(3) << op->name();
      if (op->num_inputs() != 1) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 1 for MaxPool");
      }

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
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 2 for Mean");
      }

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

      if (tf_keep_dims) {
        return tf::errors::Unimplemented(
            "keep_dims is not implemented for Mean");
      }

      std::vector<tf::int64> mean_axes;
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "_ngraph_mean_static_axes", &mean_axes));

      size_t input_rank = ng_input->get_shape().size();

      ng::AxisSet ng_reduction_axes;

      for (auto i : mean_axes) {
        if (i < 0) {
          ng_reduction_axes.insert(input_rank + i);
        } else {
          ng_reduction_axes.insert(i);
        }
      }

      auto ng_mean = ng::builder::mean(ng_input, ng_reduction_axes);

      ng_op_map[op->name()] = ng_mean;
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
    // ---
    // Pad
    // ---
    else if (op->type_string() == "Pad") {
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument("Number of inputs is not 2 for Pad");
      }

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
    // ----
    // Relu
    // ----
    else if (op->type_string() == "Relu") {
      if (op->num_inputs() != 1) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 1 for Relu");
      }

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

      auto ng_input = ng_op_map.at(tf_input->name());

      ng_op_map[op->name()] = make_shared<ng::op::Relu>(ng_input);
    }
    // ----
    // Relu6
    // ----
    else if (op->type_string() == "Relu6") {
      if (op->num_inputs() != 1) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 1 for Relu6");
      }

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
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 2 for Reshape");
      }

      tf::Node* tf_input;
      tf::Node* tf_shape_node;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));
      TF_RETURN_IF_ERROR(op->input_node(1, &tf_shape_node));

      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_shape_op = ng_op_map.at(tf_shape_node->name());

      std::vector<tf::int64> shape;
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "_ngraph_reshape_static_shape", &shape));

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
    // ---------
    // Sigmoid
    // ---------
    else if (op->type_string() == "Sigmoid") {
      if (op->num_inputs() != 1) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 1 for Sigmoid");
      }

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
    // Snapshot
    // --------
    else if (op->type_string() == "Snapshot") {
      if (op->num_inputs() != 1) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 1 for Snapshot");
      }

      tf::Node* tf_arg;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_arg));
      ng_op_map[op->name()] = ng_op_map.at(tf_arg->name());
    }
    // --------- 
    // Softmax  
    // ---------
    else if (op->type_string() == "Softmax") {
      if (op->num_inputs() != 1) { 
        return tf::errors::InvalidArgument(
            "Number of inputs is not 1 for Softmax");  
      }

      tf::Node* tf_input;
      TF_RETURN_IF_ERROR(op->input_node(0, &tf_input));

      try {
        ng_op_map.at(tf_input->name()); 
      }
      catch (const std::out_of_range&) {
        return tf::errors::NotFound(tf_input->name(), " is not found in the ng_op_map");
      }
      auto ng_input = ng_op_map.at(tf_input->name());
      auto ng_input_shape = ng_input->get_shape();

      // We apply softmax on the 2nd dimension by following TF
      // And we restrict the softmax input argument to be 2D for now
      ng::AxisSet ng_axes_softmax;
      auto shape_size = ng_input_shape.size();

      if (shape_size !=2) {
        return tf::errors::InvalidArgument("TF Softmax logits must be 2-dimensional");
      }

      ng_axes_softmax.insert(1);

      ng_op_map[op->name()] = make_shared<ng::op::Softmax>(ng_input, ng_axes_softmax);
    }
    // -------
    // Squeeze
    // -------
    else if (op->type_string() == "Squeeze") {
      if (op->num_inputs() < 1) {
        return tf::errors::InvalidArgument(
            "Number of inputs should be 1 for Squeeze");
      }

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
    // ---
    // Subtract
    // ---
    else if (op->type_string() == "Sub") {
      TF_RETURN_IF_ERROR(TranslateBinaryOp<ngraph::op::Subtract>(op, ng_op_map));
    }
    // ---
    // Sum
    // ---
    else if (op->type_string() == "Sum") {
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument("Number of inputs is not 2 for Sum");
      }

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

      if (tf_keep_dims) {
        return tf::errors::Unimplemented(
            "keep_dims is not implemented for Sum");
      }

      std::vector<tf::int64> sum_axes;
      TF_RETURN_IF_ERROR(
          tf::GetNodeAttr(op->attrs(), "_ngraph_sum_static_axes", &sum_axes));

      size_t input_rank = ng_input->get_shape().size();

      ng::AxisSet ng_reduction_axes;

      for (auto i : sum_axes) {
        if (i < 0) {
          ng_reduction_axes.insert(input_rank + i);
        } else {
          ng_reduction_axes.insert(i);
        }
      }

      auto ng_sum = make_shared<ng::op::Sum>(ng_input, ng_reduction_axes);

      ng_op_map[op->name()] = ng_sum;
    }
    // ---------
    // Tanh
    // ---------
    else if (op->type_string() == "Tanh") {  
      TF_RETURN_IF_ERROR(TranslateUnaryOp<ngraph::op::Tanh>(op, ng_op_map));
    }
    // ---------
    // Transpose
    // ---------
    else if (op->type_string() == "Transpose") {
      if (op->num_inputs() != 2) {
        return tf::errors::InvalidArgument(
            "Number of inputs is not 2 for Transpose");
      }

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
