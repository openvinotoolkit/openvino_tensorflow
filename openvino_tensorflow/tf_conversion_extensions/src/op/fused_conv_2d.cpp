// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extensions.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

ov::op::PadType convert_tf_padding(const ov::frontend::NodeContext& node,
                                   const std::string& tf_padding) {
  auto op_type = node.get_op_type();

  if (tf_padding == "VALID") {
    return ov::op::PadType::VALID;
  }
  if (tf_padding == "SAME") {
    // According to the formulas for calculating auto_pad values of the
    // Conv layer in the Operation specification,
    // the SAME_UPPER value matches to the SAME value in TensorFlow
    return ov::op::PadType::SAME_UPPER;
  }

  return ov::op::PadType::EXPLICIT;
}

void fill_explicit_pads_vectors(
    const ov::frontend::NodeContext& node, bool is_nhwc,
    size_t spatial_dims_num, const std::vector<int64_t>& tf_explicit_paddings,
    ov::CoordinateDiff& pads_begin, ov::CoordinateDiff& pads_end) {
  auto fullfill_pads = [&](ov::CoordinateDiff& pads,
                           const std::vector<int64_t>& indexes) {
    pads.resize(indexes.size());
    for (int i = 0; i < indexes.size(); ++i) {
      pads[i] = tf_explicit_paddings[indexes[i]];
    }
  };

  if (spatial_dims_num == 2) {
    // TENSORFLOW_OP_VALIDATION(node,
    //                          tf_explicit_paddings.size() == 8,
    //                          "Conv2D expects 8 padding values for EXPLICIT
    //                          padding mode.");
    // prepare pads_begin and pads_end attributes for EXPLICIT padding mode
    if (is_nhwc) {
      // For NHWC layout, explicit paddings has the following form:
      // [0, 0, pad_h1, pad_h2, pad_w1, pad_w2, 0, 0]
      fullfill_pads(pads_begin, {2, 4});
      fullfill_pads(pads_end, {3, 5});
    } else {
      // For NCHW layout, explicit paddings has the following form:
      // [0, 0, 0, 0, pad_h1, pad_h2, pad_w1, pad_w2]
      fullfill_pads(pads_begin, {4, 6});
      fullfill_pads(pads_end, {5, 7});
    }
  }
}

std::shared_ptr<ov::opset8::Transpose> make_transpose(
    const ov::Output<ov::Node>& arg, const ov::AxisVector& input_order) {
  auto order = std::make_shared<ov::opset8::Constant>(
      element::i64, Shape{input_order.size()}, input_order);
  auto transpose = std::make_shared<ov::opset8::Transpose>(arg, order);
  return transpose;
}

OutputVector translate_fused_conv_2d_op(const ov::frontend::NodeContext& node) {
  auto num_args = node.get_attribute<int64_t>("num_args");
  auto fused_ops = node.get_attribute<std::vector<string>>("fused_ops");

  auto tf_data_format = node.get_attribute<std::string>("data_format");
  bool is_nhwc = (tf_data_format == "NHWC");

  auto ng_input = node.get_input(0), ng_filter = node.get_input(1);

  int spatial_dims_num = 2;  // for conv2d

  auto CreateNgConv = [&](Output<Node>& ng_input, Output<Node>& ng_filter) {
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_dilations =
        node.get_attribute<std::vector<int64_t>>("dilations", {1, 1, 1, 1});
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, tf_padding_type);

    auto tf_explicit_paddings = std::vector<int64_t>{};
    if (auto_pad == ov::op::PadType::EXPLICIT) {
      tf_explicit_paddings =
          node.get_attribute<std::vector<int64_t>>("explicit_paddings", {});
    }

    if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
      FRONT_END_GENERAL_CHECK(false,
                              "Conv2D data format is neither NHWC nor NCHW");
    }

    // TF Kernel Test Checks
    // Strides in the batch and depth dimension is not supported
    if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
      FRONT_END_GENERAL_CHECK(
          false, "Strides in batch and depth dimensions is not supported: " +
                     node.get_op_type());
    }

    Strides ng_strides(2);
    Strides ng_dilations(2);
    Shape ng_image_shape(2);
    Shape ng_kernel_shape(2);

    convert_nhwc_to_hw(is_nhwc, tf_strides, ng_strides);
    convert_nhwc_to_hw(is_nhwc, tf_dilations, ng_dilations);

    ov::frontend::tensorflow::convert_nhwc_to_nchw(
        is_nhwc, ng_input, ov::Rank(spatial_dims_num + 2));
    ov::AxisVector permutation_2d = {3, 2, 0, 1};
    auto filter = make_transpose(ng_filter, permutation_2d);

    CoordinateDiff pads_begin;
    CoordinateDiff pads_end;
    if (auto_pad == ov::op::PadType::EXPLICIT) {
      fill_explicit_pads_vectors(node, is_nhwc, spatial_dims_num,
                                 tf_explicit_paddings, pads_begin, pads_end);
    }

    ov::Output<ov::Node> conv =
        make_shared<Convolution>(ng_input, filter, ng_strides, pads_begin,
                                 pads_end, ng_dilations, auto_pad);

    return conv;
  };

  if (vec_str_cmp(fused_ops, {"BiasAdd"}) ||
      vec_str_cmp(fused_ops, {"BiasAdd", "Relu"}) ||
      vec_str_cmp(fused_ops, {"BiasAdd", "Relu6"}) ||
      vec_str_cmp(fused_ops, {"BiasAdd", "LeakyRelu"}) ||
      vec_str_cmp(fused_ops, {"BiasAdd", "Elu"}) ||
      vec_str_cmp(fused_ops, {"BiasAdd", "Add"}) ||
      vec_str_cmp(fused_ops, {"BiasAdd", "Add", "Relu"}) ||
      vec_str_cmp(fused_ops, {"BiasAdd", "Add", "LeakyRelu"})) {
    if (vec_str_cmp(fused_ops, {"BiasAdd", "Add"}) ||
        vec_str_cmp(fused_ops, {"BiasAdd", "Add", "Relu"}) ||
        vec_str_cmp(fused_ops, {"BiasAdd", "Add", "LeakyRelu"})) {
      if (num_args != 2) {
        FRONT_END_GENERAL_CHECK(
            false, "FusedConv2DBiasAdd Add has incompatible num_args");
      }
    } else {
      if (num_args != 1) {
        FRONT_END_GENERAL_CHECK(false,
                                "FusedConv2DBiasAdd has incompatible num_args");
      }
    }

    auto ng_conv = CreateNgConv(ng_input, ng_filter);
    auto ng_bias = node.get_input(2);

    auto ng_conv_rank = ng_conv.get_partial_shape().rank();
    if (ng_conv_rank == ov::Rank::dynamic()) {
      FRONT_END_GENERAL_CHECK(false, "Convolution shape has dynamic rank");
    }
    auto ng_bias_rank = ng_bias.get_partial_shape().rank();
    if (ng_bias_rank.get_length() != 1) {
      FRONT_END_GENERAL_CHECK(
          false, "Bias argument to BiasAdd does not have one dimension");
    }

    std::vector<size_t> reshape_pattern_values(ng_conv_rank.get_length(), 1U);
    reshape_pattern_values[1] = ng_bias.get_shape().front();
    auto reshape_pattern = make_shared<Constant>(
        element::u64, Shape{reshape_pattern_values.size()},
        reshape_pattern_values);
    auto ng_bias_reshaped =
        make_shared<Reshape>(ng_bias, reshape_pattern, false);

    auto ng_add = make_shared<Add>(ng_conv, ng_bias_reshaped)->output(0);

    if (vec_str_cmp(fused_ops, {"BiasAdd", "Relu"})) {
      auto ng_relu = make_shared<Relu>(ng_add)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_relu, ov::Rank(4));
      return {ng_relu};
    } else if (vec_str_cmp(fused_ops, {"BiasAdd", "Relu6"})) {
      auto ng_relu6 = make_shared<Clamp>(ng_add, 0, 6)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_relu6, ov::Rank(4));
      return {ng_relu6};
    } else if (vec_str_cmp(fused_ops, {"BiasAdd", "LeakyRelu"})) {
      auto tf_leakyrelu_alpha = node.get_attribute<float>("leakyrelu_alpha");
      auto ng_leakyrelu_alpha =
          make_shared<Constant>(element::f32, Shape{}, tf_leakyrelu_alpha);
      auto ng_alphax = make_shared<Multiply>(ng_leakyrelu_alpha, ng_add);
      auto ng_lrelu = make_shared<Maximum>(ng_alphax, ng_add)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_lrelu, ov::Rank(4));
      return {ng_lrelu};
    } else if (vec_str_cmp(fused_ops, {"BiasAdd", "Elu"})) {
      float tf_elu_alpha = 1.0;
      tf_elu_alpha = node.get_attribute<float>("leakyrelu_alpha");
      auto ng_elu = make_shared<Elu>(ng_add, tf_elu_alpha)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_elu, ov::Rank(4));
      return {ng_elu};
    } else if (vec_str_cmp(fused_ops, {"BiasAdd", "Add"})) {
      auto ng_input2 = node.get_input(3);
      convert_nhwc_to_nchw(is_nhwc, ng_input2, ov::Rank(4));
      auto ng_out = make_shared<Add>(ng_add, ng_input2)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_out, ov::Rank(4));
      return {ng_out};
    } else if (vec_str_cmp(fused_ops, {"BiasAdd", "Add", "Relu"})) {
      auto ng_input2 = node.get_input(3);
      convert_nhwc_to_nchw(is_nhwc, ng_input2, ov::Rank(4));
      auto ng_add2 = make_shared<Add>(ng_add, ng_input2)->output(0);
      auto ng_relu = make_shared<Relu>(ng_add2)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_relu, ov::Rank(4));
      return {ng_relu};
    } else if (vec_str_cmp(fused_ops, {"BiasAdd", "Add", "LeakyRelu"})) {
      auto ng_input2 = node.get_input(3);
      convert_nhwc_to_nchw(is_nhwc, ng_input2, ov::Rank(4));
      auto ng_add2 = make_shared<Add>(ng_add, ng_input2)->output(0);
      auto tf_leakyrelu_alpha = node.get_attribute<float>("leakyrelu_alpha");
      auto ng_leakyrelu_alpha =
          make_shared<Constant>(element::f32, Shape{}, tf_leakyrelu_alpha)
              ->output(0);
      auto ng_alphax =
          make_shared<Multiply>(ng_leakyrelu_alpha, ng_add2)->output(0);
      auto ng_lrelu = make_shared<Maximum>(ng_alphax, ng_add2)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_lrelu, ov::Rank(4));
      return {ng_lrelu};
    } else {
      convert_nchw_to_nhwc(is_nhwc, ng_add, ov::Rank(4));
      return {ng_add};
    }
  } else if (vec_str_cmp(fused_ops, {"FusedBatchNorm"}) ||
             vec_str_cmp(fused_ops, {"FusedBatchNorm", "Relu"}) ||
             vec_str_cmp(fused_ops, {"FusedBatchNorm", "Relu6"}) ||
             vec_str_cmp(fused_ops, {"FusedBatchNorm", "LeakyRelu"})) {
    if (num_args != 4) {
      FRONT_END_GENERAL_CHECK(
          false, "FusedConv2D with FusedBatchNorm has incompatible num_args");
    }

    auto ng_input = node.get_input(0), ng_filter = node.get_input(1),
         ng_scale = node.get_input(2), ng_offset = node.get_input(3),
         ng_mean = node.get_input(4), ng_variance = node.get_input(5),
         ng_conv = CreateNgConv(ng_input, ng_filter);

    auto tf_epsilon = node.get_attribute<float>("epsilon");

    auto ng_batch_norm =
        make_shared<BatchNormInference>(ng_conv, ng_scale, ng_offset, ng_mean,
                                        ng_variance, tf_epsilon)
            ->output(0);

    if (vec_str_cmp(fused_ops, {"FusedBatchNorm", "Relu"})) {
      auto ng_relu = make_shared<Relu>(ng_batch_norm)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_relu, ov::Rank(4));
      return {ng_relu};
    } else if (vec_str_cmp(fused_ops, {"FusedBatchNorm", "Relu6"})) {
      auto ng_relu6 = make_shared<Clamp>(ng_batch_norm, 0, 6)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_relu6, ov::Rank(4));
      return {ng_relu6};
    } else if (vec_str_cmp(fused_ops, {"FusedBatchNorm", "LeakyRelu"})) {
      auto tf_leakyrelu_alpha = node.get_attribute<float>("leakyrelu_alpha");
      auto ng_leakyrelu_alpha =
          make_shared<Constant>(element::f32, Shape{}, tf_leakyrelu_alpha)
              ->output(0);
      auto ng_alphax =
          make_shared<Multiply>(ng_leakyrelu_alpha, ng_batch_norm)->output(0);
      auto ng_lrelu = make_shared<Maximum>(ng_alphax, ng_batch_norm)->output(0);
      convert_nchw_to_nhwc(is_nhwc, ng_lrelu, ov::Rank(4));
      return {ng_lrelu};
    } else {
      convert_nchw_to_nhwc(is_nhwc, ng_batch_norm, ov::Rank(4));
      return {ng_batch_norm};
    }
  } else {
    FRONT_END_THROW("Unsupported _FusedConv2D ");
  }
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
