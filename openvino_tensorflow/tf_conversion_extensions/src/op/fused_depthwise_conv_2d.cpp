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

OutputVector translate_depthwise_conv_2d_native_op(const ov::frontend::NodeContext& node) {
    auto ng_input = node.get_input(0);
    auto ng_filter = node.get_input(1);
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int64_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");
    FRONT_END_GENERAL_CHECK(tf_data_format == "NHWC" || tf_data_format == "NCHW",
                            "DepthwiseConv2D data format is neither NHWC nor NCHW");
    bool is_nhwc = (tf_data_format == "NHWC");
    Strides ng_strides(2);
    Strides ng_dilations(2);
    Shape ng_image_shape(2);
    Shape ng_kernel_shape(2);
    convert_nhwc_to_hw(is_nhwc, ng_input.get_shape(), ng_image_shape);
    convert_nhwc_to_hw(is_nhwc, tf_strides, ng_strides);
    convert_nhwc_to_hw(is_nhwc, tf_dilations, ng_dilations);
    convert_nhwc_to_nchw(is_nhwc, ng_input);
    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    CoordinateDiff ng_padding_below;
    CoordinateDiff ng_padding_above;
    make_padding(tf_padding_type,
                 ng_image_shape,
                 ng_kernel_shape,
                 ng_strides,
                 ng_dilations,
                 ng_padding_below,
                 ng_padding_above);
    // H W I M -> H W I 1 M
    auto filter_shape = make_shared<Constant>(
        element::u64,
        Shape{5},
        ov::Shape{ng_filter_shape[0], ng_filter_shape[1], ng_filter_shape[2], 1, ng_filter_shape[3]});
    auto reshaped_filter = make_shared<Reshape>(ng_filter, filter_shape, false);
    // H W I 1 M -> I M 1 H W
    auto order = make_shared<Constant>(element::i64, Shape{5}, vector<int64_t>{2, 4, 3, 0, 1});
    auto transposed_filter = make_shared<opset8::Transpose>(reshaped_filter, order);
    auto ng_conv_node = make_shared<GroupConvolution>(ng_input,
                                                      transposed_filter,
                                                      ng_strides,
                                                      ng_padding_below,
                                                      ng_padding_above,
                                                      ng_dilations);
    auto ng_conv = ng_conv_node->output(0);

    auto op_type = node.get_op_type();
    if (op_type == "DepthwiseConv2dNative") {
        convert_nchw_to_nhwc(is_nhwc, ng_conv);
        return {ng_conv};
    } else if (op_type == "_FusedDepthwiseConv2dNative") {
        int num_args = node.get_attribute<int>("num_args");
        auto fused_ops = node.get_attribute<vector<string>>("fused_ops");
        FRONT_END_GENERAL_CHECK(vec_str_cmp(fused_ops, {"BiasAdd"}) || vec_str_cmp(fused_ops, {"BiasAdd", "Relu6"}),
                                "Unsupported fused operations.");
        FRONT_END_GENERAL_CHECK(num_args == 1, "FusedDepthwiseConv2dNativeBiasAdd has incompatible num_args");
        auto ng_bias = node.get_input(2);

        auto ng_conv_shape = ng_conv.get_shape();
        auto ng_bias_shape = ng_bias.get_shape();
        FRONT_END_GENERAL_CHECK(ng_bias_shape.size() == 1, "Bias argument to BiasAdd does not have one dimension");

        std::vector<size_t> reshape_pattern_values(ng_conv_shape.size(), 1U);
        reshape_pattern_values[1] = ng_bias.get_shape().front();
        auto reshape_pattern =
            make_shared<Constant>(element::u64, Shape{reshape_pattern_values.size()}, reshape_pattern_values);
        auto ng_bias_reshaped = make_shared<Reshape>(ng_bias, reshape_pattern, false);

        auto ng_add = make_shared<Add>(ng_conv, ng_bias_reshaped)->output(0);

        if (fused_ops == vector<string>{"BiasAdd", "Relu6"}) {
            auto ng_relu6 = make_shared<Clamp>(ng_add, 0, 6)->output(0);
            convert_nchw_to_nhwc(is_nhwc, ng_relu6);
            return {ng_relu6};
        } else {
            convert_nchw_to_nhwc(is_nhwc, ng_add);
            return {ng_add};
        }
    }
    FRONT_END_GENERAL_CHECK(false, "Unsupported operation type.");
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov