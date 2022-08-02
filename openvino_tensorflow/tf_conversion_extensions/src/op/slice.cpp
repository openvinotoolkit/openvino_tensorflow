// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extensions.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/core/validation_util.hpp"

using namespace std;
using namespace ov::opset8;
namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_slice_op(
    const ov::frontend::NodeContext& node) {
    auto input = node.get_input(0);
    auto start = node.get_input(1);
    auto size = node.get_input(2);

    auto size_const_in = get_constant_from_source(size);
    const auto& size_vector = size_const_in->cast_vector<int64_t>();

    auto start_const_in = get_constant_from_source(start);
    const auto& start_vector = start_const_in->cast_vector<int64_t>();

    std::vector<int64_t> stop_vector(size_vector.size());

    if (size_vector.size() != input.get_shape().size()) {
        throw runtime_error("ConversionExtension error ("+node.get_op_type()+"): Size vector length is not equal to number of input dimensions.");
    }

    for (int i=0; i<size_vector.size(); i++) {
        if (size_vector[i] == -1) {
            stop_vector[i] = input.get_shape()[i];
        } else {
            stop_vector[i] = start_vector[i] + size_vector[i];
        }
    }

    auto start_const = make_shared<Constant>(element::i64, Shape{start_vector.size()}, start_vector);
    auto stop_const = make_shared<Constant>(element::i64, Shape{stop_vector.size()}, stop_vector);

    auto one = make_shared<Constant>(element::i64, Shape{1}, 1);
    auto shape = make_shared<ShapeOf>(start);
    auto step = make_shared<Broadcast>(one, shape);

    size_t input_dims = input.get_partial_shape().rank().get_length();
    if (input.get_partial_shape().is_static() && input_dims > 0 && input.get_shape()[0] == 0) {
        auto res = make_shared<Constant>(input.get_element_type(), ov::Shape{0}, std::vector<int>({0}));
        //set_node_name(node.get_name(), res);
        return res->outputs();
    } else {
        auto res = make_shared<Slice>(input, start_const, stop_const, step);
        //set_node_name(node.get_name(), res);
        return res->outputs();
    }
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
