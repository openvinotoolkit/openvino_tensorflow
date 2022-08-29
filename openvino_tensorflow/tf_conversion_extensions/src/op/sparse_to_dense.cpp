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

OutputVector translate_sparse_to_dense_op(
    const ov::frontend::NodeContext& node) {
    auto ng_indices = node.get_input(0);
    auto ng_dense_shape = node.get_input(1);
    auto ng_values = node.get_input(2); 
    auto ng_zeros = node.get_input(3);
    auto ng_dense_tensor =  make_shared<Broadcast>(ng_zeros, ng_dense_shape);

    // Scatter the values at the given indices
    auto ng_scatternd_op = make_shared<ScatterNDUpdate>(ng_dense_tensor, ng_indices, ng_values);

    return ng_scatternd_op->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
