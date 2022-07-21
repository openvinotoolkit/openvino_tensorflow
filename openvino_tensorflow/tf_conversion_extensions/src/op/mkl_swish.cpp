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

OutputVector translate_mkl_swish_op(const ov::frontend::NodeContext& node) {
  auto ng_input = node.get_input(0);
  auto ng_sigmoid = make_shared<Sigmoid>(ng_input);
  auto ng_result = make_shared<Multiply>(ng_input, ng_sigmoid);
  return {ng_result};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
