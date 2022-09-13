// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extensions.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;
namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_expand_dims_op(const ov::frontend::NodeContext& node) {
  auto input = node.get_input(0);
  auto dims = node.get_input(1);

  auto backend_name = node.get_attribute<std::string>("_ovtf_backend_name");


  if (backend_name == "MYRIAD" || backend_name == "HDDL") {
    auto dims_const_in = get_constant_from_source(dims);
    const auto& dims_vec = dims_const_in->cast_vector<int64_t>();
    dims = make_shared<Constant>(ov::element::i64, ov::Shape{dims_vec.size()}, dims_vec);
  }

  auto res = make_shared<Unsqueeze>(input, dims);
  //set_node_name(node.get_name(), res);
  return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
