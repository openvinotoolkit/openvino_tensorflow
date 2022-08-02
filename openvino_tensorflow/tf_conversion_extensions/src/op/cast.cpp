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

OutputVector translate_cast_op(
    const ov::frontend::NodeContext& node) {
    auto ng_input = node.get_input(0);

    auto ng_et = node.get_attribute<element::Type>("DstT");
    auto ng_input_dtype = ng_input.get_element_type();


    if (ng_et == ov::element::boolean && (ng_input_dtype == ov::element::f32 ||
                                          ng_input_dtype == ov::element::f64)) {
      auto zeros = make_shared<Constant>(ng_input_dtype, ov::Shape{}, 0);
      auto res = make_shared<NotEqual>(ng_input, zeros);
      return res->outputs();
    }

    auto res = make_shared<Convert>(ng_input, ng_et);
    return res->outputs();



    //auto res = make_shared<Convert>(ng_input, ng_et);
    ////set_node_name(node.get_name(), res);
    //return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
