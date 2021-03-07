/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#pragma once

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/util.hpp"

namespace tensorflow {
namespace openvino_tensorflow {
namespace pass {

class TransposeSinking : public ngraph::pass::FunctionPass {
 public:
  TransposeSinking() {
    set_property(ngraph::pass::PassProperty::REQUIRE_STATIC_SHAPE, true);
  }
  bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
};

}  // namespace pass
}  // namespace openvino_tensorflow
}  // namespace tensorflow
