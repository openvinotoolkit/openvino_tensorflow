/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/pass/transpose_folding.h"
#include "ngraph_bridge/version.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace pass {

bool TransposeFolding::run_on_function(shared_ptr<ngraph::Function> f) {
  for (auto n1 : f->get_ordered_ops()) {
    if (auto t1 = ngraph::as_type_ptr<ngraph::opset3::Transpose>(n1)) {
      // check if the next node is also a transpose
      auto n2 = t1->input_value(0).get_node_shared_ptr();
      auto t2 = ngraph::as_type_ptr<ngraph::opset3::Transpose>(n2);
      if (t2) {
        auto const1 = ngraph::as_type_ptr<ngraph::opset3::Constant>(
            n1->get_input_node_shared_ptr(1));
        auto const2 = ngraph::as_type_ptr<ngraph::opset3::Constant>(
            n2->get_input_node_shared_ptr(1));
        if (const1 && const2) {
          // apply the permutations
          auto default_order = ngraph::get_default_order(t1->get_shape());
          auto perm_t1 = ngraph::apply_permutation(
              default_order, const1->get_axis_vector_val());
          auto perm_t2 =
              ngraph::apply_permutation(perm_t1, const2->get_axis_vector_val());

          // check if the two transposes cancel each other out
          if (default_order == perm_t2) {
            NGRAPH_VLOG(4)
                << "TransposeFolding: Eliminating consecutive transposes:"
                << " t1 i/o = " << ngraph::join(const1->get_axis_vector_val())
                << " t2 i/o = " << ngraph::join(const2->get_axis_vector_val());
            ngraph::replace_node(t1, t2->input_value(0).get_node_shared_ptr());
          } else {
            // delete the second transpose first before replacing with the
            // combined transpose
            ngraph::replace_node(t2, t2->input_value(0).get_node_shared_ptr());
            auto input_order = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::u64, ngraph::Shape{perm_t2.size()}, perm_t2);
            auto combined_node = std::make_shared<ngraph::opset3::Transpose>(
                t1->input_value(0), input_order);
            ngraph::replace_node(t1, combined_node);
          }
        }
      }
    }
  }

  for (auto n : f->get_ordered_ops()) {
    n->revalidate_and_infer_types();
  }
  return true;
}

}  // namespace pass
}  // namespace ngraph_bridge
}  // namespace tensorflow
