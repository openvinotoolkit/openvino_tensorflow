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

#include "ngraph_bridge/ngraph_conversions.h"

namespace tensorflow {
namespace ngraph_bridge {

void NHWCtoNCHW(const string& op_name, bool is_nhwc,
                ngraph::Output<ngraph::Node>& node) {
  if (is_nhwc) {
    auto rank = node.get_shape().size();
    if (rank == 4) {
      Transpose<0, 3, 1, 2>(node);
    } else if (rank == 5) {
      Transpose3D<0, 4, 1, 2, 3>(node);
    }
    Builder::SetTracingInfo(op_name, node);
  }
}

void NCHWtoNHWC(const string& op_name, bool is_nhwc,
                ngraph::Output<ngraph::Node>& node) {
  if (is_nhwc) {
    auto rank = node.get_shape().size();
    if (rank == 4) {
      Transpose<0, 2, 3, 1>(node);
    } else if (rank == 5) {
      Transpose3D<0, 2, 3, 4, 1>(node);
    }
    Builder::SetTracingInfo(op_name, node);
  }
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
