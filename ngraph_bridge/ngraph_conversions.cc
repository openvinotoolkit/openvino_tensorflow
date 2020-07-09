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
#include "ngraph_bridge/ngraph_api.h"

namespace tensorflow {

namespace ngraph_bridge {

namespace detail {

void NhwcToNGraph(std::shared_ptr<ngraph::Node>& ng_node) {
  Transpose<0, 3, 1, 2>(ng_node);
}

void NdhwcToNGraph(std::shared_ptr<ngraph::Node>& ng_node) {
  Transpose3D<0, 4, 1, 2, 3>(ng_node);
}
}  // namespace detail

void BatchToNGraph(const string& op_name, bool is_nhwc,
                   std::shared_ptr<ngraph::Node>& ng_input) {
  if (is_nhwc) {
    detail::NhwcToNGraph(ng_input);
    Builder::SetTracingInfo(op_name, ng_input);
  }
}

void BatchToNGraph3D(const string& op_name, bool is_ndhwc,
                     std::shared_ptr<ngraph::Node>& ng_input) {
  if (is_ndhwc) {
    detail::NdhwcToNGraph(ng_input);
    Builder::SetTracingInfo(op_name, ng_input);
  }
}

void BatchToTensorflow(const string& op_name, bool is_nhwc,
                       std::shared_ptr<ngraph::Node>& ng_node) {
  if (!is_nhwc) {
    return;
  }
  Transpose<0, 2, 3, 1>(ng_node);
  Builder::SetTracingInfo(op_name, ng_node);
}

void BatchToTensorflow3D(const string& op_name, bool is_ndhwc,
                         std::shared_ptr<ngraph::Node>& ng_node) {
  if (!is_ndhwc) {
    return;
  }
  Transpose3D<0, 2, 3, 4, 1>(ng_node);
  Builder::SetTracingInfo(op_name, ng_node);
}
}  // namespace ngraph_bridge

}  // namespace tensorflow
