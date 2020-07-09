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

#ifndef NGRAPH_TF_BRIDGE_CONVERSIONS_H_
#define NGRAPH_TF_BRIDGE_CONVERSIONS_H_
#pragma once

#include "ngraph/opsets/opset3.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_builder.h"

namespace tensorflow {

namespace ngraph_bridge {

template <size_t a, size_t b, size_t c, size_t d>
void Transpose(std::shared_ptr<ngraph::Node>& ng_node) {
  static_assert(a < 4 && b < 4 && c < 4 && d < 4,
                "Number of dimensions cannot exceed 4");
  static_assert(a != b && a != c && a != d && b != c && b != d && c != d,
                "Dimensions indices cannot be equal");
  auto& s = ng_node->get_shape();
  ngraph::Shape reshaped_shape{s[a], s[b], s[c], s[d]};
  ngraph::Shape transpose_order{a, b, c, d};
  NGRAPH_VLOG(3) << "transposing " << ngraph::join(s) << " to "
                 << ngraph::join(reshaped_shape) << "axis-order "
                 << ngraph::join(transpose_order);
  auto ng_input_order = std::make_shared<ngraph::opset3::Constant>(
      ngraph::element::u64, ngraph::Shape{transpose_order.size()},
      transpose_order);
  ng_node =
      std::make_shared<ngraph::opset3::Transpose>(ng_node, ng_input_order);
}

template <size_t a, size_t b, size_t c, size_t d, size_t e>
void Transpose3D(std::shared_ptr<ngraph::Node>& ng_node) {
  static_assert(a < 5 && b < 5 && c < 5 && d < 5 && e < 5,
                "Number of dimensions cannot exceed 5");
  static_assert(a != b && a != c && a != d && a != e && b != c && b != d &&
                    b != e && c != d && c != e && d != e,
                "Dimensions indices cannot be equal");
  auto& s = ng_node->get_shape();
  ngraph::Shape reshaped_shape{s[a], s[b], s[c], s[d], s[e]};
  ngraph::Shape transpose_order{a, b, c, d, e};
  NGRAPH_VLOG(3) << "transposing " << ngraph::join(s) << " to "
                 << ngraph::join(reshaped_shape) << "axis-order "
                 << ngraph::join(transpose_order);
  auto ng_input_order = std::make_shared<ngraph::opset3::Constant>(
      ngraph::element::u64, ngraph::Shape{transpose_order.size()},
      transpose_order);
  ng_node =
      std::make_shared<ngraph::opset3::Transpose>(ng_node, ng_input_order);
}

namespace detail {
template <typename T>
void NhwcToNGraph(const std::vector<T>& src, std::vector<size_t>& dst) {
  dst[0] = src[1];
  dst[1] = src[2];
}

void NhwcToNGraph(std::shared_ptr<ngraph::Node>& ng_node);

template <typename T>
void NdhwcToNGraph(const std::vector<T>& src, std::vector<size_t>& dst) {
  dst[0] = src[1];
  dst[1] = src[2];
  dst[2] = src[3];
}

void NdhwcToNGraph(std::shared_ptr<ngraph::Node>& ng_node);

template <typename T>
void NchwToNGraph(const std::vector<T>& src, std::vector<size_t>& dst) {
  dst[0] = src[2];
  dst[1] = src[3];
}

template <typename T>
void NcdhwToNGraph(const std::vector<T>& src, std::vector<size_t>& dst) {
  dst[0] = src[2];
  dst[1] = src[3];
  dst[2] = src[4];
}

template <typename T>
void NhwcToNchw(const std::vector<T>& src, std::vector<size_t>& dst) {
  dst[0] = src[0];
  dst[1] = src[3];
  dst[2] = src[1];
  dst[3] = src[2];
}

template <typename T>
void NdhwcToNcdhw(const std::vector<T>& src, std::vector<size_t>& dst) {
  dst[0] = src[0];
  dst[1] = src[4];
  dst[2] = src[1];
  dst[3] = src[2];
  dst[4] = src[3];
}
}

void BatchToNGraph(const string& op_name, bool is_nhwc,
                   std::shared_ptr<ngraph::Node>& ng_input);

void BatchToNGraph3D(const string& op_name, bool is_ndhwc,
                     std::shared_ptr<ngraph::Node>& ng_input);

template <typename T>
void BatchedOpParamToNGraph(bool is_nhwc, const std::vector<T>& src,
                            std::vector<size_t>& dst) {
  if (is_nhwc) {
    detail::NhwcToNGraph(src, dst);
  } else {
    detail::NchwToNGraph(src, dst);
  }
}

template <typename T>
void BatchedOpParam3DToNGraph(bool is_ndhwc, const std::vector<T>& src,
                              std::vector<size_t>& dst) {
  if (is_ndhwc) {
    detail::NdhwcToNGraph(src, dst);
  } else {
    detail::NcdhwToNGraph(src, dst);
  }
}

template <typename T>
void BatchedOpParamReshape(bool is_nhwc, const std::vector<T>& src,
                           std::vector<size_t>& dst) {
  if (is_nhwc) {
    detail::NhwcToNchw(src, dst);
  } else {
    dst = src;
  }
}

template <typename T>
void BatchedOpParamReshape3D(bool is_ndhwc, const std::vector<T>& src,
                             std::vector<size_t>& dst) {
  if (is_ndhwc) {
    detail::NdhwcToNcdhw(src, dst);
  } else {
    dst = src;
  }
}

void BatchToTensorflow(const string& op_name, bool is_nhwc,
                       std::shared_ptr<ngraph::Node>& ng_node);

void BatchToTensorflow3D(const string& op_name, bool is_ndhwc,
                         std::shared_ptr<ngraph::Node>& ng_node);

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_CONVERSIONS_H_
