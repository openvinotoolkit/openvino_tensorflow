/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

#pragma once

#include "ngraph_builder.h"
#include "ngraph_log.h"

namespace ngraph_bridge {
template <size_t a, size_t b, size_t c, size_t d>
void Reshape(std::shared_ptr<ng::Node>& ng_node) {
  static_assert(a < 4 && b < 4 && c < 4 && d < 4,
                "Number of dimensions cannot exceed 4");
  static_assert(a != b && a != c && a != d && b != c && b != d && c != d,
                "Dimensions indices cannot be equal");
  auto& s = ng_node->get_shape();
  ng::Shape reshaped_shape{s[a], s[b], s[c], s[d]};
  NGRAPH_VLOG(3) << "reshaped_shape: " << ng::join(reshaped_shape);
  ng_node = std::make_shared<ng::op::Reshape>(
      ng_node, ng::AxisVector{a, b, c, d}, reshaped_shape);
}

namespace detail {
template <typename T>
void NhwcToNGraph(const std::vector<T>& src, std::vector<size_t>& dst) {
  dst[0] = src[1];
  dst[1] = src[2];
}

void NhwcToNGraph(std::shared_ptr<ng::Node>& ng_node) {
  Reshape<0, 3, 1, 2>(ng_node);
}

template <typename T>
void NchwToNGraph(const std::vector<T>& src, std::vector<size_t>& dst) {
  dst[0] = src[2];
  dst[1] = src[3];
}

template <typename T>
void NhwcToNchw(const std::vector<T>& src, std::vector<size_t>& dst) {
  dst[0] = src[0];
  dst[1] = src[3];
  dst[2] = src[1];
  dst[3] = src[2];
}
}

void BatchToNGraph(bool is_nhwc, std::shared_ptr<ng::Node>& ng_input) {
  if (is_nhwc) {
    detail::NhwcToNGraph(ng_input);
  }
}

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
void BatchedOpParamReshape(bool is_nhwc, const std::vector<T>& src,
                            std::vector<size_t>& dst) {
  if (is_nhwc) {
    detail::NhwcToNchw(src, dst);
  } else {
    dst = src;
  }
}

void BatchToTensorflow(bool is_nhwc, std::shared_ptr<ng::Node>& ng_node) {
  if (!is_nhwc) {
    return;
  }
  Reshape<0, 2, 3, 1>(ng_node);
}
}
