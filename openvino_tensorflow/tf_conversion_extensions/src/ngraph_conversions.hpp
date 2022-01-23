// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

template <size_t a, size_t b, size_t c, size_t d>
void Transpose(ov::Output<ov::Node>& node) {
  static_assert(a < 4 && b < 4 && c < 4 && d < 4,
                "Number of dimensions cannot exceed 4");
  static_assert(a != b && a != c && a != d && b != c && b != d && c != d,
                "Dimensions indices cannot be equal");
  auto& s = node.get_shape();
  ov::Shape reshaped_shape{s[a], s[b], s[c], s[d]};
  ov::Shape transpose_order{a, b, c, d};
  auto input_order = std::make_shared<ov::opset8::Constant>(
      ov::element::u64, ov::Shape{transpose_order.size()}, transpose_order);
  node = std::make_shared<ov::opset8::Transpose>(node, input_order);
}

template <size_t a, size_t b, size_t c, size_t d>
void Transpose(std::shared_ptr<ov::Node>& node) {
  Transpose<a, b, c, d>(node->get_default_output());
}

template <size_t a, size_t b, size_t c, size_t d, size_t e>
void Transpose3D(ov::Output<ov::Node>& node) {
  static_assert(a < 5 && b < 5 && c < 5 && d < 5 && e < 5,
                "Number of dimensions cannot exceed 5");
  static_assert(a != b && a != c && a != d && a != e && b != c && b != d &&
                    b != e && c != d && c != e && d != e,
                "Dimensions indices cannot be equal");
  auto& s = node.get_shape();
  ov::Shape reshaped_shape{s[a], s[b], s[c], s[d], s[e]};
  ov::Shape transpose_order{a, b, c, d, e};
  auto input_order = std::make_shared<ov::opset8::Constant>(
      ov::element::u64, ov::Shape{transpose_order.size()}, transpose_order);
  node = std::make_shared<ov::opset8::Transpose>(node, input_order);
}

template <size_t a, size_t b, size_t c, size_t d, size_t e>
void Transpose3D(std::shared_ptr<ov::Node>& node) {
  Transpose3D<a, b, c, d, e>(node->get_default_output());
}

namespace detail {
template <typename T>
void convert_nhwc_to_hw(const std::vector<T>& src, std::vector<size_t>& dst) {
  if (dst.size() >= 2) {
    dst[0] = src[1];
    dst[1] = src[2];
  }
  if (dst.size() >= 3) {
    dst[2] = src[3];
  }
}

template <typename T>
void convert_nchw_to_hw(const std::vector<T>& src, std::vector<size_t>& dst) {
  if (dst.size() >= 2) {
    dst[0] = src[2];
    dst[1] = src[3];
  }
  if (dst.size() >= 3) {
    dst[2] = src[4];
  }
}
}  // namespace detail

void convert_nhwc_to_nchw(bool need_convert, ov::Output<ov::Node>& ng_input);

void convert_nchw_to_nhwc(bool need_convert, ov::Output<ov::Node>& ng_node);

template <typename T>
void convert_nhwc_to_hw(bool is_nhwc, const std::vector<T>& src,
                        std::vector<size_t>& dst) {
  if (is_nhwc) {
    detail::convert_nhwc_to_hw(src, dst);
  } else {
    detail::convert_nchw_to_hw(src, dst);
  }
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
