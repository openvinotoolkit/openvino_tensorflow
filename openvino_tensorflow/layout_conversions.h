/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef OPENVINO_TF_BRIDGE_CONVERSIONS_H_
#define OPENVINO_TF_BRIDGE_CONVERSIONS_H_
#pragma once

#include "logging/ovtf_log.h"
#include "openvino_tensorflow/default_opset.h"
#include "openvino_tensorflow/ovtf_builder.h"

namespace tensorflow {
namespace openvino_tensorflow {

template <size_t a, size_t b, size_t c, size_t d>
void Transpose(ngraph::Output<ngraph::Node>& node) {
  static_assert(a < 4 && b < 4 && c < 4 && d < 4,
                "Number of dimensions cannot exceed 4");
  static_assert(a != b && a != c && a != d && b != c && b != d && c != d,
                "Dimensions indices cannot be equal");
  auto& s = node.get_shape();
  ngraph::Shape reshaped_shape{s[a], s[b], s[c], s[d]};
  ngraph::Shape transpose_order{a, b, c, d};
  OVTF_VLOG(3) << "transposing " << ngraph::join(s) << " to "
                 << ngraph::join(reshaped_shape) << " axis-order "
                 << ngraph::join(transpose_order);
  auto input_order = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{transpose_order.size()},
      transpose_order);
  node = std::make_shared<opset::Transpose>(node, input_order);
}

template <size_t a, size_t b, size_t c, size_t d>
void Transpose(std::shared_ptr<ngraph::Node>& node) {
  Transpose<a, b, c, d>(node->get_default_output());
}

template <size_t a, size_t b, size_t c, size_t d, size_t e>
void Transpose3D(ngraph::Output<ngraph::Node>& node) {
  static_assert(a < 5 && b < 5 && c < 5 && d < 5 && e < 5,
                "Number of dimensions cannot exceed 5");
  static_assert(a != b && a != c && a != d && a != e && b != c && b != d &&
                    b != e && c != d && c != e && d != e,
                "Dimensions indices cannot be equal");
  auto& s = node.get_shape();
  ngraph::Shape reshaped_shape{s[a], s[b], s[c], s[d], s[e]};
  ngraph::Shape transpose_order{a, b, c, d, e};
  OVTF_VLOG(3) << "transposing " << ngraph::join(s) << " to "
                 << ngraph::join(reshaped_shape) << "axis-order "
                 << ngraph::join(transpose_order);
  auto input_order = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{transpose_order.size()},
      transpose_order);
  node = std::make_shared<opset::Transpose>(node, input_order);
}

template <size_t a, size_t b, size_t c, size_t d, size_t e>
void Transpose3D(std::shared_ptr<ngraph::Node>& node) {
  Transpose3D<a, b, c, d, e>(node->get_default_output());
}

namespace detail {
template <typename T>
void NHWCtoHW(const std::vector<T>& src, std::vector<size_t>& dst) {
  if (dst.size() >= 2) {
    dst[0] = src[1];
    dst[1] = src[2];
  }
  if (dst.size() >= 3) {
    dst[2] = src[3];
  }
}

template <typename T>
void NCHWtoHW(const std::vector<T>& src, std::vector<size_t>& dst) {
  if (dst.size() >= 2) {
    dst[0] = src[2];
    dst[1] = src[3];
  }
  if (dst.size() >= 3) {
    dst[2] = src[4];
  }
}
}

void NHWCtoNCHW(const string& op_name, bool is_nhwc,
                ngraph::Output<ngraph::Node>& ng_input);

void NCHWtoNHWC(const string& op_name, bool is_nhwc,
                ngraph::Output<ngraph::Node>& ng_node);

template <typename T>
void NHWCtoHW(bool is_nhwc, const std::vector<T>& src,
              std::vector<size_t>& dst) {
  if (is_nhwc) {
    detail::NHWCtoHW(src, dst);
  } else {
    detail::NCHWtoHW(src, dst);
  }
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_BRIDGE_CONVERSIONS_H_
