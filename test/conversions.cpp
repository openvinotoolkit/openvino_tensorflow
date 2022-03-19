/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include "gtest/gtest.h"

#include "openvino_tensorflow/layout_conversions.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {
namespace testing {

TEST(conversions, transpose) {
  ov::Output<ov::Node> node = make_shared<ngraph::op::Parameter>(
      ov::element::f32, ov::Shape{2, 3, 4, 5});
  Transpose<3, 2, 0, 1>(node);
  ASSERT_EQ(node.get_shape(), (ov::Shape{5, 4, 2, 3}));
}

TEST(conversions, batch_to_tensorflow_nchw) {
  auto shape = ov::Shape{2, 3, 4, 5};
  ov::Output<ov::Node> node =
      make_shared<ngraph::op::Parameter>(ov::element::f32, shape);
  NCHWtoNHWC("tag", false, node);
  ASSERT_EQ(node.get_shape(), shape);
}

TEST(conversions, batch_to_tensorflow_nhwc) {
  auto shape = ov::Shape{2, 3, 4, 5};
  ov::Output<ov::Node> node =
      make_shared<ngraph::op::Parameter>(ov::element::f32, shape);
  NCHWtoNHWC("tag", true, node);
  ASSERT_EQ(node.get_shape(), (ov::Shape{2, 4, 5, 3}));
}

TEST(conversions, batch_to_ovtf_nchw) {
  auto shape = ov::Shape{2, 3, 4, 5};
  ov::Output<ov::Node> node =
      make_shared<ngraph::op::Parameter>(ov::element::f32, shape);
  NHWCtoNCHW("tag", false, node);
  ASSERT_EQ(node.get_shape(), shape);
}

TEST(conversions, param_to_ovtf_nchw) {
  vector<size_t> in1{1, 2, 3, 4};
  vector<size_t> out1(4);
  NHWCtoHW(false, in1, out1);
  ASSERT_EQ(out1[0], in1[2]);
  ASSERT_EQ(out1[1], in1[3]);
}

TEST(conversions, batch_to_ovtf_nhwc) {
  auto shape = ov::Shape{2, 3, 4, 5};
  ov::Output<ov::Node> node =
      make_shared<ngraph::op::Parameter>(ov::element::f32, shape);
  NHWCtoNCHW("tag", true, node);
  ASSERT_EQ(node.get_shape(), (ov::Shape{2, 5, 3, 4}));
}

TEST(conversions, param_to_ovtf_nhwc) {
  vector<size_t> in1{1, 2, 3, 4};
  vector<size_t> out1(4);
  NHWCtoHW(true, in1, out1);
  ASSERT_EQ(out1[0], in1[1]);
  ASSERT_EQ(out1[1], in1[2]);
}

}  // namespace testing
}  // namespace openvino_tensorflow
}  // namespace tensorflow
