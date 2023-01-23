/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include "gtest/gtest.h"

#include "openvino_tensorflow/ovtf_builder.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {
namespace testing {

// valid padding is a noop
TEST(padding, valid) {
  ngraph::CoordinateDiff ng_padding_below;
  ngraph::CoordinateDiff ng_padding_above;
  Builder::MakePadding(string("VALID"), ov::Shape{4, 5}, ov::Shape{2, 3},
                       ngraph::Strides{1, 1}, ngraph::Strides{0, 0},
                       ng_padding_below, ng_padding_above);
  ASSERT_EQ(ng_padding_below[0], 0);
  ASSERT_EQ(ng_padding_below[1], 0);
  ASSERT_EQ(ng_padding_above[0], 0);
  ASSERT_EQ(ng_padding_above[1], 0);
}

TEST(padding, divisible) {
  ngraph::CoordinateDiff ng_padding_below;
  ngraph::CoordinateDiff ng_padding_above;
  Builder::MakePadding(string("SAME"), ov::Shape{8, 9}, ov::Shape{2, 3},
                       ngraph::Strides{1, 1}, ngraph::Strides{1, 1},
                       ng_padding_below, ng_padding_above);
  ASSERT_EQ(ng_padding_below[0], 0);
  ASSERT_EQ(ng_padding_below[1], 1);
  ASSERT_EQ(ng_padding_above[0], 1);
  ASSERT_EQ(ng_padding_above[1], 1);
}

TEST(padding, indivisible) {
  ngraph::CoordinateDiff ng_padding_below;
  ngraph::CoordinateDiff ng_padding_above;
  Builder::MakePadding(string("SAME"), ov::Shape{10, 10}, ov::Shape{4, 4},
                       ngraph::Strides{3, 3}, ngraph::Strides{1, 1},
                       ng_padding_below, ng_padding_above);
  ASSERT_EQ(ng_padding_below[0], 1);
  ASSERT_EQ(ng_padding_below[1], 1);
  ASSERT_EQ(ng_padding_above[0], 2);
  ASSERT_EQ(ng_padding_above[1], 2);
}

}  // namespace testing
}  // namespace openvino_tensorflow
}  // namespace tensorflow
