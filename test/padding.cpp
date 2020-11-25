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
#include "gtest/gtest.h"

#include "ngraph_bridge/ngraph_builder.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

// valid padding is a noop
TEST(padding, valid) {
  ngraph::CoordinateDiff ng_padding_below;
  ngraph::CoordinateDiff ng_padding_above;
  Builder::MakePadding(string("VALID"), ngraph::Shape{4, 5},
                       ngraph::Shape{2, 3}, ngraph::Strides{1, 1},
                       ngraph::Strides{0, 0}, ng_padding_below,
                       ng_padding_above);
  ASSERT_EQ(ng_padding_below[0], 0);
  ASSERT_EQ(ng_padding_below[1], 0);
  ASSERT_EQ(ng_padding_above[0], 0);
  ASSERT_EQ(ng_padding_above[1], 0);
}

TEST(padding, divisible) {
  ngraph::CoordinateDiff ng_padding_below;
  ngraph::CoordinateDiff ng_padding_above;
  Builder::MakePadding(string("SAME"), ngraph::Shape{8, 9}, ngraph::Shape{2, 3},
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
  Builder::MakePadding(string("SAME"), ngraph::Shape{10, 10},
                       ngraph::Shape{4, 4}, ngraph::Strides{3, 3},
                       ngraph::Strides{1, 1}, ng_padding_below,
                       ng_padding_above);
  ASSERT_EQ(ng_padding_below[0], 1);
  ASSERT_EQ(ng_padding_below[1], 1);
  ASSERT_EQ(ng_padding_above[0], 2);
  ASSERT_EQ(ng_padding_above[1], 2);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
