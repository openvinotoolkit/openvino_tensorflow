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
#include "gtest/gtest.h"

#include "ngraph_builder.h"
#include "ngraph_utils.h"

using namespace std;

namespace ngraph_bridge {

// valid padding is a noop
TEST(padding, valid) {
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};
  Builder::MakePadding(std::string("VALID"), ng::Shape{4, 5}, ng::Shape{2, 3},
      ng::Strides{1, 1}, ng_padding_below, ng_padding_above);
  ASSERT_EQ(ng_padding_below[0], 0);
  ASSERT_EQ(ng_padding_below[1], 0);
  ASSERT_EQ(ng_padding_above[0], 0);
  ASSERT_EQ(ng_padding_above[1], 0);
}

TEST(padding, divisible) {
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};
  Builder::MakePadding(std::string("SAME"), ng::Shape{8, 9}, ng::Shape{2, 3},
      ng::Strides{1, 1}, ng_padding_below, ng_padding_above);
  ASSERT_EQ(ng_padding_below[0], 0);
  ASSERT_EQ(ng_padding_below[1], 1);
  ASSERT_EQ(ng_padding_above[0], 1);
  ASSERT_EQ(ng_padding_above[1], 1);
}

TEST(padding, indivisible) {
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};
  Builder::MakePadding(std::string("SAME"), ng::Shape{10, 10}, ng::Shape{4, 4},
      ng::Strides{3, 3}, ng_padding_below, ng_padding_above);
  ASSERT_EQ(ng_padding_below[0], 1);
  ASSERT_EQ(ng_padding_below[1], 1);
  ASSERT_EQ(ng_padding_above[0], 2);
  ASSERT_EQ(ng_padding_above[1], 2);
}
}
