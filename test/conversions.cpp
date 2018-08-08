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

#include "ngraph_conversions.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

TEST(conversions, reshape) {
  std::shared_ptr<ng::Node> ng_node =
      make_shared<ng::op::Parameter>(ng::element::f32, ng::Shape{2, 3, 4, 5});
  Reshape<3, 2, 0, 1>(ng_node);
  ASSERT_EQ(ng_node->get_shape(), (ng::Shape{5, 4, 2, 3}));
}

TEST(conversions, batch_to_tensorflow_nchw) {
  auto shape = ng::Shape{2, 3, 4, 5};
  std::shared_ptr<ng::Node> ng_node =
      make_shared<ng::op::Parameter>(ng::element::f32, shape);
  BatchToTensorflow(false, ng_node);
  ASSERT_EQ(ng_node->get_shape(), shape);
}

TEST(conversions, batch_to_tensorflow_nhwc) {
  auto shape = ng::Shape{2, 3, 4, 5};
  std::shared_ptr<ng::Node> ng_node =
      make_shared<ng::op::Parameter>(ng::element::f32, shape);
  BatchToTensorflow(true, ng_node);
  ASSERT_EQ(ng_node->get_shape(), (ng::Shape{2, 4, 5, 3}));
}

TEST(conversions, batch_to_ngraph_nchw) {
  auto shape = ng::Shape{2, 3, 4, 5};
  std::shared_ptr<ng::Node> ng_node =
      make_shared<ng::op::Parameter>(ng::element::f32, shape);
  BatchToNGraph(false, ng_node);
  ASSERT_EQ(ng_node->get_shape(), shape);
}

TEST(conversions, param_to_ngraph_nchw) {
  vector<size_t> in1{1, 2, 3, 4};
  vector<size_t> out1(4);
  BatchedOpParamToNGraph(false, in1, out1);
  ASSERT_EQ(out1[0], in1[2]);
  ASSERT_EQ(out1[1], in1[3]);
}

TEST(conversions, batch_to_ngraph_nhwc) {
  auto shape = ng::Shape{2, 3, 4, 5};
  std::shared_ptr<ng::Node> ng_node =
      make_shared<ng::op::Parameter>(ng::element::f32, shape);
  BatchToNGraph(true, ng_node);
  ASSERT_EQ(ng_node->get_shape(), (ng::Shape{2, 5, 3, 4}));
}

TEST(conversions, param_to_ngraph_nhwc) {
  vector<size_t> in1{1, 2, 3, 4};
  vector<size_t> out1(4);
  BatchedOpParamToNGraph(true, in1, out1);
  ASSERT_EQ(out1[0], in1[1]);
  ASSERT_EQ(out1[1], in1[2]);
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
