//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/pass/transpose_sinking.h"
#include "test/opexecuter.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

TEST(TransposeSinking, PassProperty) {
  auto pass = std::make_shared<pass::TransposeSinking>();
  ASSERT_TRUE(
      pass->get_property(ngraph::pass::PassProperty::REQUIRE_STATIC_SHAPE));
  ASSERT_FALSE(
      pass->get_property(ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE));
}

TEST(TransposeSinking, EdgeSplitting) {
  // checks if Transpose is pushed through ng::opset3::Abs, but stopped by
  // ReduceSum
  ng::Shape shape_nhwc{16, 28, 28, 1};
  ng::Shape shape_nchw{16, 1, 28, 28};

  auto a = make_shared<ng::opset3::Parameter>(ng::element::i32, shape_nhwc);
  auto ng_order = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 3, 1, 2});
  auto transpose = make_shared<ng::opset3::Transpose>(a, ng_order);
  auto absn = make_shared<ng::opset3::Abs>(transpose);
  auto absn2 = make_shared<ng::opset3::Abs>(absn);

  auto axes = make_shared<ng::opset3::Constant>(ng::element::i64, ng::Shape{4},
                                                vector<int64_t>{0, 1, 2, 3});
  auto sum = make_shared<ng::opset3::ReduceSum>(transpose, axes, true);

  auto func = make_shared<ng::Function>(ng::OutputVector{absn2, sum},
                                        ng::ParameterVector{a});
  ng::pass::Manager pass_manager;
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);
  ASSERT_EQ(func->get_results().at(1)->input_value(0), sum);
  auto new_transpose = ng::as_type_ptr<ng::opset3::Transpose>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), shape_nchw);
}

//            X (NHWC)
//            |
//         Transpose
//            |
//         AvgPool (NCHW)
//            |
//         Transpose
//            |   Const (NHWC)
//            |   /
//            |  /
//            | /
//           Add (NHWC)
//            |
//          Result
TEST(TransposeSinking, PoolAdd1) {
  ng::Shape input_shape{1, 3, 3, 1};  // NHWC (N=1, H=3, W=3, C=1)

  auto input_type = ng::element::f32;
  auto output_type = ng::element::f32;

  auto X = make_shared<ng::opset3::Parameter>(input_type, input_shape);  // NHWC

  auto ng_order1 = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 3, 1, 2});
  auto transpose1 =
      make_shared<ng::opset3::Transpose>(X, ng_order1);  // NCHW (1,1,3,3)

  auto avgpool = make_shared<ng::opset3::AvgPool>(
      transpose1, ng::Strides{1, 1}, ng::Shape{0, 0}, ng::Shape{0, 0},
      ng::Shape{1, 1}, true, ng::op::RoundingType::FLOOR,
      ng::op::PadType::VALID);

  auto ng_order2 = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 2, 3, 1});
  auto transpose2 =
      make_shared<ng::opset3::Transpose>(avgpool, ng_order2);  // NHWC (1,3,3,1)

  auto const1 = ng::opset3::Constant::create(input_type, ng::Shape{1, 3, 3, 1},
                                             {3});  // NHWC (1,3,3,1)
  auto add1 = make_shared<ng::opset3::Add>(transpose2, const1);
  auto func = make_shared<ng::Function>(add1, ng::ParameterVector{X});

  ng::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<ng::opset3::Transpose>(func);
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);
  size_t after_count = count_ops_of_type<ng::opset3::Transpose>(func);
  ASSERT_LE(before_count, after_count);
  auto new_transpose = ng::as_type_ptr<ng::opset3::Transpose>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), (ng::Shape{1, 3, 3, 1}));
}

TEST(TransposeSinking, PoolAdd2) {
  ng::Shape input_shape{1, 3, 3, 1};  // NHWC (N=1, H=3, W=3, C=1)

  auto input_type = ng::element::f32;
  auto output_type = ng::element::f32;

  auto X = make_shared<ng::opset3::Parameter>(input_type, input_shape);  // NHWC

  auto ng_order1 = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 3, 1, 2});
  auto transpose1 =
      make_shared<ng::opset3::Transpose>(X, ng_order1);  // NCHW (1,1,3,3)

  auto avgpool = make_shared<ng::opset3::AvgPool>(
      transpose1, ng::Strides{1, 1}, ng::Shape{0, 0}, ng::Shape{0, 0},
      ng::Shape{1, 1}, true, ng::op::RoundingType::FLOOR,
      ng::op::PadType::VALID);

  auto ng_order2 = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 2, 3, 1});
  auto transpose2 =
      make_shared<ng::opset3::Transpose>(avgpool, ng_order2);  // NHWC (1,3,3,1)
  auto maxpool = make_shared<ng::opset3::MaxPool>(
      transpose1, ng::Strides{1, 1}, ng::Shape{0, 0}, ng::Shape{0, 0},
      ng::Shape{1, 1}, ng::op::RoundingType::FLOOR, ng::op::PadType::VALID);

  auto ng_order3 = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 2, 3, 1});
  auto transpose3 = make_shared<ng::opset3::Transpose>(maxpool, ng_order3);

  auto const1 = ng::opset3::Constant::create(input_type, ng::Shape{1, 3, 3, 1},
                                             {3});  // NHWC (1,3,3,1)
  auto add1 = make_shared<ng::opset3::Add>(transpose3, const1);
  auto add2 = make_shared<ng::opset3::Add>(add1, transpose2);
  auto func = make_shared<ng::Function>(add2, ng::ParameterVector{X});

  ng::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<ng::opset3::Transpose>(func);
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);
  size_t after_count = count_ops_of_type<ng::opset3::Transpose>(func);
  ASSERT_LE(after_count, before_count);
  auto new_transpose = ng::as_type_ptr<ng::opset3::Transpose>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), (ng::Shape{1, 3, 3, 1}));
}

// Different rank constant input to Add1. After TransposeSinking the const
// would need a Reshape to have the same order as the other input to
// Add1.
TEST(TransposeSinking, PoolAdd3) {
  ng::Shape input_shape{1, 3, 3, 1};  // NHWC (N=1, H=3, W=3, C=1)

  auto input_type = ng::element::f32;
  auto output_type = ng::element::f32;

  auto X = make_shared<ng::opset3::Parameter>(input_type, input_shape);  // NHWC

  auto ng_order1 = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 3, 1, 2});
  auto transpose1 =
      make_shared<ng::opset3::Transpose>(X, ng_order1);  // NCHW (1,1,3,3)

  auto avgpool = make_shared<ng::opset3::AvgPool>(
      transpose1, ng::Strides{1, 1}, ng::Shape{0, 0}, ng::Shape{0, 0},
      ng::Shape{1, 1}, true, ng::op::RoundingType::FLOOR,
      ng::op::PadType::VALID);

  auto ng_order2 = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 2, 3, 1});
  auto transpose2 =
      make_shared<ng::opset3::Transpose>(avgpool, ng_order2);  // NHWC (1,3,3,1)

  auto const1 = ng::opset3::Constant::create(input_type, ng::Shape{1},
                                             {1});  // NHWC (1,3,3,1)
  auto add1 = make_shared<ng::opset3::Add>(transpose2, const1);
  auto func = make_shared<ng::Function>(add1, ng::ParameterVector{X});

  ng::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<ng::opset3::Transpose>(func);
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);
  size_t after_count = count_ops_of_type<ng::opset3::Transpose>(func);
  ASSERT_LE(after_count, before_count);
  auto new_transpose = ng::as_type_ptr<ng::opset3::Transpose>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), (ng::Shape{1, 3, 3, 1}));
}

TEST(TransposeSinking, Concat) {
  // checks if Transpose is pushed through ng::opset3::Concat
  ng::Shape shape_nhwc{16, 28, 28, 1};
  auto a = make_shared<ng::opset3::Parameter>(ng::element::i32, shape_nhwc);
  auto b = make_shared<ng::opset3::Parameter>(ng::element::i32, shape_nhwc);
  auto to_nchw = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 3, 1, 2});
  auto a_transpose = make_shared<ng::opset3::Transpose>(a, to_nchw);
  auto b_transpose = make_shared<ng::opset3::Transpose>(b, to_nchw);
  auto concat = make_shared<ng::opset3::Concat>(
      ng::OutputVector{a_transpose, b_transpose}, 0);
  auto to_nhwc = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 2, 3, 1});
  auto c = make_shared<ng::opset3::Transpose>(concat, to_nhwc);
  auto func =
      make_shared<ng::Function>(ng::OutputVector{c}, ng::ParameterVector{a, b});
  ng::pass::Manager pass_manager;
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);
  size_t transpose_count = count_ops_of_type<ng::opset3::Transpose>(func);
  ASSERT_EQ(0, transpose_count);
  auto result = func->get_results().at(0)->input_value(0).get_node_shared_ptr();
  ng::Shape expected_shape{32, 28, 28, 1};
  ASSERT_EQ(result->get_output_shape(0), expected_shape);
}

TEST(TransposeSinking, Concat_DummyShape) {
  // checks if Transpose is pushed through ng::opset3::Concat
  ng::Shape shape1{4, 3, 3, 1};
  ng::Shape shape2{4, 3, 3, 2};
  ng::Shape shape3{4, 3, 3, 3};
  ng::Shape shape4{4, 3, 3, 4};
  auto to_nchw = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 3, 1, 2});
  auto to_nhwc = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 2, 3, 1});

  auto a1 = make_shared<ng::opset3::Parameter>(ng::element::i32, shape1);
  auto a2 = make_shared<ng::opset3::Parameter>(ng::element::i32, shape2);
  auto a3 = make_shared<ng::opset3::Parameter>(ng::element::i32, shape3);
  auto a4 = make_shared<ng::opset3::Parameter>(ng::element::i32, shape4);
  auto a1_transpose = make_shared<ng::opset3::Transpose>(a1, to_nchw);
  auto a2_transpose = make_shared<ng::opset3::Transpose>(a2, to_nchw);
  auto a3_transpose = make_shared<ng::opset3::Transpose>(a3, to_nchw);
  auto a4_transpose = make_shared<ng::opset3::Transpose>(a4, to_nchw);
  auto concat = make_shared<ng::opset3::Concat>(
      ng::NodeVector{a1_transpose, a2_transpose, a3_transpose, a4_transpose},
      1);
  auto out = make_shared<ng::opset3::Transpose>(concat, to_nchw);
  auto func = make_shared<ng::Function>(ng::OutputVector{out},
                                        ng::ParameterVector{a1, a2, a3, a4});
  ng::pass::Manager pass_manager;
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);
  size_t transpose_count = count_ops_of_type<ng::opset3::Transpose>(func);
  ASSERT_EQ(1, transpose_count);
  auto result = func->get_results().at(0)->input_value(0).get_node_shared_ptr();
  ng::Shape expected_shape{4, 3, 10, 3};
  ASSERT_EQ(result->get_output_shape(0), expected_shape);
}

// The Transpose should sink through Pad op but stopped by ReduceSum
TEST(TransposeSinking, Pad) {
  ng::Shape shape_nhwc{100, 8, 8, 1};

  auto a = make_shared<ng::opset3::Parameter>(ng::element::f32, shape_nhwc);
  auto pad_value = ng::opset3::Constant::create<float>(
      ng::element::f32, ng::Shape{}, std::vector<float>{0.0f});

  ng::CoordinateDiff pad_end{0, 0, 0, 0};
  ng::CoordinateDiff pad_begin{0, 1, 1, 0};

  auto a_to_nchw = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 3, 1, 2});
  auto a_transpose = make_shared<ng::opset3::Transpose>(a, a_to_nchw);

  auto maxpool = make_shared<ng::opset3::MaxPool>(
      a_transpose, ng::Strides{2, 2}, ng::Shape{0, 0}, ng::Shape{0, 0},
      ng::Shape{1, 1}, ng::op::RoundingType::FLOOR, ng::op::PadType::VALID);

  auto m_to_nhwc = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 2, 3, 1});
  auto m_transpose = make_shared<ng::opset3::Transpose>(maxpool, m_to_nhwc);

  shared_ptr<ng::opset3::Constant> pads_begin_node, pads_end_node;
  pads_begin_node = make_shared<ng::opset3::Constant>(
      ng::element::i64, ng::Shape{pad_begin.size()}, pad_begin);
  pads_end_node = make_shared<ng::opset3::Constant>(
      ng::element::i64, ng::Shape{pad_end.size()}, pad_end);
  auto pad =
      make_shared<ng::opset3::Pad>(m_transpose, pads_begin_node, pads_end_node,
                                   pad_value, ng::op::PadMode::CONSTANT);

  auto axes = make_shared<ng::opset3::Constant>(ng::element::i64, ng::Shape{4},
                                                vector<int64_t>{0, 1, 2, 3});
  auto sum = make_shared<ng::opset3::ReduceSum>(pad, axes, true);

  auto f =
      make_shared<ng::Function>(ng::OutputVector{sum}, ng::ParameterVector{a});

  ng::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<ng::opset3::Transpose>(f);
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(f);
  size_t after_count = count_ops_of_type<ng::opset3::Transpose>(f);
  ASSERT_LE(after_count, before_count);
  auto result = f->get_results().at(0)->input_value(0).get_node_shared_ptr();
  ng::Shape expected_shape{1, 1, 1, 1};
  ASSERT_EQ(result->get_output_shape(0), expected_shape);
  auto out = ng::as_type_ptr<ng::opset3::ReduceSum>(
      f->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(out);
}

TEST(TransposeSinking, SimpleUnary) {
  ng::Shape shape_nhwc{16, 28, 28, 1};
  ng::Shape shape_nchw{16, 1, 28, 28};
  auto a = make_shared<ng::opset3::Parameter>(ng::element::i32, shape_nhwc);

  auto ng_order = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 3, 1, 2});
  auto transpose = make_shared<ng::opset3::Transpose>(a, ng_order);

  auto a_t = make_shared<ng::opset3::Transpose>(a, ng_order);
  auto absn = make_shared<ng::opset3::Abs>(a_t);
  auto absn2 = make_shared<ng::opset3::Abs>(absn);

  auto tf_order = std::make_shared<ng::opset3::Constant>(
      ng::element::u64, ng::Shape{4}, ng::Shape{0, 2, 3, 1});
  auto absn2_t = make_shared<ng::opset3::Transpose>(absn2, tf_order);

  auto func = make_shared<ng::Function>(ng::OutputVector{absn2_t},
                                        ng::ParameterVector{a});
  size_t before_count = count_ops_of_type<ng::opset3::Transpose>(func);
  ng::pass::Manager pass_manager;
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);
  size_t after_count = count_ops_of_type<ng::opset3::Transpose>(func);
  ASSERT_EQ(func->get_results().at(0)->input_value(0), absn2);
  EXPECT_NE(before_count, after_count);
  EXPECT_EQ(after_count, 0);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow