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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/default_opset.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/pass/transpose_sinking.h"
#include "test/opexecuter.h"
#include "test/test_utilities.h"

using namespace std;
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
  // checks if Transpose is pushed through opset::Abs, but stopped by
  // ReduceSum
  ngraph::Shape shape_nhwc{16, 28, 28, 1};
  ngraph::Shape shape_nchw{16, 1, 28, 28};

  auto a = make_shared<opset::Parameter>(ngraph::element::i32, shape_nhwc);
  auto ng_order = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose = make_shared<opset::Transpose>(a, ng_order);
  auto absn = make_shared<opset::Abs>(transpose);
  auto absn2 = make_shared<opset::Abs>(absn);

  auto axes = make_shared<opset::Constant>(
      ngraph::element::i64, ngraph::Shape{4}, vector<int64_t>{0, 1, 2, 3});
  auto sum = make_shared<opset::ReduceSum>(transpose, axes, true);

  auto func = make_shared<ngraph::Function>(ngraph::OutputVector{absn2, sum},
                                            ngraph::ParameterVector{a});
  size_t before_count = count_ops_of_type<opset::Transpose>(func);

  ngraph::pass::Manager pass_manager;
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  ASSERT_EQ(before_count, 1);
  size_t after_count = count_ops_of_type<opset::Transpose>(func);
  ASSERT_EQ(after_count, 2);
  ASSERT_EQ(func->get_results().at(1)->input_value(0), sum);
  auto new_transpose = ngraph::as_type_ptr<opset::Transpose>(
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
  ngraph::Shape input_shape{1, 3, 3, 1};  // NHWC (N=1, H=3, W=3, C=1)

  auto input_type = ngraph::element::f32;
  auto output_type = ngraph::element::f32;

  auto X = make_shared<opset::Parameter>(input_type, input_shape);  // NHWC

  auto ng_order1 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose1 =
      make_shared<opset::Transpose>(X, ng_order1);  // NCHW (1,1,3,3)

  auto avgpool = make_shared<opset::AvgPool>(
      transpose1, ngraph::Strides{1, 1}, ngraph::Shape{0, 0},
      ngraph::Shape{0, 0}, ngraph::Shape{1, 1}, true,
      ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID);

  auto ng_order2 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto transpose2 =
      make_shared<opset::Transpose>(avgpool, ng_order2);  // NHWC (1,3,3,1)

  auto const1 = opset::Constant::create(input_type, ngraph::Shape{1, 3, 3, 1},
                                        {3});  // NHWC (1,3,3,1)
  auto add1 = make_shared<opset::Add>(transpose2, const1);
  auto func = make_shared<ngraph::Function>(add1, ngraph::ParameterVector{X});

  ngraph::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<opset::Transpose>(func);
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  size_t after_count = count_ops_of_type<opset::Transpose>(func);
  ASSERT_LE(before_count, after_count);
  ASSERT_EQ(3, after_count);
  auto new_transpose = ngraph::as_type_ptr<opset::Transpose>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), (ngraph::Shape{1, 3, 3, 1}));
}

TEST(TransposeSinking, PoolAdd2) {
  ngraph::Shape input_shape{1, 3, 3, 1};  // NHWC (N=1, H=3, W=3, C=1)

  auto input_type = ngraph::element::f32;
  auto output_type = ngraph::element::f32;

  auto X = make_shared<opset::Parameter>(input_type, input_shape);  // NHWC

  auto ng_order1 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose1 =
      make_shared<opset::Transpose>(X, ng_order1);  // NCHW (1,1,3,3)

  auto avgpool = make_shared<opset::AvgPool>(
      transpose1, ngraph::Strides{1, 1}, ngraph::Shape{0, 0},
      ngraph::Shape{0, 0}, ngraph::Shape{1, 1}, true,
      ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID);

  auto ng_order2 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto transpose2 =
      make_shared<opset::Transpose>(avgpool, ng_order2);  // NHWC (1,3,3,1)
  auto maxpool = make_shared<opset::MaxPool>(
      transpose1, ngraph::Strides{1, 1}, ngraph::Shape{0, 0},
      ngraph::Shape{0, 0}, ngraph::Shape{1, 1}, ngraph::op::RoundingType::FLOOR,
      ngraph::op::PadType::VALID);

  auto ng_order3 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto transpose3 = make_shared<opset::Transpose>(maxpool, ng_order3);

  auto const1 = opset::Constant::create(input_type, ngraph::Shape{1, 3, 3, 1},
                                        {3});  // NHWC (1,3,3,1)
  auto add1 = make_shared<opset::Add>(transpose3, const1);
  auto add2 = make_shared<opset::Add>(add1, transpose2);
  auto func = make_shared<ngraph::Function>(add2, ngraph::ParameterVector{X});

  ngraph::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<opset::Transpose>(func);  // 3
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  size_t after_count = count_ops_of_type<opset::Transpose>(func);  // 4
  ASSERT_LE(before_count, after_count);
  ASSERT_EQ(4, after_count);
  auto new_transpose = ngraph::as_type_ptr<opset::Transpose>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), (ngraph::Shape{1, 3, 3, 1}));
}

// Different rank constant input to Add1. After TransposeSinking the const
// would need a Reshape to have the same order as the other input to
// Add1.
TEST(TransposeSinking, PoolAdd3) {
  ngraph::Shape input_shape{1, 3, 3, 1};  // NHWC (N=1, H=3, W=3, C=1)

  auto input_type = ngraph::element::f32;
  auto output_type = ngraph::element::f32;

  auto X = make_shared<opset::Parameter>(input_type, input_shape);  // NHWC

  auto ng_order1 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose1 =
      make_shared<opset::Transpose>(X, ng_order1);  // NCHW (1,1,3,3)

  auto avgpool = make_shared<opset::AvgPool>(
      transpose1, ngraph::Strides{1, 1}, ngraph::Shape{0, 0},
      ngraph::Shape{0, 0}, ngraph::Shape{1, 1}, true,
      ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID);

  auto ng_order2 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto transpose2 =
      make_shared<opset::Transpose>(avgpool, ng_order2);  // NHWC (1,3,3,1)

  auto const1 = opset::Constant::create(input_type, ngraph::Shape{1},
                                        {1});  // NHWC (1,3,3,1)
  auto add1 = make_shared<opset::Add>(transpose2, const1);
  auto func = make_shared<ngraph::Function>(add1, ngraph::ParameterVector{X});

  ngraph::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<opset::Transpose>(func);
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  size_t after_count = count_ops_of_type<opset::Transpose>(func);
  ASSERT_LE(after_count, before_count);
  auto new_transpose = ngraph::as_type_ptr<opset::Transpose>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), (ngraph::Shape{1, 3, 3, 1}));
}

TEST(TransposeSinking, Concat) {
  // checks if Transpose is pushed through opset::Concat
  ngraph::Shape shape_nhwc{16, 28, 28, 1};
  auto a = make_shared<opset::Parameter>(ngraph::element::i32, shape_nhwc);
  auto b = make_shared<opset::Parameter>(ngraph::element::i32, shape_nhwc);
  auto to_nchw = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto a_transpose = make_shared<opset::Transpose>(a, to_nchw);
  auto b_transpose = make_shared<opset::Transpose>(b, to_nchw);
  auto concat = make_shared<opset::Concat>(
      ngraph::OutputVector{a_transpose, b_transpose}, 0);
  auto to_nhwc = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto c = make_shared<opset::Transpose>(concat, to_nhwc);
  auto func = make_shared<ngraph::Function>(ngraph::OutputVector{c},
                                            ngraph::ParameterVector{a, b});

  ngraph::pass::Manager pass_manager;
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  size_t transpose_count = count_ops_of_type<opset::Transpose>(func);
  ASSERT_EQ(0, transpose_count);
  auto result = func->get_results().at(0)->input_value(0).get_node_shared_ptr();
  ngraph::Shape expected_shape{32, 28, 28, 1};
  ASSERT_EQ(result->get_output_shape(0), expected_shape);
}

TEST(TransposeSinking, Concat_DummyShape) {
  // checks if Transpose is pushed through opset::Concat
  ngraph::Shape shape1{4, 3, 3, 1};
  ngraph::Shape shape2{4, 3, 3, 2};
  ngraph::Shape shape3{4, 3, 3, 3};
  ngraph::Shape shape4{4, 3, 3, 4};
  auto to_nchw = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto to_nhwc = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});

  auto a1 = make_shared<opset::Parameter>(ngraph::element::i32, shape1);
  auto a2 = make_shared<opset::Parameter>(ngraph::element::i32, shape2);
  auto a3 = make_shared<opset::Parameter>(ngraph::element::i32, shape3);
  auto a4 = make_shared<opset::Parameter>(ngraph::element::i32, shape4);
  auto a1_transpose = make_shared<opset::Transpose>(a1, to_nchw);
  auto a2_transpose = make_shared<opset::Transpose>(a2, to_nchw);
  auto a3_transpose = make_shared<opset::Transpose>(a3, to_nchw);
  auto a4_transpose = make_shared<opset::Transpose>(a4, to_nchw);
  auto concat =
      make_shared<opset::Concat>(ngraph::NodeVector{a1_transpose, a2_transpose,
                                                    a3_transpose, a4_transpose},
                                 1);
  auto out = make_shared<opset::Transpose>(concat, to_nchw);
  auto func = make_shared<ngraph::Function>(
      ngraph::OutputVector{out}, ngraph::ParameterVector{a1, a2, a3, a4});

  ngraph::pass::Manager pass_manager;
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  size_t transpose_count = count_ops_of_type<opset::Transpose>(func);  // 1
  ASSERT_EQ(1, transpose_count);
  auto result = func->get_results().at(0)->input_value(0).get_node_shared_ptr();
  ngraph::Shape expected_shape{4, 3, 10, 3};
  ASSERT_EQ(result->get_output_shape(0), expected_shape);
}

// The Transpose should sink through Pad op but stopped by ReduceSum
TEST(TransposeSinking, Pad) {
  ngraph::Shape shape_nhwc{100, 8, 8, 1};

  auto a = make_shared<opset::Parameter>(ngraph::element::f32, shape_nhwc);
  auto pad_value = opset::Constant::create<float>(
      ngraph::element::f32, ngraph::Shape{}, std::vector<float>{0.0f});

  ngraph::CoordinateDiff pad_end{0, 0, 0, 0};
  ngraph::CoordinateDiff pad_begin{0, 1, 1, 0};

  auto a_to_nchw = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto a_transpose = make_shared<opset::Transpose>(a, a_to_nchw);

  auto maxpool = make_shared<opset::MaxPool>(
      a_transpose, ngraph::Strides{2, 2}, ngraph::Shape{0, 0},
      ngraph::Shape{0, 0}, ngraph::Shape{1, 1}, ngraph::op::RoundingType::FLOOR,
      ngraph::op::PadType::VALID);

  auto m_to_nhwc = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto m_transpose = make_shared<opset::Transpose>(maxpool, m_to_nhwc);

  shared_ptr<opset::Constant> pads_begin_node, pads_end_node;
  pads_begin_node = make_shared<opset::Constant>(
      ngraph::element::i64, ngraph::Shape{pad_begin.size()}, pad_begin);
  pads_end_node = make_shared<opset::Constant>(
      ngraph::element::i64, ngraph::Shape{pad_end.size()}, pad_end);
  auto pad =
      make_shared<opset::Pad>(m_transpose, pads_begin_node, pads_end_node,
                              pad_value, ngraph::op::PadMode::CONSTANT);

  auto axes = make_shared<opset::Constant>(
      ngraph::element::i64, ngraph::Shape{4}, vector<int64_t>{0, 1, 2, 3});
  auto sum = make_shared<opset::ReduceSum>(pad, axes, true);

  auto func = make_shared<ngraph::Function>(ngraph::OutputVector{sum},
                                            ngraph::ParameterVector{a});

  ngraph::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<opset::Transpose>(func);  // 2
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  size_t after_count = count_ops_of_type<opset::Transpose>(func);  // 2
  ASSERT_EQ(after_count, before_count);
  auto result = func->get_results().at(0)->input_value(0).get_node_shared_ptr();
  ngraph::Shape expected_shape{1, 1, 1, 1};
  ASSERT_EQ(result->get_output_shape(0), expected_shape);
  auto out = ngraph::as_type_ptr<opset::ReduceSum>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(out);
}

TEST(TransposeSinking, SimpleUnary) {
  ngraph::Shape shape_nhwc{16, 28, 28, 1};
  ngraph::Shape shape_nchw{16, 1, 28, 28};
  auto a = make_shared<opset::Parameter>(ngraph::element::i32, shape_nhwc);

  auto ng_order = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose = make_shared<opset::Transpose>(a, ng_order);

  auto a_t = make_shared<opset::Transpose>(a, ng_order);
  auto absn = make_shared<opset::Abs>(a_t);
  auto absn2 = make_shared<opset::Abs>(absn);

  auto tf_order = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto absn2_t = make_shared<opset::Transpose>(absn2, tf_order);

  auto func = make_shared<ngraph::Function>(ngraph::OutputVector{absn2_t},
                                            ngraph::ParameterVector{a});
  size_t before_count = count_ops_of_type<opset::Transpose>(func);  // 2

  ngraph::pass::Manager pass_manager;
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  size_t after_count = count_ops_of_type<opset::Transpose>(func);  // 0
  ASSERT_EQ(func->get_results().at(0)->input_value(0), absn2);
  EXPECT_NE(before_count, after_count);
  EXPECT_EQ(after_count, 0);
}

//            X (NCHW)
//            |
//         Transpose1
//            |
//         Split (NHWC)
//           /  \
//          /    \
//   Transpose2 Transpose3 (NCHW)
//       |        |
// Const |        |   Const (NCHW)
//  \    |        |   /
//   \   |        |  /
//    \  |        | /
//     Add        Add (NCHW)
//        \       /
//         \     /
//          \   /
//           Add (NCHW)
//            |
//          Result (NCHW)
TEST(TransposeSinking, MultiOutput) {
  ngraph::Shape shape_nhwc{1, 4, 4, 1};
  ngraph::Shape shape_nchw{1, 1, 4, 6};

  auto input_type = ngraph::element::f32;
  auto output_type = ngraph::element::f32;

  auto X = make_shared<opset::Parameter>(input_type, shape_nchw);  // NCHW

  auto ng_order1 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto transpose1 =
      make_shared<opset::Transpose>(X, ng_order1);  // NHWC (1, 4, 6, 1)
  auto ng_split_dim = std::make_shared<opset::Constant>(ngraph::element::u64,
                                                        ngraph::Shape{}, 2);

  auto split =
      make_shared<opset::Split>(transpose1, ng_split_dim, 2);  // (1, 4, 3, 1)

  auto ng_order2 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose2 =
      make_shared<opset::Transpose>(split, ng_order2);  // (1, 1, 4, 3) NCHW

  auto ng_order3 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose3 =
      make_shared<opset::Transpose>(split, ng_order3);  // (1, 1, 4, 3) NCHW

  auto const1 = opset::Constant::create(input_type, ngraph::Shape{1, 1, 4, 3},
                                        {3});  // NCHW
  auto add1 = make_shared<opset::Add>(transpose2, const1);
  auto const2 = opset::Constant::create(input_type, ngraph::Shape{1, 1, 4, 3},
                                        {3});  // NCHW
  auto add2 = make_shared<opset::Add>(transpose3, const2);
  auto add3 = make_shared<opset::Add>(add1, add2);
  auto func = make_shared<ngraph::Function>(add3, ngraph::ParameterVector{X});

  ngraph::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<opset::Transpose>(func);  // 3
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  size_t after_count = count_ops_of_type<opset::Transpose>(func);  // 4
  ASSERT_LE(before_count, after_count);
  ASSERT_EQ(4, after_count);
  auto new_transpose = ngraph::as_type_ptr<opset::Transpose>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), (ngraph::Shape{1, 1, 4, 3}));
}

//            X (NHWC)
//            |
//        Transpose (NCHW)
//            |
//         AvgPool0
//            |
//        Transpose0 (NHWC)
//            |
//          Split (NHWC)
//           /  \
//          /    \
//   Transpose1 Transpose2 (NCHW)
//       |         |
//     AvgPool1  AvgPool2
//       |         |
//   Transpose3 Transpose4 (NHWC)
//        \       /
//         \     /
//          \   /
//          Concat (NHWC)
// Const      /
//   \       /
//    \     /
//     \   /
//      \ /
//      Add (NHWC)
//       |
//     Result
TEST(TransposeSinking, AlexnetPattern) {
  ngraph::Shape shape_nhwc{1, 55, 55, 96};
  ngraph::Shape shape_nchw{1, 96, 55, 55};

  auto input_type = ngraph::element::f32;
  auto output_type = ngraph::element::f32;

  // X
  auto X = make_shared<opset::Parameter>(input_type, shape_nhwc);  // NHWC

  // T -> AvgPool0 -> T0
  auto ng_order = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose = make_shared<opset::Transpose>(X, ng_order);  // NCHW
  auto avgpool0 = make_shared<opset::AvgPool>(
      transpose, ngraph::Strides{1, 1}, ngraph::Shape{0, 0},
      ngraph::Shape{0, 0}, ngraph::Shape{1, 1}, true,
      ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID);
  auto ng_order0 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto transpose0 = make_shared<opset::Transpose>(avgpool0, ng_order0);  // NHWC

  // Split
  auto ng_split_dim = std::make_shared<opset::Constant>(ngraph::element::u64,
                                                        ngraph::Shape{}, 3);
  auto split = make_shared<opset::Split>(transpose0, ng_split_dim, 2);  // NHWC

  // T1 -> AvgPool1 -> T2
  auto ng_order1 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose1 = make_shared<opset::Transpose>(split, ng_order1);  // NCHW
  auto avgpool1 = make_shared<opset::AvgPool>(
      transpose1, ngraph::Strides{1, 1}, ngraph::Shape{0, 0},
      ngraph::Shape{0, 0}, ngraph::Shape{1, 1}, true,
      ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID);
  auto ng_order2 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto transpose2 = make_shared<opset::Transpose>(avgpool1, ng_order2);  // NHWC

  // T3 -> AvgPool2 -> T4
  auto ng_order3 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
  auto transpose3 = make_shared<opset::Transpose>(split, ng_order1);  // NCHW
  auto avgpool2 = make_shared<opset::AvgPool>(
      transpose3, ngraph::Strides{1, 1}, ngraph::Shape{0, 0},
      ngraph::Shape{0, 0}, ngraph::Shape{1, 1}, true,
      ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID);
  auto ng_order4 = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
  auto transpose4 = make_shared<opset::Transpose>(avgpool2, ng_order4);  // NHWC

  // Concat
  auto concat = make_shared<opset::Concat>(
      ngraph::OutputVector{transpose2, transpose4}, 3);  // NHWC

  // Add
  auto const1 =
      opset::Constant::create(input_type, ngraph::Shape{96}, {1});  // NCHW
  auto add1 = make_shared<opset::Add>(concat, const1);

  auto func = make_shared<ngraph::Function>(add1, ngraph::ParameterVector{X});

  ngraph::pass::Manager pass_manager;
  size_t before_count = count_ops_of_type<opset::Transpose>(func);
  pass_manager.register_pass<pass::TransposeSinking>();
  pass_manager.run_passes(func);

  size_t after_count = count_ops_of_type<opset::Transpose>(func);
  ASSERT_LE(after_count, before_count);
  ASSERT_EQ(5, after_count);
  auto new_transpose = ngraph::as_type_ptr<opset::Transpose>(
      func->get_results().at(0)->input_value(0).get_node_shared_ptr());
  ASSERT_TRUE(new_transpose);
  ASSERT_EQ(new_transpose->get_output_shape(0), (ngraph::Shape{1, 55, 55, 96}));
}

// End to end test for TransposeSinking with multi-output node
TEST(TransposeSinking, MultiOutputConV) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 1, 1, 2});
  // num_split : The number of ways to split. Must evenly divide
  // value.shape[split_dim]
  int64_t num_splits = 2;

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  Tensor axis(DT_INT32, TensorShape({}));
  AssignInputValues<int>(axis, 3);
  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    std::vector<float> input_vector = {0.1f, 1.7f};
    AssignInputValues(input_data, input_vector);

    auto R = ops::Split(root, axis, input_data, num_splits);

    vector<int64> filter_size_HWIO = {1, 1, 1, 2};

    std::vector<int> stride = {1, 1, 1, 1};
    // Dilation rates > 1 not supported by TF on CPU
    ops::Conv2D::Attrs op_attr_nhwc;
    op_attr_nhwc = op_attr_nhwc.DataFormat("NHWC");
    op_attr_nhwc = op_attr_nhwc.Dilations({1, 1, 1, 1});

    string padding_type = "SAME";

    Tensor filter1(DT_FLOAT, TensorShape(filter_size_HWIO));
    std::vector<float> filter1_vector = {1.1f, 4.4f};
    AssignInputValues<float>(filter1, filter1_vector);

    auto C1 =
        ops::Conv2D(root, R[0], filter1, stride, padding_type, op_attr_nhwc);

    Tensor filter2(DT_FLOAT, TensorShape(filter_size_HWIO));
    std::vector<float> filter2_vector = {1.9f, 5.4f};
    AssignInputValues<float>(filter2, filter2_vector);

    auto C2 =
        ops::Conv2D(root, R[1], filter2, stride, padding_type, op_attr_nhwc);

    auto A = ops::Add(root, C1, C2);

    std::vector<Output> sess_run_fetchoutputs = {A};
    OpExecuter opexecuter(root, "Split", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

TEST(TransposeSinking, MultiUser) {
  Scope root = Scope::NewRootScope();
  Tensor input(DT_FLOAT, TensorShape({4, 2, 2, 1}));
  AssignInputValuesRandom(input);

  auto input_transpose = ops::Transpose(root, input, {0, 3, 1, 2});
  auto sqrt = ops::Sqrt(root, input_transpose);
  auto sqrt_transpose = ops::Transpose(root, sqrt, {0, 2, 3, 1});
  auto add = ops::Add(root, sqrt, sqrt_transpose);

  OpExecuter opexecuter(root, "MultiUser", {add});
  opexecuter.RunTest();
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow