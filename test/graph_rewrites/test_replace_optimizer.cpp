/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "gtest/gtest.h"
#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_replace_op_utilities.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_replace_variable_modifiers.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// ReplaceModifier
TEST(ReplaceModifierTest, Momentum2) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root.WithOpName("Assign1"), var, init_value);

  auto accum = ops::Variable(root.WithOpName("accum"), varShape, DT_FLOAT);
  auto init_value2 = ops::Const(root, {{3.f, 3.f}, {3.f, 3.f}});
  auto accum_assign =
      ops::Assign(root.WithOpName("Assign2"), accum, init_value2);

  auto grad = ops::Const(root, {{2.f, 2.f}, {2.f, 2.f}});

  auto lr = ops::Const(root, 1.f);
  auto momentum = ops::Const(root, 1.f);

  ops::ApplyMomentum::Attrs op_attr_use_nestrov;

  op_attr_use_nestrov = op_attr_use_nestrov.UseNesterov(true);
  auto applymomentum_t =
      ops::ApplyMomentum(root.WithOpName("Momentum"), var, accum, lr, grad,
                         momentum, op_attr_use_nestrov);

  std::set<string> skip_these_nodes = {};

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));
  map<string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }
  ASSERT_NE(node_map.find("Momentum"), node_map.end());
  ASSERT_EQ(node_map.find("Momentum")->second->type_string(), "ApplyMomentum");
  node_map.clear();
  // Execute all the passes one by one
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));
  // Get all the nodes in map [utility]
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }
  ASSERT_EQ(node_map.find("Momentum")->second->type_string(),
            "NGraphApplyMomentum");
  node_map.clear();
  ASSERT_OK(ReplaceModifiers(&graph, 0));
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  ASSERT_NE(node_map.find("Momentum_Mul"), node_map.end());
  ASSERT_NE(node_map.find("Momentum_Add"), node_map.end());
  ASSERT_NE(node_map.find("Momentum_AccumAssign"), node_map.end());
  ASSERT_NE(node_map.find("Momentum_Mul1"), node_map.end());
  ASSERT_NE(node_map.find("Momentum_Mul2"), node_map.end());
  ASSERT_NE(node_map.find("Momentum_Mul3"), node_map.end());
  ASSERT_NE(node_map.find("Momentum_Add_1"), node_map.end());
  ASSERT_NE(node_map.find("Momentum_Sub"), node_map.end());
  ASSERT_NE(node_map.find("Momentum_NGraphAssign"), node_map.end());
  for (auto edge : graph.edges()) {
    if (edge->src()->name() == "Momentum_AccumAssign") {
      ASSERT_EQ(edge->dst()->name(), "Momentum_Mul1");
    } else if (edge->src()->name() == "Momentum_NGraphAssign") {
      ASSERT_EQ(edge->dst()->name(), "_SINK");
    } else if (edge->src()->name() == "Momentum_Mul1") {
      ASSERT_EQ(edge->dst()->name(), "Momentum_Mul3");
    }
  }
}
}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
