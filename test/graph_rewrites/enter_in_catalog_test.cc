/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_catalog.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_enter_in_catalog.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_replace_op_utilities.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_replace_variable_modifiers.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_cluster_manager.h"
#include "ngraph_bridge/ngraph_deassign_clusters.h"
#include "ngraph_bridge/ngraph_encapsulate_clusters.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_rewrite_for_tracking.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Graph with Assign ops which should have the attribute
// _ngraph_remove added and set to true.
TEST(CatalogTest, SmallGraph1) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root.WithOpName("Var_Assign"), var, init_value);

  auto c = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});

  auto add = ops::Add(root.WithOpName("Add"), var, c);

  auto assign = ops::Assign(root.WithOpName("Assign"), var, add);

  std::set<string> skip_these_nodes = {};

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // Execute all the passes one by one
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));
  ASSERT_OK(ReplaceModifiers(&graph, 0));
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));
  ASSERT_OK(AssignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, config_map, {0, {}}));
  ASSERT_OK(EnterInCatalog(&graph, 0));

  bool remove = false;
  for (auto node : graph.op_nodes()) {
    auto node_name = node->name();
    remove = false;
    if (node_name == "Assign") {
      ASSERT_OK(GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
      ASSERT_TRUE(remove);
    } else if (node_name == "Var_Assign") {
      ASSERT_OK(GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
      ASSERT_TRUE(remove);
    }
  }
}

// Graph with Assign ops, one of which should not
// have the attribute _ngraph_remove added
TEST(CatalogTest, SmallGraph2) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root.WithOpName("Var_Assign"), var, init_value);

  auto acos = ops::Acos(root.WithOpName("Acos"), var);

  auto assign = ops::Assign(root.WithOpName("Assign"), var, acos);

  std::set<string> skip_these_nodes = {};

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // Execute all the passes one by one
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));
  ASSERT_OK(ReplaceModifiers(&graph, 0));
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));
  ASSERT_OK(AssignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, config_map, {0, {}}));
  ASSERT_OK(EnterInCatalog(&graph, 0));

  bool remove = false;
  for (auto node : graph.op_nodes()) {
    auto node_name = node->name();
    remove = false;
    if (node_name == "Assign") {
      ASSERT_NOT_OK(GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
      ASSERT_FALSE(remove);
    } else if (node_name == "Var_Assign") {
      ASSERT_OK(GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
      ASSERT_TRUE(remove);
    }
  }
}

//  Const   Var_A      Const     Var_B
//   \      /|  \       /         | \
//    \    / |   \     /          |  \
//  Assign_A |    \   /           |  Assign_B
//           |     Add            |
//           |      /\            |
//           |     /  \           |
//           |    /    \          |
//          Assign_1   Acos       |
//                       \        |
//                        \       |
//                         \      |
//                         Assign_2
// Assign_A, Assign_B and Assign_1 should have the attribute
// _ngraph_remove added and set to true
// whereas Assign_2 should not have the attribute added since
// it is being fed by Acos which is not supported by nGraph.
TEST(CatalogTest, SmallGraph3) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var_a = ops::Variable(root.WithOpName("VarA"), varShape, DT_FLOAT);
  auto init_value_a = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign_a =
      ops::Assign(root.WithOpName("VarA_Assign"), var_a, init_value_a);

  auto var_b = ops::Variable(root.WithOpName("VarB"), varShape, DT_FLOAT);
  auto init_value_b = ops::Const(root, {{2.f, 2.f}, {2.f, 2.f}});
  auto var_assign_b =
      ops::Assign(root.WithOpName("VarB_Assign"), var_b, init_value_b);

  auto c = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});

  auto add = ops::Add(root.WithOpName("Add"), var_a, c);

  auto assign_1 = ops::Assign(root.WithOpName("Assign_1"), var_a, add);

  auto acos = ops::Acos(root.WithOpName("Acos"), add);

  auto assign_2 = ops::Assign(root.WithOpName("Assign_2"), var_b, acos);

  std::set<string> skip_these_nodes = {};

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // Execute all the passes one by one
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));
  ASSERT_OK(ReplaceModifiers(&graph, 0));
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));
  ASSERT_OK(AssignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, config_map, {0, {}}));
  ASSERT_OK(EnterInCatalog(&graph, 0));

  // check if the _ngraph_remove attribute is added/not-added as expected
  // check if correct information is added to the map
  bool remove;
  string key;
  for (auto node : graph.op_nodes()) {
    if (node->type_string() == "NGraphAssign") {
      key = "";
      Node* input_1;
      ASSERT_OK(node->input_node(1, &input_1));
      if (input_1->type_string() == "NGraphEncapsulate") {
        const Edge* edge;
        ASSERT_OK(node->input_edge(1, &edge));
        int output_index = edge->src_output();
        int graph_id;
        ASSERT_OK(GetNodeAttr(node->attrs(), "ngraph_graph_id", &graph_id));
        key = NGraphCatalog::CreateNodeKey(graph_id, input_1->name(),
                                           output_index);
      }
      auto node_name = node->name();
      remove = false;
      if (node_name == "VarA_Assign" || node_name == "VarB_Assign" ||
          node_name == "Assign_1") {
        ASSERT_OK(GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
        ASSERT_TRUE(remove);
        ASSERT_TRUE(NGraphCatalog::ExistsInEncapOutputInfoMap(key));
      } else if (node_name == "Assign_2") {
        ASSERT_NOT_OK(GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
        ASSERT_FALSE(remove);
        ASSERT_FALSE(NGraphCatalog::ExistsInEncapOutputInfoMap(key));
      }
    }
  }
}

// Test to check if correct information is being added to the
// encap_output_info_map_
TEST(CatalogTest, SmallGraph4) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root.WithOpName("Var_Assign"), var, init_value);

  auto c = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});

  auto add = ops::Add(root.WithOpName("Add"), var, c);

  auto assign = ops::Assign(root.WithOpName("Assign"), var, add);

  std::set<string> skip_these_nodes = {};

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // Execute all the passes one by one
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));
  ASSERT_OK(ReplaceModifiers(&graph, 0));
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));
  ASSERT_OK(AssignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, config_map, {0, {}}));
  ASSERT_OK(EnterInCatalog(&graph, 0));

  string key;
  for (auto node : graph.op_nodes()) {
    key = "";
    if (node->type_string() == "NGraphAssign") {
      Node* input_1;
      ASSERT_OK(node->input_node(1, &input_1));
      if (input_1->type_string() == "NGraphEncapsulate") {
        const Edge* edge;
        ASSERT_OK(node->input_edge(1, &edge));
        int output_index = edge->src_output();
        int graph_id;
        ASSERT_OK(GetNodeAttr(node->attrs(), "ngraph_graph_id", &graph_id));
        key = NGraphCatalog::CreateNodeKey(graph_id, input_1->name(),
                                           output_index);
      }
      auto node_name = node->name();
      if (node_name == "Assign" || node_name == "Var_Assign") {
        ASSERT_TRUE(NGraphCatalog::ExistsInEncapOutputInfoMap(key));
      }
    }
  }
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow