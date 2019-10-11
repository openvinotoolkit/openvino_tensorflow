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
#include "ngraph_bridge/enable_variable_ops/ngraph_remove_ngraphassigns.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_encapsulate_clusters.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "test/test_utilities.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Assigns are removed when the new value is computed by the
// Encapsulate Op.
// These tests check that NGraphAssign is removed and the rest of the graph
// is connected correctly.

// Var       Const
//  \         /
//   \       /
//    Assign
TEST(RemoveNGraphAssigns, Graph1) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto val = ops::Const(root.WithOpName("Val"), {{1.f, 1.f}, {1.f, 1.f}});
  auto assign = ops::Assign(root.WithOpName("VarAssign"), var, val);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // Capture Variables: to convert Var and Assign to NGraphVar and NGraphAssign
  // there is no other way to create these ops
  std::set<string> skip_these_nodes = {};
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));

  // Need encapsulate op for test
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));
  ASSERT_OK(AssignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, config_map, {0, {}}));

  // Get all the nodes in map [utility]
  map<string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  // Attach _ngraph_remove attribue to NGraphAssign (triggering the remove)
  ASSERT_NE(node_map.find("VarAssign"), node_map.end());
  Node* ng_assign = node_map.at("VarAssign");
  ASSERT_EQ(ng_assign->type_string(), "NGraphAssign");
  ng_assign->AddAttr("_ngraph_remove", true);

  // Call RemoveNGraphAssign
  ASSERT_OK(RemoveNGraphAssigns(&graph));

  // Assert NGraphAssign is not present
  for (auto node : graph.op_nodes()) {
    ASSERT_NE(node->type_string(), "NGraphAssign");
  }
}

// Var       Const
//  \         / |
//   \       /  |
//    Assign    |
//      |       |
//      |       |
//     Add <-----
//
TEST(RemoveNGraphAssigns, Graph2) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto val = ops::Const(root.WithOpName("Val"), {{1.f, 1.f}, {1.f, 1.f}});
  auto assign = ops::Assign(root.WithOpName("VarAssign"), var, val);
  auto add = ops::Add(root.WithOpName("Add"), assign, val);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // Capture Variables: to convert Var and Assign to NGraphVar and NGraphAssign
  // there is no other way to create these ops
  std::set<string> skip_these_nodes = {};
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));

  // Need encapsulate op for test
  // Keeping Add out of encap, to get the name easily and test
  char disabled_list[] = "Add";
  config::ngraph_set_disabled_ops(disabled_list);

  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));
  ASSERT_OK(AssignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, config_map, {0, {}}));

  // clean up
  config::ngraph_set_disabled_ops("");

  // Get all the nodes in map [utility]
  map<string, Node*> node_map;
  string encap_op_name;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
    if (node->type_string() == "NGraphEncapsulate") {
      encap_op_name = node->name();
    }
  }

  // Attach _ngraph_remove attribue to NGraphAssign (triggering the remove)
  ASSERT_NE(node_map.find("VarAssign"), node_map.end());
  Node* ng_assign = node_map.at("VarAssign");
  ASSERT_EQ(ng_assign->type_string(), "NGraphAssign");
  ng_assign->AddAttr("_ngraph_remove", true);

  // Call RemoveNGraphAssign
  ASSERT_OK(RemoveNGraphAssigns(&graph));

  // Reiterate the graph
  node_map.clear();
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  // Assert NGraphAssign is not present and other ops are present
  ASSERT_EQ(node_map.find("VarAssign"), node_map.end());
  ASSERT_NE(node_map.find("Var"), node_map.end());
  ASSERT_NE(node_map.find(encap_op_name), node_map.end());
  ASSERT_NE(node_map.find("Add"), node_map.end());

  Node *add_in_0, *add_in_1, *add_in_ctrl, *nd_add = node_map.at("Add");
  // NOTE:node->input_edge(...), node->input_node(...) cannot be used for
  // control edges

  int edge_count = 0;
  for (auto edge : nd_add->in_edges()) {
    if (edge->dst_input() == 0) {
      add_in_0 = edge->src();
      ASSERT_FALSE(IsRefType(nd_add->input_type(0)));
    } else if (edge->dst_input() == 1) {
      add_in_1 = edge->src();
      ASSERT_FALSE(IsRefType(nd_add->input_type(1)));
    } else if (edge->dst_input() == Graph::kControlSlot) {
      add_in_ctrl = edge->src();
    }
    edge_count++;
  }

  // Assert on edges connected to add
  ASSERT_EQ(edge_count, 3);
  ASSERT_EQ(add_in_0, node_map.at("Var"));
  ASSERT_EQ(add_in_1, node_map.at(encap_op_name));
  ASSERT_EQ(add_in_ctrl, node_map.at(encap_op_name));

  // Assert on control edge between Var and Encap
  for (auto edge : add_in_0->out_edges()) {
    if ((edge != nullptr) && (edge->IsControlEdge())) {
      ASSERT_EQ(add_in_1, edge->dst());
    }
  }
}

// Var       Const
//  \         / |
//   \       /  |
//    Assign    |
//      |       |
//      |       |
//     Assign2 <-
// Only Assign is marked for removal
// Mainly done to see if Assign2 gets the edge from Var as ref-type
// after the Assign is removed
TEST(RemoveNGraphAssigns, Graph3) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto val = ops::Const(root.WithOpName("Val"), {{1.f, 1.f}, {1.f, 1.f}});
  auto assign = ops::Assign(root.WithOpName("VarAssign"), var, val);
  auto assign2 = ops::Assign(root.WithOpName("VarAssign2"), assign, val);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // Capture Variables: to convert Var and Assign to NGraphVar and NGraphAssign
  // there is no other way to create these ops
  std::set<string> skip_these_nodes = {};
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));

  // Need encapsulate op for test
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));
  ASSERT_OK(AssignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, config_map, {0, {}}));

  // Get all the nodes in map [utility]
  map<string, Node*> node_map;

  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  // Attach _ngraph_remove attribue to NGraphAssign (triggering the remove)
  ASSERT_NE(node_map.find("VarAssign"), node_map.end());
  Node* ng_assign = node_map.at("VarAssign");
  ASSERT_EQ(ng_assign->type_string(), "NGraphAssign");
  ng_assign->AddAttr("_ngraph_remove", true);

  // Call RemoveNGraphAssign
  ASSERT_OK(RemoveNGraphAssigns(&graph));

  // Reiterate over the graph
  node_map.clear();
  string encap_op_name;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
    if (node->type_string() == "NGraphEncapsulate") {
      encap_op_name = node->name();
    }
  }

  // Assert NGraphAssign is not present and other ops are present
  ASSERT_EQ(node_map.find("VarAssign"), node_map.end());
  ASSERT_NE(node_map.find("Var"), node_map.end());
  ASSERT_NE(node_map.find(encap_op_name), node_map.end());
  ASSERT_NE(node_map.find("VarAssign2"), node_map.end());

  Node *assign_in_0, *assign_in_1, *assign_in_ctrl,
      *ng_assign2 = node_map.at("VarAssign2");
  // NOTE:node->input_edge(...), node->input_node(...) cannot be used for
  // control edges

  int edge_count = 0;
  for (auto edge : ng_assign2->in_edges()) {
    if (edge->dst_input() == 0) {
      assign_in_0 = edge->src();
      ASSERT_TRUE(IsRefType(ng_assign2->input_type(0)));
    } else if (edge->dst_input() == 1) {
      assign_in_1 = edge->src();
      ASSERT_FALSE(IsRefType(ng_assign2->input_type(1)));
    } else if (edge->dst_input() == Graph::kControlSlot) {
      assign_in_ctrl = edge->src();
    }
    edge_count++;
  }

  ASSERT_EQ(edge_count, 3);
  ASSERT_EQ(assign_in_0, node_map.at("Var"));
  ASSERT_EQ(assign_in_1, node_map.at(encap_op_name));
  ASSERT_EQ(assign_in_ctrl, node_map.at(encap_op_name));

  // Assert on control edge between Var and Encap
  for (auto edge : assign_in_0->out_edges()) {
    if ((edge != nullptr) && (edge->IsControlEdge())) {
      ASSERT_EQ(assign_in_1, edge->dst());
    }
  }
}

// Var       Const
//  \         / |
//   \       /  |
//    Assign    |
//      |       |
//      |       |
//     Assign2<--
//      |       |
//      |       |
//     Add<-----
//
// Both Assign, and Assign2 is marked for removal
TEST(RemoveNGraphAssigns, Graph4) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto val = ops::Const(root.WithOpName("Val"), {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root.WithOpName("VarAssign"), var, val);
  auto assign2 = ops::Assign(root.WithOpName("VarAssign2"), var_assign, val);
  auto add = ops::Add(root.WithOpName("Add"), assign2, val);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // Capture Variables: to convert Var and Assign to NGraphVar and NGraphAssign
  // there is no other way to create these ops
  std::set<string> skip_these_nodes = {};
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));

  // Need encapsulate op for test
  // Keeping Add out of encap, to get the name easily and test
  char disabled_list[] = "Add";
  config::ngraph_set_disabled_ops(disabled_list);

  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));
  ASSERT_OK(AssignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, config_map, {0, {}}));

  // clean up
  config::ngraph_set_disabled_ops("");

  // Get all the nodes in map [utility]
  map<string, Node*> node_map;
  string encap_op_name;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
    if (node->type_string() == "NGraphEncapsulate") {
      encap_op_name = node->name();
    }
  }

  // Attach _ngraph_remove attribue to NGraphAssign (triggering the remove)
  ASSERT_NE(node_map.find("VarAssign"), node_map.end());
  Node *ng_assign = node_map.at("VarAssign"),
       *ng_assign2 = node_map.at("VarAssign2");
  ASSERT_EQ(ng_assign->type_string(), "NGraphAssign");
  ASSERT_EQ(ng_assign2->type_string(), "NGraphAssign");
  ng_assign->AddAttr("_ngraph_remove", true);
  ng_assign2->AddAttr("_ngraph_remove", true);

  // Call RemoveNGraphAssign
  ASSERT_OK(RemoveNGraphAssigns(&graph));

  // Reiterate over the graph
  node_map.clear();
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  // Assert NGraphAssign is not present and other ops are present
  ASSERT_EQ(node_map.find("VarAssign"), node_map.end());
  ASSERT_EQ(node_map.find("VarAssign2"), node_map.end());

  ASSERT_NE(node_map.find("Var"), node_map.end());
  ASSERT_NE(node_map.find(encap_op_name), node_map.end());
  ASSERT_NE(node_map.find("Add"), node_map.end());

  Node *add_in_0, *add_in_1, *add_in_ctrl, *nd_add = node_map.at("Add");
  // NOTE:node->input_edge(...), node->input_node(...) cannot be used for
  // control edges

  int edge_count = 0;
  for (auto edge : nd_add->in_edges()) {
    if (edge->dst_input() == 0) {
      add_in_0 = edge->src();
      ASSERT_FALSE(IsRefType(nd_add->input_type(0)));
    } else if (edge->dst_input() == 1) {
      add_in_1 = edge->src();
      ASSERT_FALSE(IsRefType(nd_add->input_type(1)));
    } else if (edge->dst_input() == Graph::kControlSlot) {
      add_in_ctrl = edge->src();
    }
    edge_count++;
  }

  ASSERT_EQ(edge_count, 3);
  ASSERT_EQ(add_in_0, node_map.at("Var"));
  ASSERT_EQ(add_in_1, node_map.at(encap_op_name));
  ASSERT_EQ(add_in_ctrl, node_map.at(encap_op_name));

  // Assert on control edge between Var and Encap
  for (auto edge : add_in_0->out_edges()) {
    if ((edge != nullptr) && (edge->IsControlEdge())) {
      ASSERT_EQ(add_in_1, edge->dst());
    }
  }
}

// Var       Const
//  \         /
//   \       /
//    Assign

// Const is not Encapsulated
// Assign is marked for removal
// RemoveNGraphAssign throws an error
TEST(RemoveNGraphAssigns, Graph5) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto val = ops::Const(root.WithOpName("Val"), {{1.f, 1.f}, {1.f, 1.f}});
  auto assign = ops::Assign(root.WithOpName("VarAssign"), var, val);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // Capture Variables: to convert Var and Assign to NGraphVar and NGraphAssign
  // there is no other way to create these ops
  std::set<string> skip_these_nodes = {};
  ASSERT_OK(CaptureVariables(&graph, skip_these_nodes));

  // Need encapsulate op for test
  // Keeping Const out of encap
  char disabled_list[] = "Const";
  config::ngraph_set_disabled_ops(disabled_list);
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));
  ASSERT_OK(AssignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, config_map, {0, {}}));

  // Get all the nodes in map [utility]
  map<string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  // Clean up
  config::ngraph_set_disabled_ops("");

  // Attach _ngraph_remove attribue to NGraphAssign (triggering the remove)
  ASSERT_NE(node_map.find("VarAssign"), node_map.end());
  Node* ng_assign = node_map.at("VarAssign");
  ASSERT_EQ(ng_assign->type_string(), "NGraphAssign");
  ng_assign->AddAttr("_ngraph_remove", true);

  // Call RemoveNGraphAssign, throws error
  ASSERT_NOT_OK(RemoveNGraphAssigns(&graph));
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow