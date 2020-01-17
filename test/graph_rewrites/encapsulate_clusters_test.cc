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

#include "tensorflow/core/graph/node_builder.h"

#include "ngraph_bridge/ngraph_cluster_manager.h"
#include "ngraph_bridge/ngraph_encapsulate_clusters.h"
#include "ngraph_bridge/version.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Test that calls the functions of encapsulator in the wrong order
// Non-OK statuses are expected
TEST(EncapsulateClusters, EncapsulatorFail) {
  Encapsulator enc{nullptr};
  std::unordered_map<std::string, std::string> device_config;
  ASSERT_NOT_OK(enc.RewritePass(nullptr, 0, device_config));
  set<int> result;
  ASSERT_NOT_OK(enc.GetNewClusterIDs(result));
}

//                abs
//                 ^
//                 |
// const(0) ---> add(1) <---const(1)
TEST(EncapsulateClusters, EncapsulatorPass) {
  auto num_graphs_in_cluster_manager = []() {
    int num = 0;
    while (true) {
      if (NGraphClusterManager::GetClusterGraph(num) == nullptr) {
        break;
      } else {
        num++;
      }
    }
    return num;
  };
  NGraphClusterManager::EvictAllClusters();
  ASSERT_EQ(num_graphs_in_cluster_manager(), 0);
  Graph g(OpRegistry::Global());

  Tensor t_input_0(DT_FLOAT, TensorShape{2, 3});
  Tensor t_input_1(DT_INT32, TensorShape{2});
  t_input_1.flat<int32>().data()[0] = 3;
  t_input_1.flat<int32>().data()[1] = 2;

  int cluster_idx_0 = NGraphClusterManager::NewCluster();
  ;

  Node* node1;
  ASSERT_OK(NodeBuilder("node1", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_0)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx_0)
                .Attr("_ngraph_backend", "CPU")
                .Finalize(&g, &node1));

  int cluster_idx_1 = NGraphClusterManager::NewCluster();
  ASSERT_EQ(num_graphs_in_cluster_manager(), 2);

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_1)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx_1)
                .Attr("_ngraph_backend", "CPU")
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx_1)
                .Attr("_ngraph_backend", "CPU")
                .Finalize(&g, &node3));

  Node* node4;
  ASSERT_OK(NodeBuilder("node4", "Abs")
                .Input(node3, 0)
                .Attr("T", DT_FLOAT)
                .Finalize(&g, &node4));

  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node4, Graph::kControlSlot, sink, Graph::kControlSlot);

  ASSERT_EQ(g.num_edges(), 7);
  ASSERT_EQ(g.num_op_nodes(), 4);
  ASSERT_EQ(g.num_nodes(), 6);

  Encapsulator enc(&g);

  // Initially ClusterManager is empty
  for (int i = 0; i < 2; i++) {
    ASSERT_EQ(NGraphClusterManager::GetClusterGraph(i)->node_size(), 0);
  }
  ASSERT_OK(enc.AnalysisPass());
  // After AnalysisPass ClusterManager is populated
  // const and retval
  ASSERT_EQ(NGraphClusterManager::GetClusterGraph(0)->node_size(), 2);
  // arg, const, add and retval
  ASSERT_EQ(NGraphClusterManager::GetClusterGraph(1)->node_size(), 4);
  // But the graph structure stays same. No rewriting yet
  ASSERT_EQ(g.num_edges(), 7);
  ASSERT_EQ(g.num_op_nodes(), 4);
  ASSERT_EQ(g.num_nodes(), 6);

  set<int> newly_created_cluster_ids;
  ASSERT_OK(enc.GetNewClusterIDs(newly_created_cluster_ids));
  set<int> expected{0, 1};
  ASSERT_EQ(newly_created_cluster_ids, expected);

  auto subgraph_0 = NGraphClusterManager::GetClusterGraph(0);
  auto subgraph_1 = NGraphClusterManager::GetClusterGraph(1);
  auto subgraph_2 = NGraphClusterManager::GetClusterGraph(2);
  // Assert that there are only 2 subgraphs
  ASSERT_EQ(subgraph_2, nullptr);

  int num_encapsulates = 0;
  int num_tf_nodes = 0;

  // count number of nodes and encapsulates
  auto node_counter = [](Graph* g) {
    int num_encapsulates = 0, num_tf_nodes = 0;
    for (auto itr : g->nodes()) {
      auto node_type = itr->type_string();
      num_encapsulates += (node_type == "NGraphEncapsulate" ? 1 : 0);
      num_tf_nodes +=
          ((node_type == "Add" || node_type == "Const" || node_type == "Abs")
               ? 1
               : 0);
    }
    return make_pair(num_encapsulates, num_tf_nodes);
  };

  std::tie(num_encapsulates, num_tf_nodes) = node_counter(&g);

  // All the Add/Const/Abs nodes are left in the graph, since it is an analysis
  // pass
  ASSERT_EQ(num_tf_nodes, 4);

  // Number of encapsulates == number of functions == 0
  ASSERT_EQ(num_encapsulates, 0);

  // In analysis pass cluster manager should be populated with the subgraphs
  // Now analyse subgraph_0 and subgraph_1, which we got from ClusterManager
  ASSERT_EQ(subgraph_0->node_size(), 2);
  ASSERT_EQ(subgraph_1->node_size(), 4);

  // helper function to get nodes and their types from a graphdef
  auto get_node_name_and_types =
      [](GraphDef* subgraph) -> set<pair<string, string>> {
    set<pair<string, string>> node_info;
    for (int i = 0; i < subgraph->node_size(); i++) {
      node_info.insert({subgraph->node(i).name(), subgraph->node(i).op()});
    }
    return node_info;
  };

  ASSERT_EQ(get_node_name_and_types(subgraph_0),
            (set<pair<string, string>>{{"ngraph_output_0", "_Retval"},
                                       {"node1", "Const"}}));
  ASSERT_EQ(get_node_name_and_types(subgraph_1),
            (set<pair<string, string>>{{"ngraph_input_0", "_Arg"},
                                       {"ngraph_output_0", "_Retval"},
                                       {"node2", "Const"},
                                       {"node3", "Add"}}));

  // Now perform the actual rewrite
  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  ASSERT_OK(enc.RewritePass(fdeflib_new, 0, config_map));

  std::tie(num_encapsulates, num_tf_nodes) = node_counter(&g);
  ASSERT_EQ(num_tf_nodes, 1);  // Only Abs is left
  ASSERT_EQ(num_encapsulates, 2);
  // Number of encapsulates == number of functions
  ASSERT_EQ(num_encapsulates, fdeflib_new->function_size());

  // After RewritePass, the number of clusters is still 2 and it contains
  // populated graphdefs
  ASSERT_EQ(num_graphs_in_cluster_manager(), 2);
  ASSERT_EQ(NGraphClusterManager::GetClusterGraph(0)->node_size(), 2);
  ASSERT_EQ(NGraphClusterManager::GetClusterGraph(1)->node_size(), 4);

  // The graph structure should have changed after RewritePass
  ASSERT_EQ(g.num_edges(), 6);
  ASSERT_EQ(g.num_op_nodes(), 3);
  ASSERT_EQ(g.num_nodes(), 5);

  free(fdeflib_new);
}

// const(0) ---> add(0) <---const(0)
TEST(EncapsulateClusters, PopulateLibrary) {
  NGraphClusterManager::EvictAllClusters();
  Graph g(OpRegistry::Global());

  Tensor t_input_0(DT_FLOAT, TensorShape{2, 3});
  Tensor t_input_1(DT_INT32, TensorShape{2});
  t_input_1.flat<int32>().data()[0] = 3;
  t_input_1.flat<int32>().data()[1] = 2;

  int cluster_idx = NGraphClusterManager::NewCluster();

  Node* node1;
  ASSERT_OK(NodeBuilder("node1", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_0)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Attr("_ngraph_backend", "CPU")
                .Finalize(&g, &node1));

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_1)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Attr("_ngraph_backend", "CPU")
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Attr("_ngraph_backend", "CPU")
                .Finalize(&g, &node3));

  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node3, Graph::kControlSlot, sink, Graph::kControlSlot);

  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();

  std::unordered_map<std::string, std::string> config_map;
  config_map["ngraph_device_id"] = "";
  ASSERT_EQ(g.num_edges(), 6);
  ASSERT_EQ(g.num_op_nodes(), 3);
  ASSERT_EQ(g.num_nodes(), 5);
  ASSERT_OK(EncapsulateClusters(&g, 0, fdeflib_new, config_map, {0, {}}));

  ASSERT_EQ(g.num_edges(), 3);
  ASSERT_EQ(g.num_op_nodes(), 1);
  ASSERT_EQ(g.num_nodes(), 3);

  int num_encapsulates = 0;
  int num_tf_nodes = 0;
  for (auto itr : g.nodes()) {
    auto node_type = itr->type_string();
    num_encapsulates += (node_type == "NGraphEncapsulate" ? 1 : 0);
    num_tf_nodes += ((node_type == "Add" || node_type == "Const") ? 1 : 0);
  }

  // Number of encapsulates == number of functions
  ASSERT_EQ(num_encapsulates, fdeflib_new->function_size());

  // No Add or Const nodes left in the graph
  ASSERT_EQ(num_tf_nodes, 0);

  // In this case, only 1 function has been added in the library
  ASSERT_EQ(fdeflib_new->function_size(), 1);

  // Check the name of the signature of the first (and only) function
  auto first_func = fdeflib_new->function(0);
  ASSERT_EQ(first_func.signature().name(),
            ("ngraph_cluster_" + to_string(cluster_idx)));

  // The first function in the flib should have 3 nodes
  ASSERT_EQ(first_func.node_def_size(), 3);

  // Ensure that the function is made of 2 op types, Add, Const, Const
  auto present = multiset<string>{string(first_func.node_def(0).op()),
                                  string(first_func.node_def(1).op()),
                                  string(first_func.node_def(2).op())};
  auto expected = multiset<string>{"Const", "Add", "Const"};
  ASSERT_EQ(present, expected);
  free(fdeflib_new);
}

//   Placeholder-->Add(0)--->IdN
//                  ^
//                  |
//              Placeholder or Const(0)
TEST(EncapsulateClusters, AOT0) {
  if (!ngraph_tf_is_grappler_enabled()) return;

  vector<bool> fed_by_placeholder{true, false};
  for (auto using_placeholder : fed_by_placeholder) {
    NGraphClusterManager::EvictAllClusters();
    Graph g(OpRegistry::Global());

    int cluster_idx = NGraphClusterManager::NewCluster();

    Node* node1;
    Node* node2;

    ASSERT_OK(NodeBuilder("node1", "Placeholder")
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(&g, &node1));

    if (using_placeholder) {
      ASSERT_OK(NodeBuilder("node2", "Placeholder")
                    .Attr("dtype", DT_FLOAT)
                    .Finalize(&g, &node2));
    } else {
      Tensor t_shape(DT_INT32, TensorShape{2});
      t_shape.flat<int32>().data()[0] = 2;
      t_shape.flat<int32>().data()[1] = 2;
      ASSERT_OK(NodeBuilder("node1", "Const")
                    .Attr("dtype", DT_FLOAT)
                    .Attr("value", t_shape)
                    .Attr("_ngraph_marked_for_clustering", true)
                    .Attr("_ngraph_cluster", cluster_idx)
                    .Attr("_ngraph_backend", "INTERPRETER")
                    .Finalize(&g, &node2));
    }

    Node* node3;
    ASSERT_OK(NodeBuilder("node3", "Add")
                  .Input(node1, 0)
                  .Input(node2, 0)
                  .Attr("T", DT_FLOAT)
                  .Attr("_ngraph_marked_for_clustering", true)
                  .Attr("_ngraph_cluster", cluster_idx)
                  .Attr("_ngraph_backend", "INTERPRETER")
                  .Finalize(&g, &node3));
    Node* node4;
    std::vector<NodeBuilder::NodeOut> inputs;
    std::vector<DataType> input_types;
    inputs.push_back(NodeBuilder::NodeOut(node3, 0));
    input_types.push_back(node3->output_type(0));
    ASSERT_OK(NodeBuilder("node4", "IdentityN")
                  .Input(inputs)
                  .Attr("T", input_types)
                  .Finalize(&g, &node4));

    Node* source = g.source_node();
    Node* sink = g.sink_node();
    g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
    g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
    g.AddEdge(node4, Graph::kControlSlot, sink, Graph::kControlSlot);

    FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();

    std::vector<std::set<ShapeHintMap>> node_shapes_hints_vect = {
        {}, {{{"node1", {2, 2}}, {"node2", {2, 2}}}}};
    std::vector<bool> did_aot = {true, true};
    // Interesting case when shape hints = {}.
    // The placeholder has shape = [].
    // Which means, given no hints it is indeed possible to compile that, since
    // its fully specified (no -1s)
    int num_cases = node_shapes_hints_vect.size();
    for (int i = 0; i < num_cases; i++) {
      std::set<ShapeHintMap> hint;
      for (auto itr : node_shapes_hints_vect[i]) {
        if (using_placeholder) {
          hint.insert(itr);
        } else {
          ShapeHintMap temp;
          for (auto it : itr) {
            if (it.first != "node2") {
              temp.insert({it.first, it.second});
            }
          }
          hint.insert(temp);
        }
      }
      auto status =
          EncapsulateClusters(&g, 0, fdeflib_new, {{"ngraph_device_id", ""}},
                              make_pair(true, hint));
      if (did_aot[i]) {
        ASSERT_OK(status);
      } else {
        ASSERT_NOT_OK(status);
        continue;
      }

      int num_encapsulates = 0;
      int num_tf_nodes = 0;
      for (auto itr : g.nodes()) {
        auto node_type = itr->type_string();
        num_encapsulates += (node_type == "NGraphEncapsulate" ? 1 : 0);
        num_tf_nodes += ((node_type == "Add" || node_type == "Const") ? 1 : 0);
      }

      ASSERT_EQ(num_encapsulates, fdeflib_new->function_size());
      ASSERT_EQ(num_encapsulates, 1);

      // No Add or Const nodes left in the graph
      ASSERT_EQ(num_tf_nodes, 0);

      auto get_attr_name_for_aot = [&i, &using_placeholder](bool is_exec) {
        string attrname = "_ngraph_aot_";
        attrname += (string(is_exec ? "ngexec" : "ngfunction") + "_");
        string signature_portion = (string(i == 0 ? "" : "2,2,") + ";");
        attrname += signature_portion;
        attrname += (using_placeholder ? signature_portion : "");
        attrname += "/";
        return attrname;
      };

      for (auto itr : g.nodes()) {
        if (itr->type_string() == "NGraphEncapsulate") {
          string aot_info;
          bool found_exec =
              GetNodeAttr(itr->attrs(), get_attr_name_for_aot(true),
                          &aot_info) == tensorflow::Status::OK();
          bool found_function =
              GetNodeAttr(itr->attrs(), get_attr_name_for_aot(false),
                          &aot_info) == tensorflow::Status::OK();
          ASSERT_TRUE(found_exec == did_aot[i]);
          ASSERT_TRUE(found_function == did_aot[i]);
        }
      }
    }

    free(fdeflib_new);
  }
}

//   Placeholder-->Add(0)--->Abs(1)-->IdN
//                  ^
//                  |
//              Placeholder
// 2 Encapsulates connected in serial. Cannot AOT here
TEST(EncapsulateClusters, AOT1) {
  if (!ngraph_tf_is_grappler_enabled()) return;  // GTEST_SKIP() did not compile
  NGraphClusterManager::EvictAllClusters();
  Graph g(OpRegistry::Global());

  int cluster_idx = NGraphClusterManager::NewCluster();

  Node* node1;
  Node* node2;

  ASSERT_OK(NodeBuilder("node1", "Placeholder")
                .Attr("dtype", DT_FLOAT)
                .Finalize(&g, &node1));
  ASSERT_OK(NodeBuilder("node2", "Placeholder")
                .Attr("dtype", DT_FLOAT)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Attr("_ngraph_backend", "INTERPRETER")
                .Finalize(&g, &node3));

  Node* node4;
  cluster_idx = NGraphClusterManager::NewCluster();
  ASSERT_OK(NodeBuilder("node4", "Abs")
                .Input(node3, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Attr("_ngraph_backend", "INTERPRETER")
                .Finalize(&g, &node4));

  Node* node5;
  std::vector<NodeBuilder::NodeOut> inputs;
  std::vector<DataType> input_types;
  inputs.push_back(NodeBuilder::NodeOut(node4, 0));
  input_types.push_back(node4->output_type(0));
  ASSERT_OK(NodeBuilder("node5", "IdentityN")
                .Input(inputs)
                .Attr("T", input_types)
                .Finalize(&g, &node5));

  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node5, Graph::kControlSlot, sink, Graph::kControlSlot);

  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();

  std::vector<std::set<ShapeHintMap>> node_shapes_hints_vect = {
      {}, {{{"node1", {2, 2}}, {"node2", {2, 2}}}}};
  int num_cases = node_shapes_hints_vect.size();
  for (int i = 0; i < num_cases; i++) {
    ASSERT_NOT_OK(
        EncapsulateClusters(&g, 0, fdeflib_new, {{"ngraph_device_id", ""}},
                            make_pair(true, node_shapes_hints_vect[i])));
  }

  free(fdeflib_new);
}

//   Placeholder-->Abs(0)-->IdN
//
//   Placeholder-->Abs(1)-->IdN
// 2 Encapsulates connected in parallel
TEST(EncapsulateClusters, AOT2) {
  if (!ngraph_tf_is_grappler_enabled()) return;
  NGraphClusterManager::EvictAllClusters();
  Graph g(OpRegistry::Global());

  int cluster_idx = NGraphClusterManager::NewCluster();

  Node* node1;
  Node* node2;

  ASSERT_OK(NodeBuilder("node1", "Placeholder")
                .Attr("dtype", DT_FLOAT)
                .Finalize(&g, &node1));
  ASSERT_OK(NodeBuilder("node2", "Placeholder")
                .Attr("dtype", DT_FLOAT)
                .Finalize(&g, &node2));
  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Abs")
                .Input(node1, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Attr("_ngraph_backend", "INTERPRETER")
                .Finalize(&g, &node3));

  Node* node4;
  cluster_idx = NGraphClusterManager::NewCluster();
  ASSERT_OK(NodeBuilder("node4", "Abs")
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Attr("_ngraph_backend", "INTERPRETER")
                .Finalize(&g, &node4));

  Node* node5;
  std::vector<NodeBuilder::NodeOut> inputs;
  std::vector<DataType> input_types;
  inputs.push_back(NodeBuilder::NodeOut(node3, 0));
  input_types.push_back(node3->output_type(0));
  ASSERT_OK(NodeBuilder("node5", "IdentityN")
                .Input(inputs)
                .Attr("T", input_types)
                .Finalize(&g, &node5));
  Node* node6;
  inputs.clear();
  input_types.clear();
  inputs.push_back(NodeBuilder::NodeOut(node4, 0));
  input_types.push_back(node4->output_type(0));
  ASSERT_OK(NodeBuilder("node6", "IdentityN")
                .Input(inputs)
                .Attr("T", input_types)
                .Finalize(&g, &node6));

  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node5, Graph::kControlSlot, sink, Graph::kControlSlot);
  g.AddEdge(node6, Graph::kControlSlot, sink, Graph::kControlSlot);

  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();

  std::vector<std::set<ShapeHintMap>> node_shapes_hints_vect = {
      {},
      {{{"node1", {2, 2}}, {"node2", {2, 2}}},
       {{"node1", {2, 3}}, {"node2", {2, 3}}}}};
  std::vector<bool> did_aot = {true, true};
  int num_cases = node_shapes_hints_vect.size();
  for (int i = 0; i < num_cases; i++) {
    auto status =
        EncapsulateClusters(&g, 0, fdeflib_new, {{"ngraph_device_id", ""}},
                            make_pair(true, node_shapes_hints_vect[i]));
    if (did_aot[i]) {
      ASSERT_OK(status);
    } else {
      ASSERT_NOT_OK(status);
      continue;
    }

    int num_encapsulates = 0;
    int num_tf_nodes = 0;
    for (auto itr : g.nodes()) {
      auto node_type = itr->type_string();
      num_encapsulates += (node_type == "NGraphEncapsulate" ? 1 : 0);
      num_tf_nodes += ((node_type == "Abs") ? 1 : 0);
    }

    ASSERT_EQ(num_encapsulates, fdeflib_new->function_size());
    ASSERT_EQ(num_encapsulates, 2);

    // No Abs nodes left in the graph
    ASSERT_EQ(num_tf_nodes, 0);

    for (auto itr : g.nodes()) {
      if (itr->type_string() == "NGraphEncapsulate") {
        string aot_info;
        bool found_exec =
            GetNodeAttr(itr->attrs(), string("_ngraph_aot_ngexec_") +
                                          (i == 0 ? "" : "2,2,") + ";/",
                        &aot_info) == tensorflow::Status::OK();
        bool found_function =
            GetNodeAttr(itr->attrs(), string("_ngraph_aot_ngfunction_") +
                                          (i == 0 ? "" : "2,2,") + ";/",
                        &aot_info) == tensorflow::Status::OK();
        if (i == 1) {
          bool found_second_exec =
              GetNodeAttr(itr->attrs(), "_ngraph_aot_ngexec_2,3,;/",
                          &aot_info) == tensorflow::Status::OK();
          found_exec = found_exec && found_second_exec;
          bool found_second_function =
              GetNodeAttr(itr->attrs(), "_ngraph_aot_ngfunction_2,3,;/",
                          &aot_info) == tensorflow::Status::OK();
          found_function = found_function && found_second_function;
        }
        ASSERT_TRUE(found_exec == did_aot[i]);
        ASSERT_TRUE(found_function == did_aot[i]);
      }
    }
  }

  free(fdeflib_new);
}

//   Placeholder-->Add(0)--->IdN
//                  ^
//                  |
//              Placeholder
// Passing shape hints that will cause TranslateGraph to fail
TEST(EncapsulateClusters, AOT3) {
  if (!ngraph_tf_is_grappler_enabled()) return;

  NGraphClusterManager::EvictAllClusters();
  Graph g(OpRegistry::Global());

  int cluster_idx = NGraphClusterManager::NewCluster();

  Node* node1;
  Node* node2;

  ASSERT_OK(NodeBuilder("node1", "Placeholder")
                .Attr("dtype", DT_FLOAT)
                .Finalize(&g, &node1));

  ASSERT_OK(NodeBuilder("node2", "Placeholder")
                .Attr("dtype", DT_FLOAT)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Attr("_ngraph_backend", "INTERPRETER")
                .Finalize(&g, &node3));
  Node* node4;
  std::vector<NodeBuilder::NodeOut> inputs;
  std::vector<DataType> input_types;
  inputs.push_back(NodeBuilder::NodeOut(node3, 0));
  input_types.push_back(node3->output_type(0));
  ASSERT_OK(NodeBuilder("node4", "IdentityN")
                .Input(inputs)
                .Attr("T", input_types)
                .Finalize(&g, &node4));

  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node4, Graph::kControlSlot, sink, Graph::kControlSlot);

  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();

  std::vector<std::set<ShapeHintMap>> node_shapes_hints_vect = {
      {{{"node1", {2, 4}}, {"node2", {2, 2}}}},
      {{{"node1", {3, 2}}, {"node2", {2, 2}}}}};
  int num_cases = node_shapes_hints_vect.size();
  for (int i = 0; i < num_cases; i++) {
    ASSERT_NOT_OK(
        EncapsulateClusters(&g, 0, fdeflib_new, {{"ngraph_device_id", ""}},
                            make_pair(true, node_shapes_hints_vect[i])));
  }

  free(fdeflib_new);
}

//   Placeholder-->Add(0)--->IdN
//                  ^
//                  |
//              Placeholder
// Placeholders contain full shape information. So AOT can happen even hints are
// empty
TEST(EncapsulateClusters, AOT4) {
  if (!ngraph_tf_is_grappler_enabled()) return;

  NGraphClusterManager::EvictAllClusters();
  Graph g(OpRegistry::Global());

  int cluster_idx = NGraphClusterManager::NewCluster();

  Node* node1;
  Node* node2;

  ASSERT_OK(NodeBuilder("node1", "Placeholder")
                .Attr("dtype", DT_FLOAT)
                .Attr("shape", TensorShape{2, 3})
                .Finalize(&g, &node1));

  ASSERT_OK(NodeBuilder("node2", "Placeholder")
                .Attr("dtype", DT_FLOAT)
                .Attr("shape", TensorShape{2, 3})
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Attr("_ngraph_backend", "INTERPRETER")
                .Finalize(&g, &node3));
  Node* node4;
  std::vector<NodeBuilder::NodeOut> inputs;
  std::vector<DataType> input_types;
  inputs.push_back(NodeBuilder::NodeOut(node3, 0));
  input_types.push_back(node3->output_type(0));
  ASSERT_OK(NodeBuilder("node4", "IdentityN")
                .Input(inputs)
                .Attr("T", input_types)
                .Finalize(&g, &node4));

  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node4, Graph::kControlSlot, sink, Graph::kControlSlot);

  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();

  // The first hint is empty (but placeholders contain complete information so
  // can AOT)
  // The second hint contains info that matches info in placeholders
  // The third hint contains hints that do not match. So Encapsulation fails
  std::vector<std::set<ShapeHintMap>> node_shapes_hints_vect = {
      {},
      {{{"node1", {2, 3}}, {"node2", {2, 3}}}},
      {{{"node1", {5, 10}}, {"node2", {15, 20}}}}};
  std::vector<bool> did_aot = {true, true, false};
  int num_cases = node_shapes_hints_vect.size();
  for (int i = 0; i < num_cases; i++) {
    auto encapsulate_status =
        EncapsulateClusters(&g, 0, fdeflib_new, {{"ngraph_device_id", ""}},
                            make_pair(true, node_shapes_hints_vect[i]));
    if (did_aot[i]) {
      ASSERT_OK(encapsulate_status);
    } else {
      ASSERT_NOT_OK(encapsulate_status);
      continue;
    }

    int num_encapsulates = 0;
    int num_tf_nodes = 0;
    for (auto itr : g.nodes()) {
      auto node_type = itr->type_string();
      num_encapsulates += (node_type == "NGraphEncapsulate" ? 1 : 0);
      num_tf_nodes += ((node_type == "Add" || node_type == "Const") ? 1 : 0);
    }

    ASSERT_EQ(num_encapsulates, fdeflib_new->function_size());
    ASSERT_EQ(num_encapsulates, 1);

    // No Add or Const nodes left in the graph
    ASSERT_EQ(num_tf_nodes, 0);

    for (auto itr : g.nodes()) {
      if (itr->type_string() == "NGraphEncapsulate") {
        string aot_info;
        bool found_exec =
            GetNodeAttr(itr->attrs(), "_ngraph_aot_ngexec_2,3,;2,3,;/",
                        &aot_info) == tensorflow::Status::OK();
        bool found_function =
            GetNodeAttr(itr->attrs(), "_ngraph_aot_ngfunction_2,3,;2,3,;/",
                        &aot_info) == tensorflow::Status::OK();
        ASSERT_TRUE(found_exec == did_aot[i]);
        ASSERT_TRUE(found_function == did_aot[i]);
      }
    }
  }

  free(fdeflib_new);
}

// Test cases for AOT:
// TODO: 1. what of scalar inputs. placeholder shape is {}?
// 2. Shape hints that cause errors in TranslateGraph?. eg trying to add [2,2]
// with [2,4]? (done, AOT3)
// 3. Encapsulate being fed by another enc (wont work) (done, AOT1)
// 4. 2 encapsulates, but both are attached to inputs, so we can AOT (done,
// AOT2)
// 5. Have a test where enc is fed by a const and a placeholder (done, AOT0)
// 6. Placeholders contain full shape. Then even with no shape hints, AOT can
// happen (done, AOT4)
// 7. Fail on bad hints (partly done, AOT4)
// 8. EncapsulateClusters compiles 2 executables due to 2 different shape hints
// (done, AOT2)
// TODO: 9. Same hint passed twice or functionally same hint passed twice (it
// should create 1 exec only).
}
}
}
