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
  ASSERT_NOT_OK(enc.RewritePass(0, device_config));
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
                .Finalize(&g, &node1));

  int cluster_idx_1 = NGraphClusterManager::NewCluster();
  ASSERT_EQ(num_graphs_in_cluster_manager(), 2);

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_1)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx_1)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx_1)
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
      num_encapsulates += (node_type == "_nGraphEncapsulate" ? 1 : 0);
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
  ASSERT_OK(enc.RewritePass(0, config_map));

  std::tie(num_encapsulates, num_tf_nodes) = node_counter(&g);
  ASSERT_EQ(num_tf_nodes, 1);  // Only Abs is left
  ASSERT_EQ(num_encapsulates, 2);

  // After RewritePass, the number of clusters is still 2 and it contains
  // populated graphdefs
  ASSERT_EQ(num_graphs_in_cluster_manager(), 2);
  ASSERT_EQ(NGraphClusterManager::GetClusterGraph(0)->node_size(), 2);
  ASSERT_EQ(NGraphClusterManager::GetClusterGraph(1)->node_size(), 4);

  // The graph structure should have changed after RewritePass
  ASSERT_EQ(g.num_edges(), 6);
  ASSERT_EQ(g.num_op_nodes(), 3);
  ASSERT_EQ(g.num_nodes(), 5);
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
                .Finalize(&g, &node1));

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_1)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("_ngraph_marked_for_clustering", true)
                .Attr("_ngraph_cluster", cluster_idx)
                .Finalize(&g, &node3));

  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node3, Graph::kControlSlot, sink, Graph::kControlSlot);

  std::unordered_map<std::string, std::string> config_map;
  ASSERT_EQ(g.num_edges(), 6);
  ASSERT_EQ(g.num_op_nodes(), 3);
  ASSERT_EQ(g.num_nodes(), 5);
  ASSERT_OK(EncapsulateClusters(&g, 0, config_map));

  ASSERT_EQ(g.num_edges(), 3);
  ASSERT_EQ(g.num_op_nodes(), 1);
  ASSERT_EQ(g.num_nodes(), 3);

  int num_encapsulates = 0;
  int num_tf_nodes = 0;
  for (auto itr : g.nodes()) {
    auto node_type = itr->type_string();
    num_encapsulates += (node_type == "_nGraphEncapsulate" ? 1 : 0);
    num_tf_nodes += ((node_type == "Add" || node_type == "Const") ? 1 : 0);
  }

  // No Add or Const nodes left in the graph
  ASSERT_EQ(num_tf_nodes, 0);
}
}
}
}
