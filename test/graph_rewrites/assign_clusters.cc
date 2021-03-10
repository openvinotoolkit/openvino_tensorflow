/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include "gtest/gtest.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "logging/tf_graph_writer.h"
#include "openvino_tensorflow/assign_clusters.h"
#include "openvino_tensorflow/ovtf_utils.h"
#include "test/test_utilities.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {
namespace testing {

// Test that a "Const" fed to a static input is still coalesced with the
// reader.
TEST(AssignClusters, ConstToStatic) {
  Graph g(OpRegistry::Global());

  Tensor t_input(DT_FLOAT, TensorShape{2, 3});
  Tensor t_shape(DT_INT32, TensorShape{2});
  t_shape.flat<int32>().data()[0] = 3;
  t_shape.flat<int32>().data()[1] = 2;

  Node* node1;
  ASSERT_OK(NodeBuilder("node1", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input)
                .Attr("_ovtf_marked_for_clustering", true)
                .Finalize(&g, &node1));

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Const")
                .Attr("dtype", DT_INT32)
                .Attr("value", t_shape)
                .Attr("_ovtf_marked_for_clustering", true)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Reshape")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("Tshape", DT_INT32)
                .Attr("_ovtf_marked_for_clustering", true)
                .Attr("_ovtf_static_inputs", std::vector<int32>{1})
                .Finalize(&g, &node3));

  // Add edges from SRC to node1 and node2
  // Add edge from node3 to SINK
  // The graph is disconnected without these edges
  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node3, Graph::kControlSlot, sink, Graph::kControlSlot);

  ASSERT_OK(AssignClusters(&g));

  int node1_cluster, node2_cluster, node3_cluster;
  ASSERT_OK(GetNodeCluster(node1, &node1_cluster));
  ASSERT_OK(GetNodeCluster(node2, &node2_cluster));
  ASSERT_OK(GetNodeCluster(node3, &node3_cluster));

  ASSERT_EQ(node1_cluster, node2_cluster)
      << "Node 1 and 2 did not land up in same cluster";
  ASSERT_EQ(node2_cluster, node3_cluster)
      << "Node 2 and 3 did not land up in same cluster";
}

// Given a graph of this form:
//
//  Node1--->Node2
//    \       /
//     \     /
//      |   |
//      v   v*
//      Node3
//
// where the starred input is static, we want to make sure that Node2 and Node3
// are not accidentally coalesced by a chain of events like the following:
//
// Node1-->Node2 coalesced
// Node1-->Node3 coalesced   **actually invalid, because Node1 is now in same
//                             cluster as Node2, and we can't contract 2 & 3.
TEST(AssignClusters, Cone) {
  Graph g(OpRegistry::Global());

  Tensor t(DT_FLOAT, TensorShape{2, 3});

  Node* node1;
  ASSERT_OK(NodeBuilder("node1", "_Arg")
                .Attr("T", DT_FLOAT)
                .Attr("index", 0)
                .Attr("_ovtf_marked_for_clustering", true)
                .Finalize(&g, &node1));

  // Note: we're marking this for clustering by hand, even though as of this
  // writing we don't actually mark "Shape" in real life. This is fine---we're
  // just unit-testing AssignClusters, which doesn't care.
  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Shape")
                .Input(node1, 0)
                .Attr("T", DT_FLOAT)
                .Attr("out_type", DT_INT32)
                .Attr("_ovtf_marked_for_clustering", true)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Reshape")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Attr("Tshape", DT_INT32)
                .Attr("_ovtf_marked_for_clustering", true)
                .Attr("_ovtf_static_inputs", std::vector<int32>{1})
                .Finalize(&g, &node3));

  // Add edges from SRC to node1 and node2
  // Add edge from node3 to SINK
  // The graph is disconnected without these edges
  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node3, Graph::kControlSlot, sink, Graph::kControlSlot);

  ASSERT_OK(AssignClusters(&g));

  int node1_cluster, node2_cluster, node3_cluster;
  ASSERT_OK(GetNodeCluster(node1, &node1_cluster));
  ASSERT_OK(GetNodeCluster(node2, &node2_cluster));
  ASSERT_OK(GetNodeCluster(node3, &node3_cluster));

  ASSERT_NE(node2_cluster, node3_cluster);
  ASSERT_EQ(node1_cluster, node2_cluster);
}

}  // namespace testing
}  // namespace openvino_tensorflow
}  // namespace tensorflow
