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

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

TEST(MarkForClustering, SimpleTest) {
  Graph g(OpRegistry::Global());

  Tensor t_input_0(DT_FLOAT, TensorShape{2, 3});
  Tensor t_input_1(DT_FLOAT, TensorShape{2, 3});

  Node* node1;
  ASSERT_OK(NodeBuilder("node1", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_0)
                .Finalize(&g, &node1));

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_1)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Finalize(&g, &node3));

  Node* node4;
  ASSERT_OK(NodeBuilder("node4", "Abs")
                .Input(node3, 0)
                .Attr("T", DT_FLOAT)
                .Finalize(&g, &node4));

  // Add edges from SRC to node1 and node2
  // Add edge from node3 to SINK
  // The graph is disconnected without these edges
  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node4, Graph::kControlSlot, sink, Graph::kControlSlot);

  ASSERT_OK(MarkForClustering(&g, {}));

  string backend;
  const set<string> nodes_expected_to_be_marked{"node1", "node2", "node3",
                                                "node4"};
  for (auto node : g.op_nodes()) {
    ASSERT_EQ(nodes_expected_to_be_marked.find(node->name()) !=
                  nodes_expected_to_be_marked.end(),
              NodeIsMarkedForClustering(node));
  }
}
}
}
}