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

#include "ngraph_bridge/api.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

void ResetMarkForClustering(tensorflow::Graph* graph) {
  for (auto node : graph->nodes()) {
    node->ClearAttr("_ngraph_marked_for_clustering");
  }
}

// Set using C API, get using C API
TEST(DisableOps, SimpleSettingAndGetting1) {
  char disabled_list[] = "Add,Sub";
  api::set_disabled_ops(disabled_list);
  ASSERT_EQ(string(api::get_disabled_ops()), "Add,Sub");

  // Clean up
  api::set_disabled_ops("");
}

// Set using Cpp API, get using Cpp API
TEST(DisableOps, SimpleSettingAndGetting2) {
  api::SetDisabledOps("Add,Sub");
  auto expected = set<string>{"Add", "Sub"};
  ASSERT_EQ(api::GetDisabledOps(), expected);

  // Clean up
  api::set_disabled_ops("");
}

// Set using Cpp API, get using C API
TEST(DisableOps, SimpleSettingAndGetting3) {
  api::SetDisabledOps(std::set<string>{"Add", "Sub"});
  ASSERT_EQ(string(api::get_disabled_ops()), "Add,Sub");

  // Clean up
  api::set_disabled_ops("");
}

// Multiple tests of setting and getting executed on a graph that adds 2 consts
// Also a unit test for MarkForClustering
TEST(DisableOps, DisableTest) {
  Graph g(OpRegistry::Global());

  api::set_disabled_ops("");

  Tensor t_input(DT_FLOAT, TensorShape{2, 3});
  Tensor t_shape(DT_INT32, TensorShape{2});
  t_shape.flat<int32>().data()[0] = 3;
  t_shape.flat<int32>().data()[1] = 2;

  Node* node1;
  ASSERT_OK(NodeBuilder("node1", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input)
                .Finalize(&g, &node1));

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_shape)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Finalize(&g, &node3));

  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node3, Graph::kControlSlot, sink, Graph::kControlSlot);

  ASSERT_OK(MarkForClustering(&g, {}));

  bool marked = false;

  // No ops are disabled. All 3 are expected to be clustered
  ASSERT_OK(
      GetNodeAttr(node1->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node2->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node3->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ResetMarkForClustering(&g);

  // Add is disabled
  api::set_disabled_ops("Add,Mul");
  ASSERT_OK(MarkForClustering(&g, {}));
  ASSERT_OK(
      GetNodeAttr(node1->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node2->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_NOT_OK(
      GetNodeAttr(node3->attrs(), "_ngraph_marked_for_clustering", &marked));

  ResetMarkForClustering(&g);

  // Add,Add,Mul,Add should work too
  api::set_disabled_ops("Add,Add,Mul,Add");
  ASSERT_OK(MarkForClustering(&g, {}));
  ASSERT_OK(
      GetNodeAttr(node1->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node2->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_NOT_OK(
      GetNodeAttr(node3->attrs(), "_ngraph_marked_for_clustering", &marked));

  ResetMarkForClustering(&g);

  // Resetting it. So Add should be accepted now
  api::set_disabled_ops("");
  ASSERT_OK(MarkForClustering(&g, {}));
  ASSERT_OK(
      GetNodeAttr(node1->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node2->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node3->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ResetMarkForClustering(&g);

  // Invalid op name should trigger an error
  api::set_disabled_ops("Add,_InvalidOp");
  ASSERT_NOT_OK(MarkForClustering(&g, {}));
  ASSERT_NOT_OK(
      GetNodeAttr(node1->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_NOT_OK(
      GetNodeAttr(node2->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_NOT_OK(
      GetNodeAttr(node3->attrs(), "_ngraph_marked_for_clustering", &marked));

  // Clean up
  api::set_disabled_ops("");
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow