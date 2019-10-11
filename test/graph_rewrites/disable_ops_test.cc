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

#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Set using C API, get using C API
TEST(DisableOps, SimpleSettingAndGetting1) {
  char disabled_list[] = "Add,Sub";
  config::ngraph_set_disabled_ops(disabled_list);
  ASSERT_EQ(string(config::ngraph_get_disabled_ops()), "Add,Sub");

  // Clean up
  config::ngraph_set_disabled_ops("");
}

// Set using Cpp API, get using Cpp API
TEST(DisableOps, SimpleSettingAndGetting2) {
  config::SetDisabledOps("Add,Sub");
  auto expected = set<string>{"Add", "Sub"};
  ASSERT_EQ(config::GetDisabledOps(), expected);

  // Clean up
  config::ngraph_set_disabled_ops("");
}

// Set using Cpp API, get using C API
TEST(DisableOps, SimpleSettingAndGetting3) {
  config::SetDisabledOps(std::set<string>{"Add", "Sub"});
  ASSERT_EQ(string(config::ngraph_get_disabled_ops()), "Add,Sub");

  // Clean up
  config::ngraph_set_disabled_ops("");
}

// Multiple tests of setting and getting executed on a graph that adds 2 consts
// Also a unit test for MarkForClustering
TEST(DisableOps, DisableTest) {
  Graph g(OpRegistry::Global());

  config::ngraph_set_disabled_ops("");

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

  ASSERT_OK(MarkForClustering(&g, {}, "CPU"));

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

  node1->ClearAttr("_ngraph_marked_for_clustering");
  node2->ClearAttr("_ngraph_marked_for_clustering");
  node3->ClearAttr("_ngraph_marked_for_clustering");

  // Add is disabled
  config::ngraph_set_disabled_ops("Add,Mul");
  ASSERT_OK(MarkForClustering(&g, {}, "CPU"));
  ASSERT_OK(
      GetNodeAttr(node1->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node2->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_NOT_OK(
      GetNodeAttr(node3->attrs(), "_ngraph_marked_for_clustering", &marked));

  node1->ClearAttr("_ngraph_marked_for_clustering");
  node2->ClearAttr("_ngraph_marked_for_clustering");
  node3->ClearAttr("_ngraph_marked_for_clustering");

  // Add,Add,Mul,Add should work too
  config::ngraph_set_disabled_ops("Add,Add,Mul,Add");
  ASSERT_OK(MarkForClustering(&g, {}, "CPU"));
  ASSERT_OK(
      GetNodeAttr(node1->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node2->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_NOT_OK(
      GetNodeAttr(node3->attrs(), "_ngraph_marked_for_clustering", &marked));

  node1->ClearAttr("_ngraph_marked_for_clustering");
  node2->ClearAttr("_ngraph_marked_for_clustering");
  node3->ClearAttr("_ngraph_marked_for_clustering");

  // Resetting it. So Add should be accepted now
  config::ngraph_set_disabled_ops("");
  ASSERT_OK(MarkForClustering(&g, {}, "CPU"));
  ASSERT_OK(
      GetNodeAttr(node1->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node2->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  ASSERT_OK(
      GetNodeAttr(node3->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_TRUE(marked);

  node1->ClearAttr("_ngraph_marked_for_clustering");
  node2->ClearAttr("_ngraph_marked_for_clustering");
  node3->ClearAttr("_ngraph_marked_for_clustering");

  // Invalid op name should trigger an error
  config::ngraph_set_disabled_ops("Add,_InvalidOp");
  ASSERT_NOT_OK(MarkForClustering(&g, {}, "CPU"));
  ASSERT_NOT_OK(
      GetNodeAttr(node1->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_NOT_OK(
      GetNodeAttr(node2->attrs(), "_ngraph_marked_for_clustering", &marked));
  ASSERT_NOT_OK(
      GetNodeAttr(node3->attrs(), "_ngraph_marked_for_clustering", &marked));

  // Clean up
  config::ngraph_set_disabled_ops("");
}
}
}
}