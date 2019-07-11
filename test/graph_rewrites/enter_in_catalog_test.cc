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

#include "enable_variable_ops/ngraph_replace_op_utilities.h"
#include "enable_variable_ops/ngraph_replace_variable_modifiers.h"
#include "enable_variable_ops/ngraph_enter_in_catalog.h"
#include "enable_variable_ops/ngraph_catalog.h"
#include "ngraph_api.h"
#include "ngraph_assign_clusters.h"
#include "ngraph_capture_variables.h"
#include "ngraph_cluster_manager.h"
#include "ngraph_deassign_clusters.h"
#include "ngraph_encapsulate_clusters.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_rewrite_for_tracking.h"
#include "ngraph_utils.h"
#include "tf_graph_writer.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

// Graph with Assign ops which should have the attribute
// _ngraph_remove added and set to true
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
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, {}));
  ASSERT_OK(EnterInCatalog(&graph, 0));
//   GraphToPbTextFile(&graph, "1.pbtxt");

  bool remove = false;
  for (auto node : graph.op_nodes()) {
    auto node_name = node->name();
    remove = false;
    if (node_name == "Assign") {
        ASSERT_OK(
            GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
        ASSERT_TRUE(remove);
    } else if (node_name == "Var_Assign") {
        ASSERT_OK(
            GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
        ASSERT_TRUE(remove);
    }
  }

}

// Graph with one Assign ops which should not have the attribute
// _ngraph_remove added since this is a graph with trivial clusters
// and for testing purposes we use DeassignClusters to get rid of
// the trivial clusters which will lead to the graph having no
// NGraphEncapsulate nodes and hence none of the NGraphAssign's
// wil have _ngraph_remove attribute added.
TEST(CatalogTest, SmallGraph2) {
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
  ASSERT_OK(DeassignClusters(&graph));
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  ASSERT_OK(EncapsulateClusters(&graph, 0, fdeflib_new, {}));
  ASSERT_OK(EnterInCatalog(&graph, 0));

  bool remove = false;
  for (auto node : graph.op_nodes()) {
    auto node_name = node->name();
    remove = false;
    if (node_name == "Assign") {
        ASSERT_NOT_OK(
            GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
        ASSERT_FALSE(remove);
    } else if (node_name == "Var_Assign") {
        ASSERT_NOT_OK(
            GetNodeAttr(node->attrs(), "_ngraph_remove", &remove));
        ASSERT_FALSE(remove);
    }
  }
}

} // namespace testing
} // namespace ngraph_bridge
} // namespace tensorflow