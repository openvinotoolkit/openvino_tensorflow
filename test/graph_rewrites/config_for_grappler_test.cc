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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

// This test can only be run when nGraph-bridge is built with grappler
// When running with other modes, grappler's ngraph-optimizer is not
// run, none of the nodes are encapsulated, no attributes are attached
// etc.,etc.

TEST(GrapplerConfig, RConfig1) {
  // Create Graph
  Scope root = Scope::NewRootScope();
  auto A = ops::Const(root.WithOpName("A"), {3.f, 2.f});
  auto B = ops::Const(root.WithOpName("B"), {3.f, 2.f});
  auto Add = ops::Add(root.WithOpName("Add"), A, B);
  auto C = ops::Const(root.WithOpName("C"), {3.f, 2.f});
  auto Mul = ops::Mul(root.WithOpName("Mul"), Add, C);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // set device specification
  for (auto node : graph.op_nodes()) {
    node->set_requested_device("CPU");
  }

  // Create GraphDef and Grappler
  grappler::GrapplerItem item;
  graph.ToGraphDef(&item.graph);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("ngraph-optimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
  auto* custom_config = rewriter_config.add_custom_optimizers();
  custom_config->set_name("ngraph-optimizer");

  // Run grappler
  tensorflow::grappler::MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  // const Status status = optimizer.Optimize(nullptr, item, &output);
  ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  // GraphDef to Graph
  Graph output_graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  ASSERT_OK(ConvertGraphDefToGraph(opts, output, &output_graph));

  // There is only one node in the graph
  // And it is an NGraphEncapsulateOp
  ASSERT_EQ(output_graph.num_op_nodes(), 1);
  Node* ng_encap = nullptr;

  // TODO(malikshr) : Find a way to avoid loop
  for (auto node : output_graph.op_nodes()) {
    ng_encap = node;
  }
  ASSERT_NE(ng_encap, nullptr);
}

TEST(GrapplerConfig, RConfig4) {
  // Create Graph
  Scope root = Scope::NewRootScope();
  auto A = ops::Const(root.WithOpName("A"), {3.f, 2.f});
  auto B = ops::Const(root.WithOpName("B"), {3.f, 2.f});
  auto Add = ops::Add(root.WithOpName("Add"), A, B);
  auto C = ops::Const(root.WithOpName("C"), {3.f, 2.f});
  auto Mul = ops::Mul(root.WithOpName("Mul"), Add, C);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // set device specification
  for (auto node : graph.op_nodes()) {
    node->set_requested_device("CPU");
  }

  // Create GraphDef and Grappler
  grappler::GrapplerItem item;
  graph.ToGraphDef(&item.graph);
  ConfigProto config_proto;

  auto test_echo = AttrValue();
  test_echo.set_s("hi");

  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("ngraph-optimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);

  auto* custom_config = rewriter_config.add_custom_optimizers();
  custom_config->set_name("ngraph-optimizer");
  (*custom_config->mutable_parameter_map())["test_echo"] = test_echo;

  // Run grappler
  tensorflow::grappler::MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  // const Status status = optimizer.Optimize(nullptr, item, &output);
  ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  // GraphDef to Graph
  Graph output_graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  ASSERT_OK(ConvertGraphDefToGraph(opts, output, &output_graph));

  // There is only one node in the graph
  // And it is an NGraphEncapsulateOp
  ASSERT_EQ(output_graph.num_op_nodes(), 1);
  Node* ng_encap = nullptr;

  for (auto node : output_graph.op_nodes()) {
    ng_encap = node;
  }
  ASSERT_NE(ng_encap, nullptr);
  string ng_test_echo;
  ASSERT_OK(GetNodeAttr(ng_encap->attrs(), "_ngraph_test_echo", &ng_test_echo));
  ASSERT_EQ(ng_test_echo, "hi");
}

// Test the failure case where the compulsory attribute device_id
// is not provided using the rewriter config
TEST(GrapplerConfig, RConfig5) {
  // Create Graph
  Scope root = Scope::NewRootScope();
  auto A = ops::Const(root.WithOpName("A"), {3.f, 2.f});
  auto B = ops::Const(root.WithOpName("B"), {3.f, 2.f});
  auto Add = ops::Add(root.WithOpName("Add"), A, B);
  auto C = ops::Const(root.WithOpName("C"), {3.f, 2.f});
  auto Mul = ops::Mul(root.WithOpName("Mul"), Add, C);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // set device specification
  for (auto node : graph.op_nodes()) {
    node->set_requested_device("CPU");
  }

  // Create GraphDef and Grappler
  grappler::GrapplerItem item;
  graph.ToGraphDef(&item.graph);
  ConfigProto config_proto;

  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("ngraph-optimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);

  auto* custom_config = rewriter_config.add_custom_optimizers();
  custom_config->set_name("ngraph-optimizer");

  // Run grappler
  tensorflow::grappler::MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;

  ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
