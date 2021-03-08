/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
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
#include "openvino_tensorflow/backend_manager.h"
#include "openvino_tensorflow/default_opset.h"
#include "openvino_tensorflow/assign_clusters.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace openvino_tensorflow {
namespace testing {

// Test to verify if the set backend supports all the ops in a graph
// Tests functionality of IsSupportedByBackend in mark for clustering
// CPU, INTERPRETER should supports these ops, NOP does not.

TEST(OpByOpCapability, Backend) {
  // Create Graph
  Scope root = Scope::NewRootScope();
  auto A = ops::Const(root.WithOpName("A"), {3.f, 2.f});
  auto B = ops::Const(root.WithOpName("B"), {3.f, 2.f});
  auto Add = ops::Add(root.WithOpName("Add"), A, B);
  auto C = ops::Const(root.WithOpName("C"), {3.f, 2.f});
  auto Mul = ops::Mul(root.WithOpName("Mul"), Add, C);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  bool is_supported;
  auto backend = BackendManager::GetBackend();
  ASSERT_NE(backend, nullptr);

  auto constant = ngraph::op::Constant::create(ngraph::element::f32,
                                               ngraph::Shape{}, {2.0f});
  std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>
      TFtoNgraphOpMap{
          {"Const", {constant}},
          {"Add", {std::make_shared<opset::Add>()}},
          {"Mul",
           {std::make_shared<opset::Multiply>(),
            std::make_shared<opset::Subtract>()}},
      };

  for (auto node : graph.op_nodes()) {
    ASSERT_OK(
        IsSupportedByBackend(node, backend, TFtoNgraphOpMap, is_supported));
    ASSERT_EQ(is_supported, true);
  }
}
}
}// namespace openvino_tensorflow 
}// namespace tensorflow
