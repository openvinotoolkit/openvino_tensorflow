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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

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
  string ng_backend_type;
  ASSERT_OK(BackendManager::GetCurrentlySetBackendName(&ng_backend_type));

  // Map with all the backends, and what the boolean is_supported should be
  std::map<std::string, bool> backend_map{{"CPU", true}, {"INTERPRETER", true}};

  // Tests three cases of (Backend, is_supported=true/false)
  // 1. CPU, true
  // 2. INTERPRETER, true
  // 3. NOP, false
  for (auto it = backend_map.begin(); it != backend_map.end(); it++) {
    ASSERT_OK(BackendManager::SetBackendName(it->first));
    // Create nGraph backend
    ASSERT_OK(BackendManager::CreateBackend(it->first));
    ng::runtime::Backend* backend = BackendManager::GetBackend(it->first);

    // Create dummy node for Const as there is no default ctor yet
    // TODO(Sindhu) Change this to default once this is added in nGraph
    auto constant = ngraph::op::Constant::create(ngraph::element::f32,
                                                 ngraph::Shape{}, {2.0f});
    std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>
        TFtoNgraphOpMap{
            {"Const", {constant}},
            {"Add", {std::make_shared<ngraph::op::Add>()}},
            {"Mul",
             {std::make_shared<ngraph::op::Multiply>(),
              std::make_shared<ngraph::op::Subtract>()}},
        };

    for (auto node : graph.op_nodes()) {
      ASSERT_OK(
          IsSupportedByBackend(node, backend, TFtoNgraphOpMap, is_supported));
      ASSERT_EQ(is_supported, it->second);
    }
    BackendManager::ReleaseBackend(it->first);
  }
  ASSERT_OK(BackendManager::SetBackendName(ng_backend_type));
}
}
}
}
