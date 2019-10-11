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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "gtest/gtest.h"
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

// ReplaceModifier
TEST(ReplaceModifier, Momentum1) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var"), varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root.WithOpName("Assign1"), var, init_value);

  auto accum = ops::Variable(root.WithOpName("accum"), varShape, DT_FLOAT);
  auto init_value2 = ops::Const(root, {{3.f, 3.f}, {3.f, 3.f}});
  auto accum_assign =
      ops::Assign(root.WithOpName("Assign2"), accum, init_value2);

  auto grad = ops::Const(root, {{2.f, 2.f}, {2.f, 2.f}});

  auto lr = ops::Const(root, 1.f);
  auto momentum = ops::Const(root, 1.f);

  ops::ApplyMomentum::Attrs op_attr_use_nestrov;

  auto applymomentum_f = ops::ApplyMomentum(root.WithOpName("Momentum"), var,
                                            accum, lr, grad, momentum);

  op_attr_use_nestrov = op_attr_use_nestrov.UseNesterov(true);
  auto applymomentum_t =
      ops::ApplyMomentum(root.WithOpName("Momentum"), var, accum, lr, grad,
                         momentum, op_attr_use_nestrov);
  // Turn off optimizations so that all the nodes are processed
  tensorflow::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tensorflow::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tensorflow::RewriterConfig::OFF);

  // Run on nGraph
  ActivateNGraph();
  ClientSession ng_session(root, options);
  std::vector<tensorflow::Tensor> ng_outputs1;
  std::vector<tensorflow::Tensor> ng_outputs2;
  std::vector<tensorflow::Tensor> ng_outputs3;
  ASSERT_OK(ng_session.Run({{var_assign, accum_assign}}, &ng_outputs1));

  // Run on TF
  for (int i = 0; i < 10; i++) {
    ASSERT_OK(ng_session.Run({applymomentum_f}, &ng_outputs2));
  }

  for (int i = 0; i < 10; i++) {
    ASSERT_OK(ng_session.Run({applymomentum_t}, &ng_outputs3));
  }

  DeactivateNGraph();

  // Run on TF
  ClientSession tf_session(root, options);
  std::vector<tensorflow::Tensor> tf_outputs1;
  std::vector<tensorflow::Tensor> tf_outputs2;
  std::vector<tensorflow::Tensor> tf_outputs3;
  ASSERT_OK(tf_session.Run({{var_assign, accum_assign}}, &tf_outputs1));

  for (int i = 0; i < 10; i++) {
    ASSERT_OK(tf_session.Run({applymomentum_f}, &tf_outputs2));
  }

  for (int i = 0; i < 10; i++) {
    ASSERT_OK(tf_session.Run({applymomentum_t}, &tf_outputs3));
  }

  Compare(tf_outputs1, ng_outputs1);
  Compare(tf_outputs2, ng_outputs2);
  Compare(tf_outputs3, ng_outputs3);

  ActivateNGraph();
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
