/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"
#include "test/test_utilities.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

TEST(TFExec, SingleGraphOn2Threads) {
  string graph_name = "test_axpy.pbtxt";
  unique_ptr<Session> session;
  ASSERT_OK(CreateSession(graph_name, session));

  auto worker = [&session](size_t thread_id) {
    string inp_tensor_name_0{"x"};
    string inp_tensor_name_1{"y"};
    string out_tensor_name{"add"};
    std::vector<Tensor> out_tensor_vals;

    for (int i = 0; i < 10; i++) {
      Tensor inp_tensor_val(tensorflow::DT_FLOAT,
                            tensorflow::TensorShape({2, 3}));
      vector<float> in_vals(6, float(i));
      AssignInputValues<float>(inp_tensor_val, in_vals);
      Tensor out_tensor_expected_val(tensorflow::DT_FLOAT,
                                     tensorflow::TensorShape({2, 3}));
      vector<float> out_vals(6, 6.0 * float(i));
      AssignInputValues<float>(out_tensor_expected_val, out_vals);

      std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
          {inp_tensor_name_0, inp_tensor_val},
          {inp_tensor_name_1, inp_tensor_val}};

      NGRAPH_VLOG(5) << "thread_id: " << thread_id << " started: " << i;
      ASSERT_OK(session->Run(inputs, {out_tensor_name}, {}, &out_tensor_vals));
      NGRAPH_VLOG(5) << "thread_id: " << thread_id << " finished: " << i;
      Compare(out_tensor_vals, {out_tensor_expected_val});
    }
  };

  std::thread thread0(worker, 0);
  std::thread thread1(worker, 1);

  thread0.join();
  thread1.join();
}

TEST(TFExec, hello_world) {
  Scope root = Scope::NewRootScope();

  // root = root.WithDevice("/device:NGRAPH:0");
  // Matrix A = [3 2; -1 0]
  auto A = ops::Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = ops::Const(root, {{3.f, 5.f}});
  // v = Ab^T
  auto v =
      ops::MatMul(root.WithOpName("v"), A, b, ops::MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  ASSERT_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
}

TEST(TFExec, axpy) {
  GraphDef gdef;
  // auto status = ReadTextProto(Env::Default(), "test_py.pbtxt",
  // &gdef);
  auto status = ReadTextProto(Env::Default(), "test_axpy.pbtxt", &gdef);
  ASSERT_TRUE(status == Status::OK()) << "Can't read protobuf graph";

  // graph::SetDefaultDevice("/device:NGRAPH:0", &gdef);

  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tf::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tf::RewriterConfig::OFF);

  if (ngraph_tf_is_grappler_enabled()) {
    auto* custom_config = options.config.mutable_graph_options()
                              ->mutable_rewrite_options()
                              ->add_custom_optimizers();

    custom_config->set_name("ngraph-optimizer");
    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_min_graph_nodes(-1);

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_meta_optimizer_iterations(tf::RewriterConfig::ONE);
  }

  ConfigProto& config = options.config;
  config.set_allow_soft_placement(true);
  std::unique_ptr<Session> session(NewSession(options));

  ASSERT_OK(session->Create(gdef));

  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  Tensor y(DT_FLOAT, TensorShape({2, 3}));
  auto y_flat = y.flat<float>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1.0;
  }

  std::vector<Tensor> outputs;

  ASSERT_OK(session->Run({{"x", x}, {"y", y}}, {"mul", "add"}, {}, &outputs));

  ASSERT_EQ(outputs.size(), 2);
  auto mat1 = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(5.0, mat1(0, 0));
  EXPECT_FLOAT_EQ(5.0, mat1(1, 0));

  auto mat2 = outputs[1].matrix<float>();
  EXPECT_FLOAT_EQ(6.0, mat2(0, 0));
  EXPECT_FLOAT_EQ(6.0, mat2(1, 0));

  for (auto output : outputs) {
    auto output_flat = output.flat<float>();
    for (int i = 0; i < x_flat.size(); i++) {
      cout << output_flat.data()[i] << " ";
    }
    cout << endl;
  }
}

TEST(tf_exec, BatchMatMul) {
  Scope root = Scope::NewRootScope();

  auto A = ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f},
                      TensorShape({2, 2, 2, 1}));
  auto B = ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f},
                      TensorShape({2, 2, 1, 2}));

  Tensor X(DT_FLOAT, TensorShape({2, 3, 4, 5}));
  auto X_flat = X.flat<float>();
  for (int i = 0; i < X_flat.size(); i++) {
    X_flat.data()[i] = -1.1f * i;
  }
  Tensor Y(DT_FLOAT, TensorShape({2, 3, 4, 5}));
  auto Y_flat = Y.flat<float>();
  for (int i = 0; i < Y_flat.size(); i++) {
    Y_flat.data()[i] = -0.5f * i;
  }

  // Run on nGraph
  auto R = ops::BatchMatMul(root.WithOpName("R"), A, B);
  auto attrs_x = ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = ops::BatchMatMul::Attrs().AdjY(true);
  auto Z1 = ops::BatchMatMul(root.WithOpName("Z1"), X, Y, attrs_x);
  auto Z2 = ops::BatchMatMul(root.WithOpName("Z2"), X, Y, attrs_y);

  std::vector<Tensor> outputs_ng;
  std::vector<Tensor> outputs_z1_ng;
  std::vector<Tensor> outputs_z2_ng;

  ActivateNGraph();
  ClientSession session_ng(root);
  ASSERT_OK(session_ng.Run({R}, &outputs_ng));
  ASSERT_OK(session_ng.Run({Z1}, &outputs_z1_ng));
  ASSERT_OK(session_ng.Run({Z2}, &outputs_z2_ng));

  std::vector<Tensor> outputs_tf;
  std::vector<Tensor> outputs_z1_tf;
  std::vector<Tensor> outputs_z2_tf;

  DeactivateNGraph();
  ClientSession session_tf(root);
  ASSERT_OK(session_tf.Run({R}, &outputs_tf));
  ASSERT_OK(session_tf.Run({Z1}, &outputs_z1_tf));
  ASSERT_OK(session_tf.Run({Z2}, &outputs_z2_tf));

  // Check results for equality
  ASSERT_EQ(outputs_ng[0].shape(), outputs_tf[0].shape());
  ASSERT_EQ(outputs_z1_ng[0].shape(), outputs_z1_tf[0].shape());
  ASSERT_EQ(outputs_z2_ng[0].shape(), outputs_z2_tf[0].shape());
  Compare<float>(outputs_z1_ng[0], outputs_z1_tf[0]);
  Compare<float>(outputs_z2_ng[0], outputs_z2_tf[0]);

  // Activate NGraph : Otherwise the tests dont run through NGraph
  ActivateNGraph();
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow