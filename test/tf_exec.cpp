/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include <thread>
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
#include "tensorflow/core/public/version.h"
#if (TF_MAJOR_VERSION >= 2) && (TF_MINOR_VERSION > 2)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#else
#include "tensorflow/core/graph/graph_constructor.h"
#endif
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "openvino_tensorflow/ovtf_builder.h"
#include "openvino_tensorflow/ovtf_utils.h"
#include "openvino_tensorflow/version.h"
#include "test/test_utilities.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {
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

      OVTF_VLOG(5) << "thread_id: " << thread_id << " started: " << i;
      ASSERT_OK(session->Run(inputs, {out_tensor_name}, {}, &out_tensor_vals));
      OVTF_VLOG(5) << "thread_id: " << thread_id << " finished: " << i;
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

  if (is_grappler_enabled()) {
    auto* custom_config = options.config.mutable_graph_options()
                              ->mutable_rewrite_options()
                              ->add_custom_optimizers();

    custom_config->set_name("ovtf-optimizer");
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

}  // namespace testing
}  // namespace openvino_tensorflow
}  // namespace tensorflow