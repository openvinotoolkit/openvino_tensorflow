/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

#include "ngraph_builder.h"
#include "ngraph_utils.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

using namespace std;
namespace tf = tensorflow;

namespace ngraph_bridge {

TEST(tf_exec, hello_world) {
  tf::Scope root = tf::Scope::NewRootScope();
 
  root = root.WithDevice("/device:NGRAPH:0"); 
  // Matrix A = [3 2; -1 0]
  auto A = tf::ops::Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = tf::ops::Const(root, {{3.f, 5.f}});
  // v = Ab^T
  auto v = tf::ops::MatMul(root.WithOpName("v"), A, b,
                           tf::ops::MatMul::TransposeB(true));
  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
}

TEST(tf_exec, axpy) {
  tf::GraphDef gdef;
  // auto status = tf::ReadTextProto(tf::Env::Default(), "test_py.pbtxt",
  // &gdef);
  auto status = tf::ReadTextProto(tf::Env::Default(), "test_axpy.pbtxt", &gdef);
  ASSERT_TRUE(status == tf::Status::OK()) << "Can't read protobuf graph";

  // tf::graph::SetDefaultDevice("/device:NGRAPH:0", &gdef);

  tf::SessionOptions options;
  tf::ConfigProto& config = options.config;
  config.set_allow_soft_placement(true);
  std::unique_ptr<tf::Session> session(tf::NewSession(options));

  TF_CHECK_OK(session->Create(gdef));

  // Create the inputs for this graph
  tf::Tensor x(tf::DT_FLOAT, tf::TensorShape({2, 3}));
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  tf::Tensor y(tf::DT_FLOAT, tf::TensorShape({2, 3}));
  auto y_flat = y.flat<float>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1.0;
  }

  std::vector<tf::Tensor> outputs;

  TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"mul", "add"}, {}, &outputs));

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

// Test Op :"Op_RealDiv"

TEST(tf_exec, Op_RealDiv) {
  tf::Scope root = tf::Scope::NewRootScope();
  root = root.WithDevice("/device:NGRAPH:0");

  auto A = tf::ops::Const(root, {{3.f, 5.f}, {2.f, 0.f}});
  auto B = tf::ops::Const(root, {{3.f, 2.f}, {.1f, 1.f}});
  auto r = tf::ops::RealDiv(root.WithOpName("r"), A, B);

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);

  TF_CHECK_OK(session.Run({r}, &outputs));
  
  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2,2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.0, mat(0, 0));
  EXPECT_FLOAT_EQ(2.5, mat(0, 1));
  EXPECT_FLOAT_EQ(20.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));

}

}  // namespace ngraph_bridge
