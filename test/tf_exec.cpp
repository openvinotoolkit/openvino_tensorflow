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

void AssertTensorEquals(tf::Tensor T1, tf::Tensor T2) {
  auto T_size = T1.flat<float>().size();
  for (int k=0; k<T_size; k++) {
    auto a = T1.flat<float>().data()[k];
    auto b = T2.flat<float>().data()[k];
    EXPECT_FLOAT_EQ(a, b);
  } 
}

TEST(tf_exec, BatchMatMul_0D) { 
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");

  tf::Tensor X1(tf::DT_FLOAT, tf::TensorShape({2, 0, 4, 5}));
  tf::Tensor Y1(tf::DT_FLOAT, tf::TensorShape({2, 0, 4, 5}));
  tf::Tensor X2(tf::DT_FLOAT, tf::TensorShape({2, 3, 0, 5}));
  tf::Tensor Y2(tf::DT_FLOAT, tf::TensorShape({2, 3, 0, 5}));

  auto attrs_x = tf::ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = tf::ops::BatchMatMul::Attrs().AdjY(true); 
  auto Z1 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z1"), X1, Y1, attrs_x);
  auto Z2 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z2"), X2, Y2, attrs_x);
  auto Z = tf::ops::BatchMatMul(dev_scope.WithOpName("Z"), X2, Y2, attrs_y);
  std::vector<tf::Tensor> outputs_z1;
  std::vector<tf::Tensor> outputs_z2;
  std::vector<tf::Tensor> outputs_z;
  // Run and fetch v
  tf::ClientSession session(dev_scope);
  TF_CHECK_OK(session.Run({Z1}, &outputs_z1)); 
  TF_CHECK_OK(session.Run({Z2}, &outputs_z2)); 
  TF_CHECK_OK(session.Run({Z}, &outputs_z));
  // Expect outputs[0] == [19; -3]

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_z1_cpu;
  std::vector<tf::Tensor> outputs_z2_cpu;
  std::vector<tf::Tensor> outputs_z_cpu;
  auto W1 = tf::ops::BatchMatMul(root.WithOpName("W1"), X1, Y1, attrs_x);
  auto W2 = tf::ops::BatchMatMul(root.WithOpName("W2"), X2, Y2, attrs_x);
  auto W = tf::ops::BatchMatMul(root.WithOpName("W"), X2, Y2, attrs_y);
  TF_CHECK_OK(sess.Run({W1}, &outputs_z1_cpu));
  TF_CHECK_OK(sess.Run({W2}, &outputs_z2_cpu));
  TF_CHECK_OK(sess.Run({W}, &outputs_z_cpu));
  ASSERT_EQ(outputs_z1[0].shape(),outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(),outputs_z2_cpu[0].shape());
  ASSERT_EQ(outputs_z[0].shape(),outputs_z_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0],outputs_z1_cpu[0]); 
  AssertTensorEquals(outputs_z2[0],outputs_z2_cpu[0]); 
  AssertTensorEquals(outputs_z[0],outputs_z_cpu[0]);
}

TEST(tf_exec, BatchMatMul) { 
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = tf::ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2,2,1})); 
  auto B = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2,1,2})); 
  tf::Tensor X(tf::DT_FLOAT, tf::TensorShape({2, 3, 4, 5}));
  auto X_flat = X.flat<float>();
  for (int i = 0; i < X_flat.size(); i++) {
    X_flat.data()[i] = -1.1f*i;
  }
  tf::Tensor Y(tf::DT_FLOAT, tf::TensorShape({2, 3, 4, 5}));
  auto Y_flat = Y.flat<float>();
  for (int i = 0; i < Y_flat.size(); i++) {
    Y_flat.data()[i] = -0.5f*i;
  }


  auto R = tf::ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  auto attrs_x = tf::ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = tf::ops::BatchMatMul::Attrs().AdjY(true); 
  auto Z1 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z1"), X, Y, attrs_x);
  auto Z2 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z2"), X, Y, attrs_y);
  std::vector<tf::Tensor> outputs;
  std::vector<tf::Tensor> outputs_z1;
  std::vector<tf::Tensor> outputs_z2;
  // Run and fetch v
  tf::ClientSession session(dev_scope);
  TF_CHECK_OK(session.Run({R}, &outputs));
  TF_CHECK_OK(session.Run({Z1}, &outputs_z1));
  TF_CHECK_OK(session.Run({Z2}, &outputs_z2)); 
  // Expect outputs[0] == [19; -3]

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_cpu;
  std::vector<tf::Tensor> outputs_z1_cpu;
  std::vector<tf::Tensor> outputs_z2_cpu;
  auto C = tf::ops::BatchMatMul(root.WithOpName("C"), A, B);
  auto W1 = tf::ops::BatchMatMul(root.WithOpName("W1"), X, Y, attrs_x); 
  auto W2 = tf::ops::BatchMatMul(root.WithOpName("W2"), X, Y, attrs_y);
  TF_CHECK_OK(sess.Run({C}, &outputs_cpu));
  TF_CHECK_OK(sess.Run({W1}, &outputs_z1_cpu));
  TF_CHECK_OK(sess.Run({W2}, &outputs_z2_cpu));
  ASSERT_EQ(outputs[0].shape(),outputs_cpu[0].shape());
  ASSERT_EQ(outputs_z1[0].shape(),outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(),outputs_z2_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0],outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0],outputs_z2_cpu[0]); 
}

TEST(tf_exec, BatchMatMul_3D) { 
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = tf::ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2,2})); 
  auto B = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2,2})); 
  auto R = tf::ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  tf::Tensor X(tf::DT_FLOAT, tf::TensorShape({2, 3, 4}));
  auto X_flat = X.flat<float>();
  for (int i = 0; i < X_flat.size(); i++) {
    X_flat.data()[i] = -1.1f*i;
  }
  tf::Tensor Y(tf::DT_FLOAT, tf::TensorShape({2, 3, 4}));
  auto Y_flat = Y.flat<float>();
  for (int i = 0; i < Y_flat.size(); i++) {
    Y_flat.data()[i] = -0.5f*i;
  }

  auto attrs_x = tf::ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = tf::ops::BatchMatMul::Attrs().AdjY(true); 
  auto Z1 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z1"), X, Y, attrs_x);
  auto Z2 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z2"), X, Y, attrs_y);
  std::vector<tf::Tensor> outputs;
  std::vector<tf::Tensor> outputs_z1;
  std::vector<tf::Tensor> outputs_z2;
  // Run and fetch v
  tf::ClientSession session(dev_scope);
  TF_CHECK_OK(session.Run({R}, &outputs));
  TF_CHECK_OK(session.Run({Z1}, &outputs_z1));
  TF_CHECK_OK(session.Run({Z2}, &outputs_z2));
  // Expect outputs[0] == [19; -3]

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_cpu;
  std::vector<tf::Tensor> outputs_z1_cpu;
  std::vector<tf::Tensor> outputs_z2_cpu;
  auto C = tf::ops::BatchMatMul(root.WithOpName("C"), A, B);
  auto W1 = tf::ops::BatchMatMul(root.WithOpName("W1"), X, Y, attrs_x); 
  auto W2 = tf::ops::BatchMatMul(root.WithOpName("W2"), X, Y, attrs_y);
  TF_CHECK_OK(sess.Run({C}, &outputs_cpu));
  TF_CHECK_OK(sess.Run({W1}, &outputs_z1_cpu));
  TF_CHECK_OK(sess.Run({W2}, &outputs_z2_cpu));
  ASSERT_EQ(outputs[0].shape(),outputs_cpu[0].shape());
  ASSERT_EQ(outputs_z1[0].shape(),outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(),outputs_z2_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0],outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0],outputs_z2_cpu[0]); 
}

TEST(tf_exec, BatchMatMul_2D) { 
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = tf::ops::Const(root, {-1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2})); 
  auto B = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f}, tf::TensorShape({2,2})); 
  auto R = tf::ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  std::vector<tf::Tensor> outputs;
  // Run and fetch R
  tf::ClientSession session(dev_scope);
  TF_CHECK_OK(session.Run({R}, &outputs));
  // Expect outputs[0] == [19; -3]
  auto mat = outputs[0].matrix<float>();
  ASSERT_EQ(-3.f, mat(0,0));
  ASSERT_EQ(-4.f, mat(0,1)); 
  ASSERT_EQ(-1.f, mat(1,0));
  ASSERT_EQ(-8.f, mat(1,1));

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_cpu;
  auto C = tf::ops::BatchMatMul(root.WithOpName("C"), A, B);
  TF_CHECK_OK(sess.Run({C}, &outputs_cpu));
  ASSERT_EQ(outputs[0].shape(),outputs_cpu[0].shape());
  AssertTensorEquals(outputs[0],outputs_cpu[0]);
}

// Test Op :"Op_RealDiv"
// With Const inputs tensorflow's constant folding optimisation converts the op to "Mul". 
// To test "RealDiv" operator, explicitly placed the op on NGRAPH and the inputs as placeholders
TEST(tf_exec, Op_RealDiv) {
  tf::Scope root = tf::Scope::NewRootScope();
  tf::Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = tf::ops::Placeholder(root, tf::DataType::DT_FLOAT);
  auto B = tf::ops::Placeholder(root, tf::DataType::DT_FLOAT);
  auto r = tf::ops::RealDiv(root_ngraph.WithOpName("r"), A, B);

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);

  TF_CHECK_OK(session.Run({{A, {{3.f, 5.f}, {2.f, 0.f}}}, {B, {{3.f, 2.f}, {.1f, 1.f}}}}, {r}, &outputs));
  
  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2,2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.0, mat(0, 0));
  EXPECT_FLOAT_EQ(2.5, mat(0, 1));
  EXPECT_FLOAT_EQ(20.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));

}

TEST(tf_exec, Op_Square) {
  tf::Scope root = tf::Scope::NewRootScope();
  root = root.WithDevice("/device:NGRAPH:0");

  auto A = tf::ops::Const(root, {{3.f, 5.f}, {-2.f, 0.f}});
  auto r = tf::ops::Square(root.WithOpName("r"), A);

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);

  TF_CHECK_OK(session.Run({r}, &outputs));

  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2,2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(9.0, mat(0, 0));
  EXPECT_FLOAT_EQ(25.0, mat(0, 1));
  EXPECT_FLOAT_EQ(4.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));
}

TEST(tf_exec, Op_Rsqrt) {
  tf::Scope root = tf::Scope::NewRootScope();
  root = root.WithDevice("/device:NGRAPH:0");

  auto A = tf::ops::Const(root, {{256.f, 16.f}, {4.f, 64.f}});
  auto r = tf::ops::Rsqrt(root.WithOpName("r"), A);

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);

  TF_CHECK_OK(session.Run({r}, &outputs));

  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2,2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.f/16.f, mat(0, 0));
  EXPECT_FLOAT_EQ(1.f/4.f, mat(0, 1));
  EXPECT_FLOAT_EQ(1.f/2.f, mat(1, 0));
  EXPECT_FLOAT_EQ(1.f/8.f, mat(1, 1));
}

}  // namespace ngraph_bridge
