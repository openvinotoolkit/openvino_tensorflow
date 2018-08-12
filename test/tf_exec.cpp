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
#include "tensorflow/core/framework/tensor_types.h"
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

namespace tensorflow {

namespace ngraph_bridge {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

TEST(tf_exec, hello_world) {
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

#if !defined(NGRAPH_EMBEDDED_IN_TENSORFLOW)
TEST(tf_exec, axpy) {
  GraphDef gdef;
  // auto status = ReadTextProto(Env::Default(), "test_py.pbtxt",
  // &gdef);
  auto status = ReadTextProto(Env::Default(), "test_axpy.pbtxt", &gdef);
  ASSERT_TRUE(status == Status::OK()) << "Can't read protobuf graph";

  // graph::SetDefaultDevice("/device:NGRAPH:0", &gdef);

  SessionOptions options;
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
#endif

void AssertTensorEquals(Tensor T1, Tensor T2) {
  auto T_size = T1.flat<float>().size();
  auto T1_data = T1.flat<float>().data();
  auto T2_data = T2.flat<float>().data();
  for (int k = 0; k < T_size; k++) {
    auto a = T1_data[k];
    auto b = T2_data[k];
    EXPECT_FLOAT_EQ(a, b);
  }
}

void ValidateTensorData(Tensor T1, Tensor T2, float tol) {
  auto T_size = T1.flat<float>().size();
  auto T1_data = T1.flat<float>().data();
  auto T2_data = T2.flat<float>().data();
  for (int k = 0; k < T_size; k++) {
    auto a = T1_data[k];
    auto b = T2_data[k];
    if (a == 0) { 
      EXPECT_NEAR(a, b, tol);
    } else {
      auto rel = a-b;
      auto rel_div = std::abs(rel/a);
      EXPECT_TRUE(rel_div < tol);
    }
  }
}

void AssignInputValues(Tensor& A, float x) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x * i;
  }
}

TEST(tf_exec, BatchMatMul_0D) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");

  Tensor X1(DT_FLOAT, TensorShape({2, 0, 4, 5}));
  Tensor Y1(DT_FLOAT, TensorShape({2, 0, 4, 5}));
  Tensor X2(DT_FLOAT, TensorShape({2, 3, 0, 5}));
  Tensor Y2(DT_FLOAT, TensorShape({2, 3, 0, 5}));

  auto attrs_x = ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = ops::BatchMatMul::Attrs().AdjY(true);
  auto Z1 = ops::BatchMatMul(dev_scope.WithOpName("Z1"), X1, Y1, attrs_x);
  auto Z2 = ops::BatchMatMul(dev_scope.WithOpName("Z2"), X2, Y2, attrs_x);
  auto Z = ops::BatchMatMul(dev_scope.WithOpName("Z"), X2, Y2, attrs_y);
  std::vector<Tensor> outputs_z1;
  std::vector<Tensor> outputs_z2;
  std::vector<Tensor> outputs_z;
  // Run and fetch v
  ClientSession session(dev_scope);
  ASSERT_OK(session.Run({Z1}, &outputs_z1));
  ASSERT_OK(session.Run({Z2}, &outputs_z2));
  ASSERT_OK(session.Run({Z}, &outputs_z));

  ClientSession sess(root);
  std::vector<Tensor> outputs_z1_cpu;
  std::vector<Tensor> outputs_z2_cpu;
  std::vector<Tensor> outputs_z_cpu;
  auto W1 = ops::BatchMatMul(root.WithOpName("W1"), X1, Y1, attrs_x);
  auto W2 = ops::BatchMatMul(root.WithOpName("W2"), X2, Y2, attrs_x);
  auto W = ops::BatchMatMul(root.WithOpName("W"), X2, Y2, attrs_y);
  ASSERT_OK(sess.Run({W1}, &outputs_z1_cpu));
  ASSERT_OK(sess.Run({W2}, &outputs_z2_cpu));
  ASSERT_OK(sess.Run({W}, &outputs_z_cpu));
  ASSERT_EQ(outputs_z1[0].shape(), outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(), outputs_z2_cpu[0].shape());
  ASSERT_EQ(outputs_z[0].shape(), outputs_z_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0], outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0], outputs_z2_cpu[0]);
  AssertTensorEquals(outputs_z[0], outputs_z_cpu[0]);
}

void ActivateNGraph() {
  setenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1", 1);
  unsetenv("NGRAPH_TF_DISABLE");
}

void DeactivateNGraph() {
  unsetenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS");
  setenv("NGRAPH_TF_DISABLE", "1", 1);
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
  AssertTensorEquals(outputs_z1_ng[0], outputs_z1_tf[0]);
  AssertTensorEquals(outputs_z2_ng[0], outputs_z2_tf[0]);
}

TEST(tf_exec, BatchMatMul_3D) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f},
                      TensorShape({2, 2, 2}));
  auto B = ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f},
                      TensorShape({2, 2, 2}));
  auto R = ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  Tensor X(DT_FLOAT, TensorShape({2, 3, 4}));
  auto X_flat = X.flat<float>();
  for (int i = 0; i < X_flat.size(); i++) {
    X_flat.data()[i] = -1.1f * i;
  }
  Tensor Y(DT_FLOAT, TensorShape({2, 3, 4}));
  auto Y_flat = Y.flat<float>();
  for (int i = 0; i < Y_flat.size(); i++) {
    Y_flat.data()[i] = -0.5f * i;
  }

  auto attrs_x = ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = ops::BatchMatMul::Attrs().AdjY(true);
  auto Z1 = ops::BatchMatMul(dev_scope.WithOpName("Z1"), X, Y, attrs_x);
  auto Z2 = ops::BatchMatMul(dev_scope.WithOpName("Z2"), X, Y, attrs_y);
  std::vector<Tensor> outputs;
  std::vector<Tensor> outputs_z1;
  std::vector<Tensor> outputs_z2;
  ClientSession session(dev_scope);
  ASSERT_OK(session.Run({R}, &outputs));
  ASSERT_OK(session.Run({Z1}, &outputs_z1));
  ASSERT_OK(session.Run({Z2}, &outputs_z2));

  ClientSession sess(root);
  std::vector<Tensor> outputs_cpu;
  std::vector<Tensor> outputs_z1_cpu;
  std::vector<Tensor> outputs_z2_cpu;
  auto C = ops::BatchMatMul(root.WithOpName("C"), A, B);
  auto W1 = ops::BatchMatMul(root.WithOpName("W1"), X, Y, attrs_x);
  auto W2 = ops::BatchMatMul(root.WithOpName("W2"), X, Y, attrs_y);
  ASSERT_OK(sess.Run({C}, &outputs_cpu));
  ASSERT_OK(sess.Run({W1}, &outputs_z1_cpu));
  ASSERT_OK(sess.Run({W2}, &outputs_z2_cpu));
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  ASSERT_EQ(outputs_z1[0].shape(), outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(), outputs_z2_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0], outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0], outputs_z2_cpu[0]);
}

TEST(tf_exec, BatchMatMul_2D) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = ops::Const(root, {-1.f, 2.f, 3.f, 4.f}, TensorShape({2, 2}));
  auto B =
      ops::Const(root, {1.f, 0.f, -1.f, -2.f}, TensorShape({2, 2}));
  auto R = ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  std::vector<Tensor> outputs;
  ClientSession session(dev_scope);
  ASSERT_OK(session.Run({R}, &outputs));
  auto mat = outputs[0].matrix<float>();
  ASSERT_EQ(-3.f, mat(0, 0));
  ASSERT_EQ(-4.f, mat(0, 1));
  ASSERT_EQ(-1.f, mat(1, 0));
  ASSERT_EQ(-8.f, mat(1, 1));

  ClientSession sess(root);
  std::vector<Tensor> outputs_cpu;
  auto C = ops::BatchMatMul(root.WithOpName("C"), A, B);
  ASSERT_OK(sess.Run({C}, &outputs_cpu));
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  AssertTensorEquals(outputs[0], outputs_cpu[0]);
}

TEST(tf_exec, BiasAddGrad) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  Tensor X(DT_FLOAT, TensorShape({2, 3, 4, 5}));
  Tensor X2D(DT_FLOAT, TensorShape({2, 3}));
  Tensor X3D(DT_FLOAT, TensorShape({2, 3, 4}));
  Tensor X5D(DT_FLOAT, TensorShape({2, 3, 4, 5, 6}));
  auto val = 0.86f;
  AssignInputValues(X2D, val);
  AssignInputValues(X3D, val);
  AssignInputValues(X5D, val);

  auto attrs = ops::BiasAddGrad::Attrs();
  attrs.data_format_ = "NHWC";
  std::vector<Tensor> outputs_ngraph_nhwc;
  std::vector<Tensor> outputs_CPU_nhwc;
  auto R_ngraph_nhwc = ops::BiasAddGrad(dev_scope.WithOpName("R_ngraph_nhwc"), X, attrs);
  auto R_CPU_nhwc = ops::BiasAddGrad(root.WithOpName("R_CPU_nhwc"), X, attrs);
  
  ClientSession session(dev_scope);
  ClientSession sess(root);

  ASSERT_OK(session.Run({R_ngraph_nhwc}, &outputs_ngraph_nhwc));
  ASSERT_OK(sess.Run({R_CPU_nhwc}, &outputs_CPU_nhwc));

  ASSERT_EQ(outputs_ngraph_nhwc[0].shape(), outputs_CPU_nhwc[0].shape());
  ValidateTensorData(outputs_ngraph_nhwc[0], outputs_CPU_nhwc[0], 1e-6);
  //Check 2D Tensor
  R_ngraph_nhwc = ops::BiasAddGrad(dev_scope.WithOpName("R_ngraph_nhwc"), X2D, attrs);
  R_CPU_nhwc = ops::BiasAddGrad(root.WithOpName("R_CPU_nhwc"), X2D, attrs);
  
  ASSERT_OK(session.Run({R_ngraph_nhwc}, &outputs_ngraph_nhwc));
  ASSERT_OK(sess.Run({R_CPU_nhwc}, &outputs_CPU_nhwc));

  ASSERT_EQ(outputs_ngraph_nhwc[0].shape(), outputs_CPU_nhwc[0].shape());
  ValidateTensorData(outputs_ngraph_nhwc[0], outputs_CPU_nhwc[0], 1e-6);
  //check 3D tensor
  R_ngraph_nhwc = ops::BiasAddGrad(dev_scope.WithOpName("R_ngraph_nhwc"), X3D, attrs);
  R_CPU_nhwc = ops::BiasAddGrad(root.WithOpName("R_CPU_nhwc"), X3D, attrs);
  
  ASSERT_OK(session.Run({R_ngraph_nhwc}, &outputs_ngraph_nhwc));
  ASSERT_OK(sess.Run({R_CPU_nhwc}, &outputs_CPU_nhwc));

  ASSERT_EQ(outputs_ngraph_nhwc[0].shape(), outputs_CPU_nhwc[0].shape());
  ValidateTensorData(outputs_ngraph_nhwc[0], outputs_CPU_nhwc[0], 1e-6);
  //check 5D tensor
  R_ngraph_nhwc = ops::BiasAddGrad(dev_scope.WithOpName("R_ngraph_nhwc"), X5D, attrs);
  R_CPU_nhwc = ops::BiasAddGrad(root.WithOpName("R_CPU_nhwc"), X5D, attrs);
  
  ASSERT_OK(session.Run({R_ngraph_nhwc}, &outputs_ngraph_nhwc));
  ASSERT_OK(sess.Run({R_CPU_nhwc}, &outputs_CPU_nhwc));

  ASSERT_EQ(outputs_ngraph_nhwc[0].shape(), outputs_CPU_nhwc[0].shape());
  ValidateTensorData(outputs_ngraph_nhwc[0], outputs_CPU_nhwc[0], 1e-6);

  attrs.data_format_ = "NCHW";
  std::vector<Tensor> outputs_ngraph_nchw;
  std::vector<Tensor> outputs_CPU_nchw;
  
  auto R_ngraph_nchw = ops::BiasAddGrad(dev_scope.WithOpName("R_ngraph_nchw"), X, attrs);
  auto R_CPU_nchw = ops::BiasAddGrad(root.WithOpName("R_CPU_nchw"), X, attrs);
  ASSERT_OK(session.Run({R_ngraph_nchw}, &outputs_ngraph_nchw));
  ASSERT_OK(sess.Run({R_CPU_nchw}, &outputs_CPU_nchw));

  ASSERT_EQ(outputs_ngraph_nchw[0].shape(), outputs_CPU_nchw[0].shape());
  ValidateTensorData(outputs_ngraph_nchw[0], outputs_CPU_nchw[0], 1e-6);
}

TEST(tf_exec, FusedBatchNormGrad_NHWC) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  Tensor tf_input(DT_FLOAT, TensorShape({5, 3, 4, 2}));
  Tensor tf_delta(DT_FLOAT, TensorShape({5, 3, 4, 2}));
  Tensor tf_mean(DT_FLOAT, TensorShape({2}));
  Tensor tf_variance(DT_FLOAT, TensorShape({2}));
  Tensor tf_gamma(DT_FLOAT, TensorShape({2}));

  auto tf_input_flat = tf_input.flat<float>();
  for (int i = 0; i < tf_input_flat.size(); i++) {
    tf_input_flat.data()[i] = -1.1f * i;
  }
  auto tf_delta_flat = tf_delta.flat<float>();
  for (int i = 0; i < tf_delta_flat.size(); i++) {
    tf_delta_flat.data()[i] = -2.1f * i;
  }
  auto tf_mean_flat = tf_mean.flat<float>();
  for (int i = 0; i < tf_mean_flat.size(); i++) {
    tf_mean_flat.data()[i] = 1.1f * i;
  }
  auto tf_variance_flat = tf_variance.flat<float>();
  for (int i = 0; i < tf_variance_flat.size(); i++) {
    tf_variance_flat.data()[i] = 0.5f * i;
  }
  auto tf_gamma_flat = tf_gamma.flat<float>();
  for (int i = 0; i < tf_gamma_flat.size(); i++) {
    tf_gamma_flat.data()[i] = -1.6f * i;
  }

  auto attrs = ops::FusedBatchNormGrad::Attrs();
  attrs.is_training_ = true;
  attrs.epsilon_ = 0.0001f;
  attrs.data_format_ = "NHWC";

  std::vector<Tensor> outputs;
  ClientSession session(dev_scope);
  auto R =
      ops::FusedBatchNormGrad(dev_scope.WithOpName("R"), tf_delta, tf_input,
                                  tf_gamma, tf_mean, tf_variance, attrs);
  ASSERT_OK(session.Run({R.x_backprop, R.scale_backprop, R.offset_backprop},
                          &outputs));

  ClientSession sess(root);
  std::vector<Tensor> outputs_cpu;
  auto C = ops::FusedBatchNormGrad(root.WithOpName("C"), tf_delta, tf_input,
                                       tf_gamma, tf_mean, tf_variance, attrs);
  ASSERT_OK(sess.Run({C.x_backprop, C.scale_backprop, C.offset_backprop},
                       &outputs_cpu));
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  ASSERT_EQ(outputs[1].shape(), outputs_cpu[1].shape());
  ASSERT_EQ(outputs[2].shape(), outputs_cpu[2].shape());
  AssertTensorEquals(outputs[0], outputs_cpu[0]);
  AssertTensorEquals(outputs[1], outputs_cpu[1]);
  AssertTensorEquals(outputs[2], outputs_cpu[2]);
}

// Test Op :"Op_L2Loss"
TEST(tf_exec, Op_L2Loss) {
  Scope root = Scope::NewRootScope();
  Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  std::vector< std::vector<int64> > input_sizes;
  input_sizes.push_back( {2, 3, 4} );
  input_sizes.push_back( {0} );

  for(auto const& input_size: input_sizes ) {
    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValues(input_data, 0.0);

    ClientSession session(root);
    std::vector<Tensor> outputs_ngraph;
    std::vector<Tensor> outputs_cpu;

    auto r_ngraph = ops::L2Loss(
        root_ngraph.WithOpName("r_NGRAPH"), input_data);

    auto r_cpu = ops::L2Loss(
        root.WithOpName("r_CPU"), input_data);

    ASSERT_OK(session.Run({r_ngraph}, &outputs_ngraph));
    ASSERT_OK(session.Run({r_cpu}, &outputs_cpu));
    
    ASSERT_EQ(outputs_ngraph[0].shape(), outputs_cpu[0].shape());
    AssertTensorEquals(outputs_ngraph[0], outputs_cpu[0]);
    }
  }

// Test Op :"Op_Unpack"
TEST(tf_exec, Op_Unpack) {
  Scope root = Scope::NewRootScope();
  Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root = root.WithDevice("/device:CPU:0");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  std::vector< std::vector<int64> > input_sizes;

  int input_rank = 3; 
  
  input_sizes.push_back( {3, 2, 3} );
  input_sizes.push_back( {4, 3, 6} );
  input_sizes.push_back( {7, 8, 3} );
  
  std::vector< int64> axes({0, 1, 2});

  for(auto i = 0;i < input_sizes.size(); ++i) {
    Tensor input_data(DT_FLOAT, TensorShape(input_sizes[i]));
    AssignInputValues(input_data, 0.0);

    ClientSession session(root);
    std::vector<Tensor>  outputs_ngraph;
    std::vector<Tensor> outputs_cpu;
    ops::Unstack::Attrs attrs;
    attrs.axis_ = axes[i];

    auto r_ngraph = ops::Unstack(
        root_ngraph.WithOpName("r_NGRAPH"), input_data, input_sizes[i][axes[i]], attrs);

    auto r_cpu = ops::Unstack(
        root, input_data, input_sizes[i][axes[i]], attrs);

    ASSERT_OK(session.Run({r_cpu[0], r_cpu[1], r_cpu[2]}, &outputs_cpu));
    ASSERT_OK(session.Run({r_ngraph[0], r_ngraph[1], r_ngraph[2]}, &outputs_ngraph));
    for (auto j = 0; j < input_rank;++j) {     
        ASSERT_EQ(outputs_ngraph[j].shape(), outputs_cpu[j].shape());
        AssertTensorEquals(outputs_ngraph[j], outputs_cpu[j]);
      }
    }
  }


TEST(tf_exec, Tile) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  Tensor A(DT_FLOAT, TensorShape({2, 3, 4}));
  auto A_flat = A.flat<float>();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat.data()[i] = -1.1f * i;
  }
  auto X = ops::Const(root, {int64(3), int64(4), int64(2)},
                          TensorShape({3}));
  auto Y = ops::Const(root, {int64(1), int64(0), int64(3)},
                          TensorShape({3}));
  auto C = ops::Tile(dev_scope.WithOpName("C"), A, X);
  auto D = ops::Tile(dev_scope.WithOpName("D"), A, Y);
  std::vector<Tensor> outputs_C;
  std::vector<Tensor> outputs_D;

  ClientSession session(dev_scope);
  ASSERT_OK(session.Run({C}, &outputs_C));
  ASSERT_OK(session.Run({D}, &outputs_D));

  ClientSession sess(root);
  std::vector<Tensor> outputs_C_cpu;
  std::vector<Tensor> outputs_D_cpu;
  auto C_cpu = ops::Tile(root.WithOpName("C_cpu"), A, X);
  auto D_cpu = ops::Tile(root.WithOpName("D_cpu"), A, Y);
  ASSERT_OK(sess.Run({C_cpu}, &outputs_C_cpu));
  ASSERT_OK(sess.Run({D_cpu}, &outputs_D_cpu));
  ASSERT_EQ(outputs_C[0].shape(), outputs_C_cpu[0].shape());
  ASSERT_EQ(outputs_D[0].shape(), outputs_D_cpu[0].shape());
  AssertTensorEquals(outputs_C[0], outputs_C_cpu[0]);
  AssertTensorEquals(outputs_D[0], outputs_D_cpu[0]);
}

TEST(tf_exec, Op_Conv2DBackpropFilter) {
  Scope root = Scope::NewRootScope();
  Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  // TF Default formats
  // Input NHWC :[batch, in_height, in_width, in_channels]
  std::vector<int64> input_size_NHWC = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  std::vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  // Out_delta :[batch, out_height, out_width, out_channels]
  std::vector<int64> output_del_size_valid = {1, 3, 2, 2};
  std::vector<int64> output_del_size_same = {1, 4, 3, 2};
  Tensor output_delta_valid(DT_FLOAT,
                                TensorShape(output_del_size_valid));
  Tensor output_delta_same(DT_FLOAT,
                               TensorShape(output_del_size_same));
  AssignInputValues(output_delta_valid, -1.1f);
  AssignInputValues(output_delta_same, -1.1f);

  std::map<std::string, Tensor*> out_delta_size_map = {
      {"VALID", &output_delta_valid}, {"SAME", &output_delta_same}};

  std::vector<int> stride = {1, 2, 2, 1};
  Tensor input_data(DT_FLOAT, TensorShape(input_size_NHWC));
  AssignInputValues(input_data, -1.1f);

  auto filter_sizes = ops::Const(root, {3, 3, 2, 2});

  ClientSession session(root);
  std::vector<Tensor> outputs_ngraph;
  std::vector<Tensor> outputs_cpu;

  // TEST NHWC : default data format
  for (auto map_iterator : out_delta_size_map) {
    auto padding_type = map_iterator.first;
    auto output_delta = *(out_delta_size_map[padding_type]);

    auto r_ngraph = ops::Conv2DBackpropFilter(
        root_ngraph.WithOpName("r_NGRAPH"), input_data, filter_sizes,
        output_delta, stride, padding_type);

    auto r_cpu = ops::Conv2DBackpropFilter(
        root.WithOpName("r_CPU"), input_data, filter_sizes, output_delta,
        stride, padding_type);

    ASSERT_OK(session.Run({r_ngraph}, &outputs_ngraph));
    ASSERT_OK(session.Run({r_cpu}, &outputs_cpu));

    ASSERT_EQ(outputs_ngraph[0].shape(), outputs_cpu[0].shape());
    AssertTensorEquals(outputs_ngraph[0], outputs_cpu[0]);
  }

  // TEST NCHW
  // Dialtion rates > 1 not supported on CPU
  // Current testing only with dialtion rate 1
  ops::Conv2DBackpropFilter::Attrs op_attr_nchw;
  op_attr_nchw = op_attr_nchw.DataFormat("NCHW");
  op_attr_nchw = op_attr_nchw.Dilations({1, 1, 1, 1});

  ops::Conv2DBackpropFilter::Attrs op_attr_nhwc;
  op_attr_nhwc = op_attr_nhwc.DataFormat("NHWC");
  op_attr_nhwc = op_attr_nhwc.Dilations({1, 1, 1, 1});

  for (auto map_iterator : out_delta_size_map) {
    auto padding_type = map_iterator.first;
    auto output_delta = *(out_delta_size_map[padding_type]);

    auto input_data_NCHW = ops::Transpose(root, input_data, {0, 3, 1, 2});
    auto output_delta_NCHW =
        ops::Transpose(root, output_delta, {0, 3, 1, 2});
    auto stride_NCHW(stride);
    stride_NCHW[1] = stride[3];
    stride_NCHW[2] = stride[1];
    stride_NCHW[3] = stride[2];

    auto r_ngraph = ops::Conv2DBackpropFilter(
        root_ngraph.WithOpName("r_NGRAPH"), input_data_NCHW, filter_sizes,
        output_delta_NCHW, stride_NCHW, padding_type, op_attr_nchw);

    // CPU supports only NHWC
    auto r_cpu = ops::Conv2DBackpropFilter(
        root.WithOpName("r_CPU"), input_data, filter_sizes, output_delta,
        stride, padding_type, op_attr_nhwc);

    ASSERT_OK(session.Run({r_ngraph}, &outputs_ngraph));
    ASSERT_OK(session.Run({r_cpu}, &outputs_cpu));

    ASSERT_EQ(outputs_ngraph[0].shape(), outputs_cpu[0].shape());
    AssertTensorEquals(outputs_ngraph[0], outputs_cpu[0]);
  }

}  // namespace ngraph_bridge

// Test Op :"Op_RealDiv"
// With Const inputs tensorflow's constant folding optimisation converts the op
// to "Mul". To test "RealDiv" operator, explicitly placed the op on NGRAPH and
// the inputs as placeholders
TEST(tf_exec, Op_RealDiv) {
  Scope root = Scope::NewRootScope();
  Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto r = ops::RealDiv(root_ngraph.WithOpName("r"), A, B);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run(
      {{A, {{3.f, 5.f}, {2.f, 0.f}}}, {B, {{3.f, 2.f}, {.1f, 1.f}}}}, {r},
      &outputs));

  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.0, mat(0, 0));
  EXPECT_FLOAT_EQ(2.5, mat(0, 1));
  EXPECT_FLOAT_EQ(20.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));
}

TEST(tf_exec, Op_Reciprocal) {
  Scope root = Scope::NewRootScope();
  Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto r = ops::Reciprocal(root_ngraph.WithOpName("r"), A);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run({{A, {{1.f, 5.f}, {2.f, 1.f}}}}, {r}, &outputs));
  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  ASSERT_OK(session.Run({{A, {{1.f, 5.f}, {2.f, 1.f}}}}, {r}, &outputs));
  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.0, mat(0, 0));
  EXPECT_FLOAT_EQ(0.2, mat(0, 1));
  EXPECT_FLOAT_EQ(0.5, mat(1, 0));
  EXPECT_FLOAT_EQ(1.0, mat(1, 1));
}

TEST(tf_exec, Op_Square) {
  Scope root = Scope::NewRootScope();
  root = root.WithDevice("/device:NGRAPH:0");

  auto A = ops::Const(root, {{3.f, 5.f}, {-2.f, 0.f}});
  auto r = ops::Square(root.WithOpName("r"), A);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run({r}, &outputs));

  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(9.0, mat(0, 0));
  EXPECT_FLOAT_EQ(25.0, mat(0, 1));
  EXPECT_FLOAT_EQ(4.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));
}

TEST(tf_exec, Op_SquaredDifference) {
  Scope root = Scope::NewRootScope();
  Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto r = ops::SquaredDifference(root_ngraph.WithOpName("r"), A, B);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run(
      {{A, {{3.f, 5.f}, {2.f, 0.f}}}, {B, {{1.f, 2.f}, {-1.f, 1.f}}}}, {r},
      &outputs));
  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(4.0, mat(0, 0));
  EXPECT_FLOAT_EQ(9.0, mat(0, 1));
  EXPECT_FLOAT_EQ(9.0, mat(1, 0));
  EXPECT_FLOAT_EQ(1.0, mat(1, 1));
}

TEST(tf_exec, Op_Rsqrt) {
  Scope root = Scope::NewRootScope();
  root = root.WithDevice("/device:NGRAPH:0");

  auto A = ops::Const(root, {{256.f, 16.f}, {4.f, 64.f}});
  auto r = ops::Rsqrt(root.WithOpName("r"), A);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run({r}, &outputs));

  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.f / 16.f, mat(0, 0));
  EXPECT_FLOAT_EQ(1.f / 4.f, mat(0, 1));
  EXPECT_FLOAT_EQ(1.f / 2.f, mat(1, 0));
  EXPECT_FLOAT_EQ(1.f / 8.f, mat(1, 1));
}

TEST(tf_exec, Op_Negate) {
  Scope scope_cpu = Scope::NewRootScope();
  Scope scope_ng = scope_cpu.WithDevice("/device:NGRAPH:0");

  // ngraph execution
  auto A_ng = ops::Const(scope_ng, {{-256.f, 16.5f}, {0.f, 64.f}});
  auto r_ng = ops::Negate(scope_ng.WithOpName("r"), A_ng);

  std::vector<Tensor> outputs_ng;
  ClientSession session_ng(scope_ng);

  ASSERT_OK(session_ng.Run({r_ng}, &outputs_ng));
  ASSERT_EQ(outputs_ng[0].shape(), TensorShape({2, 2}));

  // reference CPU execution
  auto A_cpu = ops::Const(scope_cpu, {{-256.f, 16.5f}, {0.f, 64.f}});
  auto r_cpu = ops::Negate(scope_cpu.WithOpName("r"), A_cpu);

  std::vector<Tensor> outputs_cpu;
  ClientSession session_cpu(scope_cpu);

  ASSERT_OK(session_cpu.Run({r_cpu}, &outputs_cpu));
  ASSERT_EQ(outputs_cpu[0].shape(), TensorShape({2, 2}));

  AssertTensorEquals(outputs_cpu[0], outputs_ng[0]);
}

TEST(tf_exec, Op_FloorDiv) {
  Scope scope_cpu = Scope::NewRootScope();
  Scope scope_ng = scope_cpu.WithDevice("/device:NGRAPH:0");

  // ngraph execution
  auto A_ng = ops::Const(scope_ng, {{5.f, 6.f, 7.5f, -1.f, 2.f, -3.f},
                                        {1.3f, 1.f, -5.f, -3.f, 0.f, -2.f}});
  auto B_ng = ops::Const(scope_ng, {{1.f, 4.f, 3.f, 3.3f, -3.f, -2.f},
                                        {2.f, 2.f, 2.f, 4.f, 10.f, -3.f}});
  // Test with broadcasting
  auto C_ng = ops::Const(scope_ng, {1.f, 4.f, 3.f, 3.3f, -3.f, -2.f});
  auto r0_ng = ops::FloorDiv(scope_ng.WithOpName("r0"), A_ng, B_ng);
  auto r1_ng = ops::FloorDiv(scope_ng.WithOpName("r1"), A_ng, C_ng);

  std::vector<Tensor> outputs_ng;
  ClientSession session_ng(scope_ng);

  ASSERT_OK(session_ng.Run({r0_ng, r1_ng}, &outputs_ng));
  ASSERT_EQ(outputs_ng[0].shape(), TensorShape({2, 6}));
  ASSERT_EQ(outputs_ng[1].shape(), TensorShape({2, 6}));

  // reference CPU execution
  auto A_cpu = ops::Const(scope_cpu, {{5.f, 6.f, 7.5f, -1.f, 2.f, -3.f},
                                          {1.3f, 1.f, -5.f, -3.f, 0.f, -2.f}});
  auto B_cpu = ops::Const(scope_cpu, {{1.f, 4.f, 3.f, 3.3f, -3.f, -2.f},
                                          {2.f, 2.f, 2.f, 4.f, 10.f, -3.f}});
  auto C_cpu = ops::Const(scope_cpu, {1.f, 4.f, 3.f, 3.3f, -3.f, -2.f});
  auto r0_cpu = ops::FloorDiv(scope_cpu.WithOpName("r0"), A_cpu, B_cpu);
  auto r1_cpu = ops::FloorDiv(scope_cpu.WithOpName("r1"), A_cpu, C_cpu);

  std::vector<Tensor> outputs_cpu;
  ClientSession session_cpu(scope_cpu);

  ASSERT_OK(session_cpu.Run({r0_cpu, r1_cpu}, &outputs_cpu));
  ASSERT_EQ(outputs_cpu[0].shape(), TensorShape({2, 6}));
  ASSERT_EQ(outputs_cpu[1].shape(), TensorShape({2, 6}));

  AssertTensorEquals(outputs_cpu[0], outputs_ng[0]);
  AssertTensorEquals(outputs_cpu[1], outputs_ng[1]);
}

TEST(tf_exec, Op_FloorMod) {
  Scope scope_cpu = Scope::NewRootScope();
  Scope scope_ng = scope_cpu.WithDevice("/device:NGRAPH:0");

  // ngraph execution
  auto A_ng = ops::Const(scope_ng, {{5.f, 6.f, 7.5f, -1.f, 2.f, -3.f},
                                        {1.3f, 1.f, -5.f, -3.f, 0.f, -2.f}});
  auto B_ng = ops::Const(scope_ng, {{1.f, 4.f, 3.f, 3.3f, -3.f, -2.f},
                                        {2.f, 2.f, 2.f, 4.f, 10.f, -3.f}});
  // Test with broadcasting
  auto C_ng = ops::Const(scope_ng, {1.f, 4.f, 3.f, 3.3f, -3.f, -2.f});
  auto r0_ng = ops::FloorMod(scope_ng.WithOpName("r0"), A_ng, B_ng);
  auto r1_ng = ops::FloorMod(scope_ng.WithOpName("r1"), A_ng, C_ng);

  std::vector<Tensor> outputs_ng;
  ClientSession session_ng(scope_ng);

  ASSERT_OK(session_ng.Run({r0_ng, r1_ng}, &outputs_ng));
  ASSERT_EQ(outputs_ng[0].shape(), TensorShape({2, 6}));
  ASSERT_EQ(outputs_ng[1].shape(), TensorShape({2, 6}));

  // reference CPU execution
  auto A_cpu = ops::Const(scope_cpu, {{5.f, 6.f, 7.5f, -1.f, 2.f, -3.f},
                                          {1.3f, 1.f, -5.f, -3.f, 0.f, -2.f}});
  auto B_cpu = ops::Const(scope_cpu, {{1.f, 4.f, 3.f, 3.3f, -3.f, -2.f},
                                          {2.f, 2.f, 2.f, 4.f, 10.f, -3.f}});
  auto C_cpu = ops::Const(scope_cpu, {1.f, 4.f, 3.f, 3.3f, -3.f, -2.f});
  auto r0_cpu = ops::FloorMod(scope_cpu.WithOpName("r0"), A_cpu, B_cpu);
  auto r1_cpu = ops::FloorMod(scope_cpu.WithOpName("r1"), A_cpu, C_cpu);

  std::vector<Tensor> outputs_cpu;
  ClientSession session_cpu(scope_cpu);

  ASSERT_OK(session_cpu.Run({r0_cpu, r1_cpu}, &outputs_cpu));
  ASSERT_EQ(outputs_cpu[0].shape(), TensorShape({2, 6}));
  ASSERT_EQ(outputs_cpu[1].shape(), TensorShape({2, 6}));

  AssertTensorEquals(outputs_cpu[0], outputs_ng[0]);
  AssertTensorEquals(outputs_cpu[1], outputs_ng[1]);
}

TEST(tf_exec, Op_AddN) {
  Scope scope_cpu = Scope::NewRootScope();
  Scope scope_ng = scope_cpu.WithDevice("/device:NGRAPH:0");

  // ngraph execution
  auto A_ng = ops::Const(scope_ng, {{256.f, 16.f}, {4.f, 64.f}});
  auto B_ng = ops::Const(scope_ng, {{1.f, 2.f}, {3.f, 4.f}});
  auto C_ng = ops::Const(scope_ng, {{5.f, 6.f}, {7.f, 8.f}});
  auto r_ng =
      ops::AddN(scope_ng.WithOpName("r"), {A_ng, C_ng, B_ng, A_ng, A_ng});
  // No broadcast test needed since AddN does not support it:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/math_ops.cc#L355

  std::vector<Tensor> outputs_ng;
  ClientSession session_ng(scope_ng);
  ASSERT_OK(session_ng.Run({r_ng}, &outputs_ng));

  ASSERT_EQ(outputs_ng[0].shape(), TensorShape({2, 2}));

  // reference CPU execution
  auto A_cpu = ops::Const(scope_cpu, {{256.f, 16.f}, {4.f, 64.f}});
  auto B_cpu = ops::Const(scope_cpu, {{1.f, 2.f}, {3.f, 4.f}});
  auto C_cpu = ops::Const(scope_cpu, {{5.f, 6.f}, {7.f, 8.f}});
  auto r_cpu = ops::AddN(scope_cpu.WithOpName("r"),
                             {A_cpu, C_cpu, B_cpu, A_cpu, A_cpu});

  std::vector<Tensor> outputs_cpu;
  ClientSession session_cpu(scope_cpu);
  ASSERT_OK(session_cpu.Run({r_cpu}, &outputs_cpu));

  ASSERT_EQ(outputs_cpu[0].shape(), TensorShape({2, 2}));

  AssertTensorEquals(outputs_cpu[0], outputs_ng[0]);
}

TEST(tf_exec, Op_PreventGradient) {
  Scope scope_cpu = Scope::NewRootScope();
  Scope scope_ng = scope_cpu.WithDevice("/device:NGRAPH:0");

  // ngraph execution
  auto A_ng = ops::Placeholder(scope_cpu, DataType::DT_FLOAT);
  auto r_ng = ops::PreventGradient(scope_ng.WithOpName("r"), A_ng);

  std::vector<Tensor> outputs_ng;
  ClientSession session_ng(scope_ng);

  ASSERT_OK(session_ng.Run({{A_ng, {{2.f, 4.f}, {6.f, 8.f}}}}, {r_ng}, &outputs_ng));
  ASSERT_EQ(outputs_ng[0].shape(), TensorShape({2, 2}));

  // reference CPU execution
  auto A_cpu = ops::Placeholder(scope_cpu, DataType::DT_FLOAT);
  auto r_cpu = ops::PreventGradient(scope_cpu.WithOpName("r"), A_cpu);

  std::vector<Tensor> outputs_cpu;
  ClientSession session_cpu(scope_cpu);

  ASSERT_OK(session_cpu.Run({{A_cpu, {{2.f, 4.f}, {6.f, 8.f}}}}, {r_cpu}, &outputs_cpu));
  ASSERT_EQ(outputs_cpu[0].shape(), TensorShape({2, 2}));

  AssertTensorEquals(outputs_cpu[0], outputs_ng[0]);
}

#undef ASSERT_OK

}  // namespace ngraph_bridge

}  // namespace tensorflow
