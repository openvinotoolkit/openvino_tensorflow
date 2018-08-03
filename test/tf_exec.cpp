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

#if !defined(NGRAPH_EMBEDDED_IN_TENSORFLOW)
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
#endif

void AssertTensorEquals(tf::Tensor T1, tf::Tensor T2) {
  auto T_size = T1.flat<float>().size();
  auto T1_data = T1.flat<float>().data();
  auto T2_data = T2.flat<float>().data();
  for (int k = 0; k < T_size; k++) {
    auto a = T1_data[k];
    auto b = T2_data[k];
    EXPECT_FLOAT_EQ(a, b);
  }
}

void AssignInputValues(tf::Tensor& A) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = -1.1f * i;
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
  ASSERT_EQ(outputs_z1[0].shape(), outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(), outputs_z2_cpu[0].shape());
  ASSERT_EQ(outputs_z[0].shape(), outputs_z_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0], outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0], outputs_z2_cpu[0]);
  AssertTensorEquals(outputs_z[0], outputs_z_cpu[0]);
}

TEST(tf_exec, BatchMatMul) {
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = tf::ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f},
                          tf::TensorShape({2, 2, 2, 1}));
  auto B = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f},
                          tf::TensorShape({2, 2, 1, 2}));
  tf::Tensor X(tf::DT_FLOAT, tf::TensorShape({2, 3, 4, 5}));
  auto X_flat = X.flat<float>();
  for (int i = 0; i < X_flat.size(); i++) {
    X_flat.data()[i] = -1.1f * i;
  }
  tf::Tensor Y(tf::DT_FLOAT, tf::TensorShape({2, 3, 4, 5}));
  auto Y_flat = Y.flat<float>();
  for (int i = 0; i < Y_flat.size(); i++) {
    Y_flat.data()[i] = -0.5f * i;
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
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  ASSERT_EQ(outputs_z1[0].shape(), outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(), outputs_z2_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0], outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0], outputs_z2_cpu[0]);
}

TEST(tf_exec, BatchMatMul_3D) {
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = tf::ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f},
                          tf::TensorShape({2, 2, 2}));
  auto B = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f},
                          tf::TensorShape({2, 2, 2}));
  auto R = tf::ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  tf::Tensor X(tf::DT_FLOAT, tf::TensorShape({2, 3, 4}));
  auto X_flat = X.flat<float>();
  for (int i = 0; i < X_flat.size(); i++) {
    X_flat.data()[i] = -1.1f * i;
  }
  tf::Tensor Y(tf::DT_FLOAT, tf::TensorShape({2, 3, 4}));
  auto Y_flat = Y.flat<float>();
  for (int i = 0; i < Y_flat.size(); i++) {
    Y_flat.data()[i] = -0.5f * i;
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
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  ASSERT_EQ(outputs_z1[0].shape(), outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(), outputs_z2_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0], outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0], outputs_z2_cpu[0]);
}

TEST(tf_exec, BatchMatMul_2D) {
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = tf::ops::Const(root, {-1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2, 2}));
  auto B =
      tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f}, tf::TensorShape({2, 2}));
  auto R = tf::ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  std::vector<tf::Tensor> outputs;
  // Run and fetch R
  tf::ClientSession session(dev_scope);
  TF_CHECK_OK(session.Run({R}, &outputs));
  // Expect outputs[0] == [19; -3]
  auto mat = outputs[0].matrix<float>();
  ASSERT_EQ(-3.f, mat(0, 0));
  ASSERT_EQ(-4.f, mat(0, 1));
  ASSERT_EQ(-1.f, mat(1, 0));
  ASSERT_EQ(-8.f, mat(1, 1));

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_cpu;
  auto C = tf::ops::BatchMatMul(root.WithOpName("C"), A, B);
  TF_CHECK_OK(sess.Run({C}, &outputs_cpu));
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  AssertTensorEquals(outputs[0], outputs_cpu[0]);
}

TEST(tf_exec, FusedBatchNormGrad_NHWC) {
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  tf::Tensor tf_input(tf::DT_FLOAT, tf::TensorShape({5, 3, 4, 2}));
  tf::Tensor tf_delta(tf::DT_FLOAT, tf::TensorShape({5, 3, 4, 2}));
  tf::Tensor tf_mean(tf::DT_FLOAT, tf::TensorShape({2}));
  tf::Tensor tf_variance(tf::DT_FLOAT, tf::TensorShape({2}));
  tf::Tensor tf_gamma(tf::DT_FLOAT, tf::TensorShape({2}));

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

  auto attrs = tf::ops::FusedBatchNormGrad::Attrs();
  attrs.is_training_ = true;
  attrs.epsilon_ = 0.0001f;
  attrs.data_format_ = "NHWC";

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(dev_scope);
  auto R =
      tf::ops::FusedBatchNormGrad(dev_scope.WithOpName("R"), tf_delta, tf_input,
                                  tf_gamma, tf_mean, tf_variance, attrs);
  TF_CHECK_OK(session.Run({R.x_backprop, R.scale_backprop, R.offset_backprop},
                          &outputs));

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_cpu;
  auto C = tf::ops::FusedBatchNormGrad(root.WithOpName("C"), tf_delta, tf_input,
                                       tf_gamma, tf_mean, tf_variance, attrs);
  TF_CHECK_OK(sess.Run({C.x_backprop, C.scale_backprop, C.offset_backprop},
                       &outputs_cpu));
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  ASSERT_EQ(outputs[1].shape(), outputs_cpu[1].shape());
  ASSERT_EQ(outputs[2].shape(), outputs_cpu[2].shape());
  AssertTensorEquals(outputs[0], outputs_cpu[0]);
  AssertTensorEquals(outputs[1], outputs_cpu[1]);
  AssertTensorEquals(outputs[2], outputs_cpu[2]);
}

TEST(tf_exec, Tile) {
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  tf::Tensor A(tf::DT_FLOAT, tf::TensorShape({2, 3, 4}));
  auto A_flat = A.flat<float>();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat.data()[i] = -1.1f * i;
  }
  auto X = tf::ops::Const(root, {tf::int64(3), tf::int64(4), tf::int64(2)},
                          tf::TensorShape({3}));
  auto Y = tf::ops::Const(root, {tf::int64(1), tf::int64(0), tf::int64(3)},
                          tf::TensorShape({3}));
  auto C = tf::ops::Tile(dev_scope.WithOpName("C"), A, X);
  auto D = tf::ops::Tile(dev_scope.WithOpName("D"), A, Y);
  std::vector<tf::Tensor> outputs_C;
  std::vector<tf::Tensor> outputs_D;

  tf::ClientSession session(dev_scope);
  TF_CHECK_OK(session.Run({C}, &outputs_C));
  TF_CHECK_OK(session.Run({D}, &outputs_D));

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_C_cpu;
  std::vector<tf::Tensor> outputs_D_cpu;
  auto C_cpu = tf::ops::Tile(root.WithOpName("C_cpu"), A, X);
  auto D_cpu = tf::ops::Tile(root.WithOpName("D_cpu"), A, Y);
  TF_CHECK_OK(sess.Run({C_cpu}, &outputs_C_cpu));
  TF_CHECK_OK(sess.Run({D_cpu}, &outputs_D_cpu));
  ASSERT_EQ(outputs_C[0].shape(), outputs_C_cpu[0].shape());
  ASSERT_EQ(outputs_D[0].shape(), outputs_D_cpu[0].shape());
  AssertTensorEquals(outputs_C[0], outputs_C_cpu[0]);
  AssertTensorEquals(outputs_D[0], outputs_D_cpu[0]);
}

TEST(tf_exec, Op_Conv2DBackpropFilter) {
  tf::Scope root = tf::Scope::NewRootScope();
  tf::Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  // TF Default formats
  // Input NHWC :[batch, in_height, in_width, in_channels]
  std::vector<tf::int64> input_size_NHWC = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  std::vector<tf::int64> filter_size_HWIO = {3, 3, 2, 2};
  // Out_delta :[batch, out_height, out_width, out_channels]
  std::vector<tf::int64> output_del_size_valid = {1, 3, 2, 2};
  std::vector<tf::int64> output_del_size_same = {1, 4, 3, 2};
  tf::Tensor output_delta_valid(tf::DT_FLOAT,
                                tf::TensorShape(output_del_size_valid));
  tf::Tensor output_delta_same(tf::DT_FLOAT,
                               tf::TensorShape(output_del_size_same));
  AssignInputValues(output_delta_valid);
  AssignInputValues(output_delta_same);

  std::map<std::string, tf::Tensor*> out_delta_size_map = {
      {"VALID", &output_delta_valid}, {"SAME", &output_delta_same}};

  std::vector<int> stride = {1, 2, 2, 1};
  tf::Tensor input_data(tf::DT_FLOAT, tf::TensorShape(input_size_NHWC));
  AssignInputValues(input_data);

  auto filter_sizes = tf::ops::Const(root, {3, 3, 2, 2});

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs_ngraph;
  std::vector<tf::Tensor> outputs_cpu;

  // TEST NHWC : default data format
  for (auto map_iterator : out_delta_size_map) {
    auto padding_type = map_iterator.first;
    auto output_delta = *(out_delta_size_map[padding_type]);

    auto r_ngraph = tf::ops::Conv2DBackpropFilter(
        root_ngraph.WithOpName("r_NGRAPH"), input_data, filter_sizes,
        output_delta, stride, padding_type);

    auto r_cpu = tf::ops::Conv2DBackpropFilter(
        root.WithOpName("r_CPU"), input_data, filter_sizes, output_delta,
        stride, padding_type);

    TF_CHECK_OK(session.Run({r_ngraph}, &outputs_ngraph));
    TF_CHECK_OK(session.Run({r_cpu}, &outputs_cpu));

    ASSERT_EQ(outputs_ngraph[0].shape(), outputs_cpu[0].shape());
    AssertTensorEquals(outputs_ngraph[0], outputs_cpu[0]);
  }

  // TEST NCHW
  // Dialtion rates > 1 not supported on CPU
  // Current testing only with dialtion rate 1
  tf::ops::Conv2DBackpropFilter::Attrs op_attr_nchw;
  op_attr_nchw = op_attr_nchw.DataFormat("NCHW");
  op_attr_nchw = op_attr_nchw.Dilations({1, 1, 1, 1});

  tf::ops::Conv2DBackpropFilter::Attrs op_attr_nhwc;
  op_attr_nhwc = op_attr_nhwc.DataFormat("NHWC");
  op_attr_nhwc = op_attr_nhwc.Dilations({1, 1, 1, 1});

  for (auto map_iterator : out_delta_size_map) {
    auto padding_type = map_iterator.first;
    auto output_delta = *(out_delta_size_map[padding_type]);

    auto input_data_NCHW = tf::ops::Transpose(root, input_data, {0, 3, 1, 2});
    auto output_delta_NCHW =
        tf::ops::Transpose(root, output_delta, {0, 3, 1, 2});
    auto stride_NCHW(stride);
    stride_NCHW[1] = stride[3];
    stride_NCHW[2] = stride[1];
    stride_NCHW[3] = stride[2];

    auto r_ngraph = tf::ops::Conv2DBackpropFilter(
        root_ngraph.WithOpName("r_NGRAPH"), input_data_NCHW, filter_sizes,
        output_delta_NCHW, stride_NCHW, padding_type, op_attr_nchw);

    // CPU supports only NHWC
    auto r_cpu = tf::ops::Conv2DBackpropFilter(
        root.WithOpName("r_CPU"), input_data, filter_sizes, output_delta,
        stride, padding_type, op_attr_nhwc);

    TF_CHECK_OK(session.Run({r_ngraph}, &outputs_ngraph));
    TF_CHECK_OK(session.Run({r_cpu}, &outputs_cpu));

    ASSERT_EQ(outputs_ngraph[0].shape(), outputs_cpu[0].shape());
    AssertTensorEquals(outputs_ngraph[0], outputs_cpu[0]);
  }

}  // namespace ngraph_bridge

// Test Op :"Op_RealDiv"
// With Const inputs tensorflow's constant folding optimisation converts the op
// to "Mul". To test "RealDiv" operator, explicitly placed the op on NGRAPH and
// the inputs as placeholders
TEST(tf_exec, Op_RealDiv) {
  tf::Scope root = tf::Scope::NewRootScope();
  tf::Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = tf::ops::Placeholder(root, tf::DataType::DT_FLOAT);
  auto B = tf::ops::Placeholder(root, tf::DataType::DT_FLOAT);
  auto r = tf::ops::RealDiv(root_ngraph.WithOpName("r"), A, B);

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);

  TF_CHECK_OK(session.Run(
      {{A, {{3.f, 5.f}, {2.f, 0.f}}}, {B, {{3.f, 2.f}, {.1f, 1.f}}}}, {r},
      &outputs));

  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.0, mat(0, 0));
  EXPECT_FLOAT_EQ(2.5, mat(0, 1));
  EXPECT_FLOAT_EQ(20.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));
}

TEST(tf_exec, Op_Reciprocal) {
  tf::Scope root = tf::Scope::NewRootScope();
  tf::Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = tf::ops::Placeholder(root, tf::DataType::DT_FLOAT);
  auto r = tf::ops::Reciprocal(root_ngraph.WithOpName("r"), A);

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);

  TF_CHECK_OK(session.Run({{A, {{1.f, 5.f}, {2.f, 1.f}}}}, {r}, &outputs));
  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.0, mat(0, 0));
  EXPECT_FLOAT_EQ(0.2, mat(0, 1));
  EXPECT_FLOAT_EQ(0.5, mat(1, 0));
  EXPECT_FLOAT_EQ(1.0, mat(1, 1));
}

TEST(tf_exec, Op_Square) {
  tf::Scope root = tf::Scope::NewRootScope();
  root = root.WithDevice("/device:NGRAPH:0");

  auto A = tf::ops::Const(root, {{3.f, 5.f}, {-2.f, 0.f}});
  auto r = tf::ops::Square(root.WithOpName("r"), A);

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);

  TF_CHECK_OK(session.Run({r}, &outputs));

  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(9.0, mat(0, 0));
  EXPECT_FLOAT_EQ(25.0, mat(0, 1));
  EXPECT_FLOAT_EQ(4.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));
}

TEST(tf_exec, Op_SquaredDifference) {
  tf::Scope root = tf::Scope::NewRootScope();
  tf::Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = tf::ops::Placeholder(root, tf::DataType::DT_FLOAT);
  auto B = tf::ops::Placeholder(root, tf::DataType::DT_FLOAT);
  auto r = tf::ops::SquaredDifference(root_ngraph.WithOpName("r"), A, B);

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);

  TF_CHECK_OK(session.Run(
      {{A, {{3.f, 5.f}, {2.f, 0.f}}}, {B, {{1.f, 2.f}, {-1.f, 1.f}}}}, {r},
      &outputs));
  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(4.0, mat(0, 0));
  EXPECT_FLOAT_EQ(9.0, mat(0, 1));
  EXPECT_FLOAT_EQ(9.0, mat(1, 0));
  EXPECT_FLOAT_EQ(1.0, mat(1, 1));
}

TEST(tf_exec, Op_Rsqrt) {
  tf::Scope root = tf::Scope::NewRootScope();
  root = root.WithDevice("/device:NGRAPH:0");

  auto A = tf::ops::Const(root, {{256.f, 16.f}, {4.f, 64.f}});
  auto r = tf::ops::Rsqrt(root.WithOpName("r"), A);

  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);

  TF_CHECK_OK(session.Run({r}, &outputs));

  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.f / 16.f, mat(0, 0));
  EXPECT_FLOAT_EQ(1.f / 4.f, mat(0, 1));
  EXPECT_FLOAT_EQ(1.f / 2.f, mat(1, 0));
  EXPECT_FLOAT_EQ(1.f / 8.f, mat(1, 1));
}

}  // namespace ngraph_bridge
