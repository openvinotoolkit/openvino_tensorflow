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
#include "opexecuter.h"
#include "test_utilities.h"

#include "ngraph_utils.h"
#include "tf_graph_writer.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

// Test(TestCaseName, TestName)
// Please ensure
// Neither TestCaseName nor TestName should contain underscore
// https://github.com/google/googletest/blob/master/googletest/docs/primer.md
// Use only Tensors and ops::Const() to provide input to the test op

// The backword operation for "BiasAdd" on the bias tensor.
// NHWC: out_backprop input at least rank 2
// NCHW: out_backprop input only rank 4
TEST(NNOps, BiasAddGrad) {
  // define the shape for the out_backprop input shape
  vector<int64> out_backprop_shape_2D = {10, 20};
  vector<int64> out_backprop_shape_3D = {1, 3, 6};
  vector<int64> out_backprop_shape_4D = {
      1, 2, 3, 4};  // NCHW only supports 4D input/output
  vector<int64> out_backprop_shape_5D = {2, 4, 6, 8, 10};

  vector<vector<int64>> shape_vector;
  vector<int64> value_vector;

  shape_vector.push_back(out_backprop_shape_2D);
  shape_vector.push_back(out_backprop_shape_3D);
  shape_vector.push_back(out_backprop_shape_4D);
  shape_vector.push_back(out_backprop_shape_5D);

  value_vector.push_back(0.86f);
  value_vector.push_back(-0.83f);
  value_vector.push_back(-0.0003f);
  value_vector.push_back(29.09f);

  // op has one attribute : data_format
  auto attrs = ops::BiasAddGrad::Attrs();
  attrs.data_format_ = "NHWC";

  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (int i = 0; i < 4; i++) {
    Scope root = Scope::NewRootScope();
    auto tensor_shape = shape_vector[i];
    auto tensor_value = value_vector[i];

    Tensor out_backprop(DT_FLOAT, TensorShape(tensor_shape));
    AssignInputValues(out_backprop, tensor_value);

    auto R = ops::BiasAddGrad(root, out_backprop, attrs);
    std::vector<Output> sess_run_fetchoutputs = {
        R};  // tf session run parameter
    OpExecuter opexecuter(root, "BiasAddGrad", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);
    opexecuter.RunTest();
  }

  attrs.data_format_ = "NCHW";
  Scope s_nchw = Scope::NewRootScope();

  Tensor out_backprop_4D(DT_FLOAT, TensorShape(out_backprop_shape_4D));
  AssignInputValues(out_backprop_4D, 0.99f);

  auto R_4D = ops::BiasAddGrad(s_nchw, out_backprop_4D, attrs);
  std::vector<Output> sess_run_fetchoutputs_4D = {
      R_4D};  // tf session run parameter
  OpExecuter opexecuter_4D(s_nchw, "BiasAddGrad", static_input_indexes,
                           output_datatypes, sess_run_fetchoutputs_4D);

  opexecuter_4D.RunTest();
}

TEST(NNOps, Conv2DBackpropFilterNHWC) {
  // TF Default formats
  // Input NHWC :[batch, in_height, in_width, in_channels]
  vector<int64> input_size_NHWC = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  // Out_delta :[batch, out_height, out_width, out_channels]
  vector<int64> output_del_size_valid = {1, 3, 2, 2};
  vector<int64> output_del_size_same = {1, 4, 3, 2};

  std::vector<int> stride = {1, 2, 2, 1};

  std::map<std::string, vector<int64>> out_delta_size_map = {
      {"VALID", output_del_size_valid}, {"SAME", output_del_size_same}};

  vector<int> static_input_indexes = {1};
  // TEST NHWC : default data format
  for (auto map_iterator : out_delta_size_map) {
    Scope root = Scope::NewRootScope();
    auto padding_type = map_iterator.first;
    auto output_delta_size = out_delta_size_map[padding_type];

    Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
    AssignInputValues(output_delta, -1.1f);

    auto filter_sizes = ops::Const(root, {3, 3, 2, 2});

    Tensor input_data(DT_FLOAT, TensorShape(input_size_NHWC));
    AssignInputValues(input_data, -1.1f);

    auto R = ops::Conv2DBackpropFilter(root, input_data, filter_sizes,
                                       output_delta, stride, padding_type);

    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

// FusedBatchNormGrad : Gradient for batch normalization
TEST(NNOps, FusedBatchNormGrad_NHWC) {
  Scope root = Scope::NewRootScope();

  // 4D tensor for the gradient with respect to y
  Tensor y_backprop(DT_FLOAT, TensorShape({5, 4, 3, 2}));
  // 4D tensor for input data
  Tensor x(DT_FLOAT, TensorShape({5, 4, 3, 2}));
  // 1D tensor for scaling the normalized x
  Tensor scale(DT_FLOAT, TensorShape({2}));
  // 1D tensor for population mean
  Tensor reserve_space_1_mean(DT_FLOAT, TensorShape({2}));
  // 1D tensor for population varience
  Tensor reserve_space_2_varience(DT_FLOAT, TensorShape({2}));

  AssignInputValuesAnchor(y_backprop, -2.1f);
  AssignInputValuesAnchor(x, -1.1f);
  AssignInputValuesAnchor(scale, -1.6f);
  AssignInputValuesAnchor(reserve_space_1_mean, 1.1f);
  AssignInputValuesAnchor(reserve_space_2_varience, 0.5f);

  auto attrs = ops::FusedBatchNormGrad::Attrs();
  attrs.is_training_ = true;
  attrs.epsilon_ = 0.0001f;
  attrs.data_format_ = "NHWC";

  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  auto R =
      ops::FusedBatchNormGrad(root, y_backprop, x, scale, reserve_space_1_mean,
                              reserve_space_2_varience, attrs);
  std::vector<Output> sess_run_fetchoutputs = {R.x_backprop, R.scale_backprop,
                                               R.offset_backprop};
  OpExecuter opexecuter(root, "FusedBatchNormGrad", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);
  opexecuter.RunTest();

  // To do: runs on TF and gives error when running on NGRAPH
  Scope all_output_test = Scope::NewRootScope();
  vector<DataType> output_datatypes_all = {DT_FLOAT, DT_FLOAT, DT_FLOAT,
                                           DT_FLOAT, DT_FLOAT};
  R = ops::FusedBatchNormGrad(all_output_test, y_backprop, x, scale,
                              reserve_space_1_mean, reserve_space_2_varience,
                              attrs);
  std::vector<Output> sess_run_fetchoutputs_all = {
      R.x_backprop, R.scale_backprop, R.offset_backprop, R.reserve_space_3,
      R.reserve_space_4};
  OpExecuter opexecuter_all_output(all_output_test, "FusedBatchNormGrad",
                                   static_input_indexes, output_datatypes_all,
                                   sess_run_fetchoutputs_all);
  opexecuter_all_output.RunTest();
  opexecuter_all_output.ExecuteOnTF();

  // To do : Fail right now

  // attrs.is_training_ = false;
  // Scope inference_scope = Scope::NewRootScope();
  // auto R = ops::FusedBatchNormGrad(inference_scope, y_backprop, x,
  //                                 scale, reserve_space_1_mean,
  //                                 reserve_space_2_varience,attrs);
  // std::vector<Output> sess_run_fetchoutputs = {R.x_backprop,
  // R.scale_backprop, R.offset_backprop};
  // OpExecuter opexecuter_inference(inference_scope, "FusedBatchNormGrad",
  // static_input_indexes,
  //                                 output_datatypes, sess_run_fetchoutputs);

  // opexecuter_inference.RunTest();
  // opexecuter_inference.ExecuteOnNGraph();
  // opexecuter_inference.ExecuteOnTF();
}

// Test Op :"Op_L2Loss"
TEST(NNOps, Op_L2Loss) {
  std::vector<std::vector<int64>> input_sizes;
  input_sizes.push_back({2, 3, 4});
  input_sizes.push_back({0});

  vector<int> static_input_indexes = {};

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValues(input_data, 0.0);

    auto R = ops::L2Loss(root, input_data);
    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "L2Loss", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

// Computes softmax cross entropy cost and gradients to backpropagate.
TEST(NNOps, SparseSoftmaxCrossEntropyWithLogits) {
  Scope root = Scope::NewRootScope();
  int batch = 10;
  int num_of_classes = 2;

  Tensor A(DT_FLOAT, TensorShape({batch, num_of_classes}));
  Tensor B(DT_INT32, TensorShape({batch}));

  AssignInputValues(A, 2.0f);
  AssignInputIntValues(B, num_of_classes);

  vector<int> static_input_indexes = {};
  auto R = ops::SparseSoftmaxCrossEntropyWithLogits(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.loss, R.backprop};
  OpExecuter opexecuter(root, "SparseSoftmaxCrossEntropyWithLogits",
                        static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

}  // namespace testing

}  // namespace ngraph_bridge
}  // namespace tensorflow
