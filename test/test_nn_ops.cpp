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

// The backward operation for "BiasAdd" on the bias tensor.
// NHWC: out_backprop input at least rank 2
// NCHW: out_backprop input only rank 4
TEST(NNOps, BiasAddGrad) {
  // define the shape for the out_backprop input shape
  vector<int64> out_backprop_shape_2D = {10, 20};
  vector<int64> out_backprop_shape_3D = {1, 3, 6};
  vector<int64> out_backprop_shape_4D = {
      1, 2, 3, 4};  // NCHW only supports 4D input/output on TF CPU
  vector<int64> out_backprop_shape_5D = {2, 4, 6, 8, 10};

  vector<vector<int64>> shape_vector;

  shape_vector.push_back(out_backprop_shape_2D);
  shape_vector.push_back(out_backprop_shape_3D);
  shape_vector.push_back(out_backprop_shape_4D);
  shape_vector.push_back(out_backprop_shape_5D);

  auto attrs = ops::BiasAddGrad::Attrs();
  attrs.data_format_ = "NHWC";

  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (int i = 0; i < 4; i++) {
    Scope root = Scope::NewRootScope();
    auto tensor_shape = shape_vector[i];

    Tensor out_backprop(DT_FLOAT, TensorShape(tensor_shape));
    AssignInputValuesRandom<float>(out_backprop, -5, 10);

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
  AssignInputValuesRandom<float>(out_backprop_4D, -10, 20);

  auto R_4D = ops::BiasAddGrad(s_nchw, out_backprop_4D, attrs);
  std::vector<Output> sess_run_fetchoutputs_4D = {
      R_4D};  // tf session run parameter
  OpExecuter opexecuter_4D(s_nchw, "BiasAddGrad", static_input_indexes,
                           output_datatypes, sess_run_fetchoutputs_4D);

  opexecuter_4D.RunTest();
}

// TF does not support NCHW kernels
// To test NCHW data format
// Create graph with inputs in NCHW format and execute on NGraph
// Reshape the NCHW inputs to NHWC and run on TF
// Compare the results
TEST(NNOps, Conv2DBackpropFilterNCHWSame) {
  string padding_type = "SAME";
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size_HWIO = {3, 3, 2, 2};
  vector<DataType> output_datatypes = {DT_FLOAT};
  vector<int> static_input_indexes = {1};

  // Define scope for nGraph
  // Data Format : NCHW
  Scope ngraph_scope = Scope::NewRootScope();
  vector<int64> input_size_NCHW = {1, 2, 7, 6};
  Tensor input_data_NCHW(DT_FLOAT, TensorShape(input_size_NCHW));
  AssignInputValuesRandom<float>(input_data_NCHW, -15.0f, 15.0f);

  vector<int64> output_del_size_NCHW = {1, 2, 4, 3};
  Tensor output_delta_NCHW(DT_FLOAT, TensorShape(output_del_size_NCHW));
  AssignInputValuesRandom<float>(output_delta_NCHW, -20.0f, 20.0f);

  auto filter_sizes = ops::Const(ngraph_scope, filter_size_HWIO);
  vector<int> stride_NCHW = {1, 1, 2, 2};

  // Dilation rates > 1 not supported by TF on CPU
  ops::Conv2DBackpropFilter::Attrs op_attr_nchw;
  op_attr_nchw = op_attr_nchw.DataFormat("NCHW");
  op_attr_nchw = op_attr_nchw.Dilations({1, 1, 1, 1});

  auto r_ngraph = ops::Conv2DBackpropFilter(
      ngraph_scope, input_data_NCHW, filter_sizes, output_delta_NCHW,
      stride_NCHW, padding_type, op_attr_nchw);

  vector<Output> sess_run_fetchoutputs = {r_ngraph};
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropFilter",
                               static_input_indexes, output_datatypes,
                               sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter_ngraph.ExecuteOnNGraph(ngraph_outputs);

  // Define scope for tf (without nGraph)
  // Data Format: NHWC
  Scope tf_scope = Scope::NewRootScope();
  auto input_data_NHWC =
      ops::Transpose(tf_scope, input_data_NCHW, {0, 2, 3, 1});
  auto output_delta_NHWC =
      ops::Transpose(tf_scope, output_delta_NCHW, {0, 2, 3, 1});
  auto filter_sizes_tf = ops::Const(tf_scope, filter_size_HWIO);
  vector<int> stride_NHWC = {1, 2, 2, 1};
  auto r_tf =
      ops::Conv2DBackpropFilter(tf_scope, input_data_NHWC, filter_sizes_tf,
                                output_delta_NHWC, stride_NHWC, padding_type);
  vector<Output> sess_run_fetchoutputs_tf = {r_tf};
  OpExecuter opexecuter_tf(tf_scope, "Conv2DBackpropFilter",
                           static_input_indexes, output_datatypes,
                           sess_run_fetchoutputs_tf);

  vector<Tensor> tf_outputs;
  opexecuter_tf.ExecuteOnTF(tf_outputs);

  // Compare NGraph and TF Outputs
  Compare(tf_outputs, ngraph_outputs);
}

TEST(NNOps, Conv2DBackpropFilterNCHWSameWithDilation) {
  // TF Default formats : NHWC
  vector<int64> input_size = {1, 2, 7, 6};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size = {3, 3, 2, 2};
  vector<int64> output_delta_size = {1, 2, 4, 3};
  vector<int> stride = {1, 1, 2, 2};
  string padding_type = "SAME";
  // change the dilation attribute
  ops::Conv2DBackpropFilter::Attrs op_attr;
  op_attr = op_attr.DataFormat("NCHW");
  op_attr = op_attr.Dilations({1, 1, 3, 2});

  vector<int> static_input_indexes = {1};

  Scope root = Scope::NewRootScope();

  Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
  int size_vect =
      std::accumulate(output_delta_size.begin(), output_delta_size.end(), 1,
                      std::multiplies<int64>());
  std::vector<float> output_vector(size_vect);
  std::iota(output_vector.begin(), output_vector.end(), 0);
  AssignInputValues<float>(output_delta, output_vector);

  auto filter_sizes = ops::Const(root, filter_size);

  Tensor input_data(DT_FLOAT, TensorShape(input_size));
  size_vect = std::accumulate(input_size.begin(), input_size.end(), 1,
                              std::multiplies<int64>());
  std::vector<float> input_vector(size_vect);
  std::iota(input_vector.begin(), input_vector.end(), 0);
  AssignInputValues<float>(input_data, input_vector);

  auto R =
      ops::Conv2DBackpropFilter(root, input_data, filter_sizes, output_delta,
                                stride, padding_type, op_attr);

  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter.ExecuteOnNGraph(ngraph_outputs);

  // Construct tf_outputs using gathered values
  vector<Tensor> tf_outputs;
  Tensor tf_result(DT_FLOAT, TensorShape({3, 3, 2, 2}));
  std::vector<float> tf_output_vector{
      542.f,  1214.f,  2054.f, 4742.f, 827.f,  1907.f, 2969.f, 7073.f, 550.f,
      1318.f, 1894.f,  4678.f, 1324.f, 3244.f, 3340.f, 9292.f, 1942.f, 4966.f,
      4714.f, 13786.f, 1244.f, 3356.f, 2924.f, 9068.f, 350.f,  1598.f, 854.f,
      4118.f, 467.f,   2411.f, 1097.f, 6065.f, 262.f,  1606.f, 598.f,  3958.f};
  AssignInputValues(tf_result, tf_output_vector);
  tf_outputs.push_back(tf_result);

  Compare(tf_outputs, ngraph_outputs);
}  // end of Conv2DBackpropFilterNCHWSameWithDilation

TEST(NNOps, Conv2DBackpropFilterNCHWValid) {
  string padding_type = "VALID";
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size_HWIO = {3, 3, 2, 2};
  vector<DataType> output_datatypes = {DT_FLOAT};
  vector<int> static_input_indexes = {1};

  // Define scope for nGraph
  // Data Format : NCHW
  Scope ngraph_scope = Scope::NewRootScope();
  vector<int64> input_size_NCHW = {1, 2, 7, 6};
  Tensor input_data_NCHW(DT_FLOAT, TensorShape(input_size_NCHW));
  AssignInputValuesRandom<float>(input_data_NCHW, -15.0f, 15.0f);

  vector<int64> output_del_size_NCHW = {1, 2, 3, 2};
  Tensor output_delta_NCHW(DT_FLOAT, TensorShape(output_del_size_NCHW));
  AssignInputValuesRandom<float>(output_delta_NCHW, -20.0f, 20.0f);

  auto filter_sizes = ops::Const(ngraph_scope, filter_size_HWIO);
  vector<int> stride_NCHW = {1, 1, 2, 2};

  // Dilation rates > 1 not supported by TF on CPU
  ops::Conv2DBackpropFilter::Attrs op_attr_nchw;
  op_attr_nchw = op_attr_nchw.DataFormat("NCHW");
  op_attr_nchw = op_attr_nchw.Dilations({1, 1, 1, 1});

  auto r_ngraph = ops::Conv2DBackpropFilter(
      ngraph_scope, input_data_NCHW, filter_sizes, output_delta_NCHW,
      stride_NCHW, padding_type, op_attr_nchw);

  vector<Output> sess_run_fetchoutputs = {r_ngraph};
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropFilter",
                               static_input_indexes, output_datatypes,
                               sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter_ngraph.ExecuteOnNGraph(ngraph_outputs);

  // Define scope for tf (without nGraph)
  // Data Format: NHWC
  Scope tf_scope = Scope::NewRootScope();
  auto input_data_NHWC =
      ops::Transpose(tf_scope, input_data_NCHW, {0, 2, 3, 1});
  auto output_delta_NHWC =
      ops::Transpose(tf_scope, output_delta_NCHW, {0, 2, 3, 1});
  auto filter_sizes_tf = ops::Const(tf_scope, filter_size_HWIO);
  vector<int> stride_NHWC = {1, 2, 2, 1};
  auto r_tf =
      ops::Conv2DBackpropFilter(tf_scope, input_data_NHWC, filter_sizes_tf,
                                output_delta_NHWC, stride_NHWC, padding_type);
  vector<Output> sess_run_fetchoutputs_tf = {r_tf};
  OpExecuter opexecuter_tf(tf_scope, "Conv2DBackpropFilter",
                           static_input_indexes, output_datatypes,
                           sess_run_fetchoutputs_tf);

  vector<Tensor> tf_outputs;
  opexecuter_tf.ExecuteOnTF(tf_outputs);

  // Compare NGraph and TF Outputs
  Compare(tf_outputs, ngraph_outputs);
}

TEST(NNOps, Conv2DBackpropFilterNCHWValidWithDilation) {
  // NCHW
  vector<int64> input_size = {1, 2, 7, 6};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size = {3, 3, 2, 2};
  vector<int64> output_delta_size = {1, 2, 1, 1};
  vector<int> stride = {1, 1, 2, 2};
  string padding_type = "VALID";
  // change the dilation attribute
  ops::Conv2DBackpropFilter::Attrs op_attr;
  op_attr = op_attr.DataFormat("NCHW");
  op_attr = op_attr.Dilations({1, 1, 3, 2});

  vector<int> static_input_indexes = {1};

  Scope root = Scope::NewRootScope();

  Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
  int size_vect =
      std::accumulate(output_delta_size.begin(), output_delta_size.end(), 1,
                      std::multiplies<int64>());
  std::vector<float> output_vector(size_vect);
  std::iota(output_vector.begin(), output_vector.end(), 0);
  AssignInputValues<float>(output_delta, output_vector);

  auto filter_sizes = ops::Const(root, filter_size);

  Tensor input_data(DT_FLOAT, TensorShape(input_size));
  size_vect = std::accumulate(input_size.begin(), input_size.end(), 1,
                              std::multiplies<int64>());
  std::vector<float> input_vector(size_vect);
  std::iota(input_vector.begin(), input_vector.end(), 0);
  AssignInputValues<float>(input_data, input_vector);

  auto R =
      ops::Conv2DBackpropFilter(root, input_data, filter_sizes, output_delta,
                                stride, padding_type, op_attr);

  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter.ExecuteOnNGraph(ngraph_outputs);

  // Construct tf_outputs using gathered values
  vector<Tensor> tf_outputs;
  Tensor tf_result(DT_FLOAT, TensorShape({3, 3, 2, 2}));
  std::vector<float> tf_output_vector{
      0.f, 0.f,  0.f, 42.f, 0.f, 2.f,  0.f, 44.f, 0.f, 4.f,  0.f, 46.f,
      0.f, 18.f, 0.f, 60.f, 0.f, 20.f, 0.f, 62.f, 0.f, 22.f, 0.f, 64.f,
      0.f, 36.f, 0.f, 78.f, 0.f, 38.f, 0.f, 80.f, 0.f, 40.f, 0.f, 82.f};
  AssignInputValues(tf_result, tf_output_vector);
  tf_outputs.push_back(tf_result);

  Compare(tf_outputs, ngraph_outputs);
}  // end of Conv2DBackpropFilterNCHWValidWithDilation

TEST(NNOps, Conv2DBackpropFilterNHWCSame) {
  // TF Default formats : NHWC
  vector<int64> input_size = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size = {3, 3, 2, 2};
  vector<int> stride = {1, 2, 2, 1};
  string padding_type = "SAME";
  vector<int64> output_delta_size = {1, 4, 3, 2};
  vector<int> static_input_indexes = {1};

  Scope root = Scope::NewRootScope();

  Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
  AssignInputValuesRandom<float>(output_delta, -1.1f, 15.0f);

  auto filter_sizes = ops::Const(root, filter_size);

  Tensor input_data(DT_FLOAT, TensorShape(input_size));
  AssignInputValuesRandom<float>(input_data, -1.1f, 10.0f);

  auto R = ops::Conv2DBackpropFilter(root, input_data, filter_sizes,
                                     output_delta, stride, padding_type);

  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(NNOps, Conv2DBackpropFilterNHWCSameWithDilation) {
  // TF Default formats : NHWC
  vector<int64> input_size = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size = {3, 3, 2, 2};
  vector<int64> output_delta_size = {1, 4, 3, 2};
  vector<int> stride = {1, 2, 2, 1};
  string padding_type = "SAME";
  // change the dilation attribute
  ops::Conv2DBackpropFilter::Attrs op_attr;
  op_attr = op_attr.Dilations({1, 3, 2, 1});

  vector<int> static_input_indexes = {1};

  Scope root = Scope::NewRootScope();

  Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
  int size_vect =
      std::accumulate(output_delta_size.begin(), output_delta_size.end(), 1,
                      std::multiplies<int64>());
  std::vector<float> output_vector(size_vect);
  std::iota(output_vector.begin(), output_vector.end(), 0);
  AssignInputValues<float>(output_delta, output_vector);

  auto filter_sizes = ops::Const(root, filter_size);

  Tensor input_data(DT_FLOAT, TensorShape(input_size));
  size_vect = std::accumulate(input_size.begin(), input_size.end(), 1,
                              std::multiplies<int64>());
  std::vector<float> input_vector(size_vect);
  std::iota(input_vector.begin(), input_vector.end(), 0);
  AssignInputValues<float>(input_data, input_vector);

  auto R =
      ops::Conv2DBackpropFilter(root, input_data, filter_sizes, output_delta,
                                stride, padding_type, op_attr);

  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter.ExecuteOnNGraph(ngraph_outputs);

  // Construct tf_outputs using gathered values
  vector<Tensor> tf_outputs;
  Tensor tf_result(DT_FLOAT, TensorShape({3, 3, 2, 2}));
  std::vector<float> tf_output_vector{
      2168.f, 2280.f, 2240.f, 2356.f, 3308.f, 3488.f, 3410.f, 3596.f, 2200.f,
      2328.f, 2264.f, 2396.f, 5296.f, 5616.f, 5392.f, 5720.f, 7768.f, 8272.f,
      7900.f, 8416.f, 4976.f, 5328.f, 5056.f, 5416.f, 1400.f, 1608.f, 1424.f,
      1636.f, 1868.f, 2192.f, 1898.f, 2228.f, 1048.f, 1272.f, 1064.f, 1292.f};
  AssignInputValues(tf_result, tf_output_vector);
  tf_outputs.push_back(tf_result);

  Compare(tf_outputs, ngraph_outputs);
}  // end of Conv2DBackpropFilterNHWCSameWithDilation

TEST(NNOps, Conv2DBackpropFilterNHWCValid) {
  // TF Default formats : NHWC
  vector<int64> input_size = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size = {3, 3, 2, 2};
  vector<int64> output_delta_size = {1, 3, 2, 2};
  vector<int> stride = {1, 2, 2, 1};
  string padding_type = "VALID";

  vector<int> static_input_indexes = {1};

  Scope root = Scope::NewRootScope();

  Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
  AssignInputValuesRandom<float>(output_delta, -1.1f, 15.0f);

  auto filter_sizes = ops::Const(root, filter_size);

  Tensor input_data(DT_FLOAT, TensorShape(input_size));
  AssignInputValuesRandom<float>(input_data, -1.1f, 10.0f);

  auto R = ops::Conv2DBackpropFilter(root, input_data, filter_sizes,
                                     output_delta, stride, padding_type);

  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(NNOps, Conv2DBackpropFilterNHWCValidWithDilation) {
  // TF Default formats : NHWC
  vector<int64> input_size = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size = {3, 3, 2, 2};
  vector<int64> output_delta_size = {1, 1, 1, 2};
  vector<int> stride = {1, 2, 2, 1};
  string padding_type = "VALID";
  // change the dilation attribute
  ops::Conv2DBackpropFilter::Attrs op_attr;
  op_attr = op_attr.Dilations({1, 3, 2, 1});

  vector<int> static_input_indexes = {1};

  Scope root = Scope::NewRootScope();

  Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
  int size_vect =
      std::accumulate(output_delta_size.begin(), output_delta_size.end(), 1,
                      std::multiplies<int64>());
  std::vector<float> output_vector(size_vect);
  std::iota(output_vector.begin(), output_vector.end(), 0);
  AssignInputValues<float>(output_delta, output_vector);

  auto filter_sizes = ops::Const(root, filter_size);

  Tensor input_data(DT_FLOAT, TensorShape(input_size));
  size_vect = std::accumulate(input_size.begin(), input_size.end(), 1,
                              std::multiplies<int64>());
  std::vector<float> input_vector(size_vect);
  std::iota(input_vector.begin(), input_vector.end(), 0);
  AssignInputValues<float>(input_data, input_vector);

  auto R =
      ops::Conv2DBackpropFilter(root, input_data, filter_sizes, output_delta,
                                stride, padding_type, op_attr);

  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter.ExecuteOnNGraph(ngraph_outputs);

  // Construct tf_outputs using gathered values
  vector<Tensor> tf_outputs;
  Tensor tf_result(DT_FLOAT, TensorShape({3, 3, 2, 2}));
  std::vector<float> tf_output_vector{
      0.f, 0.f,  0.f, 1.f,  0.f, 4.f,  0.f, 5.f,  0.f, 8.f,  0.f, 9.f,
      0.f, 36.f, 0.f, 37.f, 0.f, 40.f, 0.f, 41.f, 0.f, 44.f, 0.f, 45.f,
      0.f, 72.f, 0.f, 73.f, 0.f, 76.f, 0.f, 77.f, 0.f, 80.f, 0.f, 81.f};
  AssignInputValues(tf_result, tf_output_vector);
  tf_outputs.push_back(tf_result);

  Compare(tf_outputs, ngraph_outputs);
}  // end of Conv2DBackpropFilterNHWCValidWithDilation

// Conv2DBackpropInput op : compute the graidents of conv with respects to input
// Input is in NCHW format, with padding type "SAME"
TEST(NNOps, Conv2DBackpropInputNCHWSame) {
  string padding_type = "SAME";
  initializer_list<int32> input_size_NCHW = {1, 2, 7, 6};
  initializer_list<int32> input_size_NHWC = {1, 7, 6, 2};

  // Filter :[filter_height, filter_width, in_channels, out_channels]
  vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  vector<int64> output_del_size_same_NCHW = {1, 2, 4, 3};
  std::vector<int> stride_NCHW = {1, 1, 2, 2};
  std::vector<int> stride_NHWC = {1, 2, 2, 1};
  // Conv2DBackpropInput has static input of index 0
  vector<int> static_input_indexes = {0};

  Scope ngraph_scope = Scope::NewRootScope();
  ops::Conv2DBackpropInput::Attrs op_attr_nchw;
  op_attr_nchw = op_attr_nchw.DataFormat("NCHW");

  auto input_data_NCHW = ops::Const(ngraph_scope, input_size_NCHW);

  Tensor output_delta(DT_FLOAT, TensorShape(output_del_size_same_NCHW));
  AssignInputValuesRandom<float>(output_delta, -10.0f, 15.0f);

  Tensor filter(DT_FLOAT, TensorShape(filter_size_HWIO));
  AssignInputValuesRandom<float>(filter, -1.1f, 10.0f);

  auto r_ngraph = ops::Conv2DBackpropInput(ngraph_scope, input_data_NCHW,
                                           filter, output_delta, stride_NCHW,
                                           padding_type, op_attr_nchw);

  vector<Output> sess_run_fetchoutputs = {r_ngraph};
  vector<DataType> output_datatypes = {DT_FLOAT};
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropInput",
                               static_input_indexes, output_datatypes,
                               sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter_ngraph.ExecuteOnNGraph(ngraph_outputs);

  // Define scope for tf (without nGraph)
  // Data Format: NHWC
  Scope tf_scope = Scope::NewRootScope();
  auto input_data_NHWC = ops::Const(tf_scope, input_size_NHWC);
  auto output_delta_NHWC = ops::Transpose(tf_scope, output_delta, {0, 2, 3, 1});

  auto r_tf =
      ops::Conv2DBackpropInput(tf_scope, input_data_NHWC, filter,
                               output_delta_NHWC, stride_NHWC, padding_type);

  // Need to transpose the TF output to NCHW
  auto tf_output_transposed = ops::Transpose(tf_scope, r_tf, {0, 3, 1, 2});
  vector<Output> sess_run_fetchoutputs_tf = {tf_output_transposed};
  OpExecuter opexecuter_tf(tf_scope, "Conv2DBackpropInput",
                           static_input_indexes, output_datatypes,
                           sess_run_fetchoutputs_tf);

  vector<Tensor> tf_outputs;
  opexecuter_tf.ExecuteOnTF(tf_outputs);

  // Compare NGraph and TF Outputs
  Compare(tf_outputs, ngraph_outputs);
}  // end of op Conv2DBackpropInputNCHWSame

// Conv2DBackpropInput op : compute the graidents of conv with respects to input
// Input is in NCHW format, padding type = "SAME, with non-trivial dilation
// attributes
TEST(NNOps, Conv2DBackpropInputNCHWSameWithDilation) {
  string padding_type = "SAME";
  initializer_list<int32> input_size_NCHW = {1, 2, 7, 6};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  vector<int64> output_del_size_same_NCHW = {1, 2, 4, 3};
  std::vector<int> stride_NCHW = {1, 1, 2, 2};

  // Conv2DBackpropInput has static input of index 0
  vector<int> static_input_indexes = {0};

  Scope ngraph_scope = Scope::NewRootScope();
  ops::Conv2DBackpropInput::Attrs op_attr_nchw;
  op_attr_nchw = op_attr_nchw.DataFormat("NCHW");
  op_attr_nchw = op_attr_nchw.Dilations({1, 1, 3, 2});

  auto input_data_NCHW = ops::Const(ngraph_scope, input_size_NCHW);

  Tensor output_delta(DT_FLOAT, TensorShape(output_del_size_same_NCHW));
  int size_vect = std::accumulate(output_del_size_same_NCHW.begin(),
                                  output_del_size_same_NCHW.end(), 1,
                                  std::multiplies<int64>());
  std::vector<float> output_vector(size_vect);
  std::iota(output_vector.begin(), output_vector.end(), 0);
  AssignInputValues<float>(output_delta, output_vector);

  Tensor filter(DT_FLOAT, TensorShape(filter_size_HWIO));
  size_vect = std::accumulate(filter_size_HWIO.begin(), filter_size_HWIO.end(),
                              1, std::multiplies<int64>());
  std::vector<float> filter_vector(size_vect);
  std::iota(filter_vector.begin(), filter_vector.end(), 0);
  AssignInputValues<float>(filter, filter_vector);

  auto r_ngraph = ops::Conv2DBackpropInput(ngraph_scope, input_data_NCHW,
                                           filter, output_delta, stride_NCHW,
                                           padding_type, op_attr_nchw);

  vector<Output> sess_run_fetchoutputs = {r_ngraph};
  vector<DataType> output_datatypes = {DT_FLOAT};
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropInput",
                               static_input_indexes, output_datatypes,
                               sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter_ngraph.ExecuteOnNGraph(ngraph_outputs);

  // Construct tf_outputs using gathered values
  vector<Tensor> tf_outputs;
  Tensor tf_result(DT_FLOAT, TensorShape({1, 2, 7, 6}));
  std::vector<float> tf_output_vector = {
      0.f, 385.f,  0.f, 695.f,  0.f, 563.f,  0.f, 133.f,  0.f, 353.f,
      0.f, 359.f,  0.f, 559.f,  0.f, 992.f,  0.f, 785.f,  0.f, 860.f,
      0.f, 1633.f, 0.f, 1360.f, 0.f, 733.f,  0.f, 1289.f, 0.f, 1007.f,
      0.f, 1015.f, 0.f, 1712.f, 0.f, 1289.f, 0.f, 907.f,  0.f, 1586.f,
      0.f, 1229.f, 0.f, 437.f,  0.f, 779.f,  0.f, 623.f,  0.f, 233.f,
      0.f, 509.f,  0.f, 467.f,  0.f, 635.f,  0.f, 1112.f, 0.f, 869.f,
      0.f, 1036.f, 0.f, 1909.f, 0.f, 1552.f, 0.f, 833.f,  0.f, 1445.f,
      0.f, 1115.f, 0.f, 1091.f, 0.f, 1832.f, 0.f, 1373.f, 0.f, 1031.f,
      0.f, 1778.f, 0.f, 1361.f};
  AssignInputValues(tf_result, tf_output_vector);
  tf_outputs.push_back(tf_result);

  // Compare NGraph and TF Outputs
  Compare(tf_outputs, ngraph_outputs);
}  // end of op Conv2DBackpropInputNCHWSameWithDilation

// Conv2DBackpropInput op : compute the graidents of conv with respects to input
// input is in the NCHW format, padding_type = "VALID"
TEST(NNOps, Conv2DBackpropInputNCHWValid) {
  string padding_type = "VALID";
  initializer_list<int32> input_size_NCHW = {1, 2, 7, 6};
  initializer_list<int32> input_size_NHWC = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  vector<int64> output_del_size_valid_NCHW = {1, 2, 3, 2};
  std::vector<int> stride_NCHW = {1, 1, 2, 2};
  std::vector<int> stride_NHWC = {1, 2, 2, 1};
  // Conv2DBackpropInput has static input of index 0
  vector<int> static_input_indexes = {0};

  Scope ngraph_scope = Scope::NewRootScope();
  ops::Conv2DBackpropInput::Attrs op_attr_nchw;
  op_attr_nchw = op_attr_nchw.DataFormat("NCHW");

  auto input_data_NCHW = ops::Const(ngraph_scope, input_size_NCHW);

  Tensor output_delta(DT_FLOAT, TensorShape(output_del_size_valid_NCHW));
  AssignInputValuesRandom<float>(output_delta, -10.0f, 15.0f);

  Tensor filter(DT_FLOAT, TensorShape(filter_size_HWIO));
  AssignInputValuesRandom<float>(filter, -1.1f, 10.0f);

  auto r_ngraph = ops::Conv2DBackpropInput(ngraph_scope, input_data_NCHW,
                                           filter, output_delta, stride_NCHW,
                                           padding_type, op_attr_nchw);

  vector<Output> sess_run_fetchoutputs = {r_ngraph};
  vector<DataType> output_datatypes = {DT_FLOAT};
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropInput",
                               static_input_indexes, output_datatypes,
                               sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter_ngraph.ExecuteOnNGraph(ngraph_outputs);

  // Define scope for tf (without nGraph)
  // Data Format: NHWC
  Scope tf_scope = Scope::NewRootScope();
  auto input_data_NHWC = ops::Const(tf_scope, input_size_NHWC);
  auto output_delta_NHWC = ops::Transpose(tf_scope, output_delta, {0, 2, 3, 1});

  auto r_tf =
      ops::Conv2DBackpropInput(tf_scope, input_data_NHWC, filter,
                               output_delta_NHWC, stride_NHWC, padding_type);

  // Need to transpose the TF output to NCHW
  auto tf_output_transposed = ops::Transpose(tf_scope, r_tf, {0, 3, 1, 2});
  vector<Output> sess_run_fetchoutputs_tf = {tf_output_transposed};
  OpExecuter opexecuter_tf(tf_scope, "Conv2DBackpropInput",
                           static_input_indexes, output_datatypes,
                           sess_run_fetchoutputs_tf);

  vector<Tensor> tf_outputs;
  opexecuter_tf.ExecuteOnTF(tf_outputs);

  // Compare NGraph and TF Outputs
  Compare(tf_outputs, ngraph_outputs);
}  // end of op Conv2DBackpropInputNCHWValid

// Conv2DBackpropInput op : compute the graidents of conv with respects to input
// Input is in NCHW format, padding="VALID" and with non-trivial dilation
// attributes
TEST(NNOps, Conv2DBackpropInputNCHWValidWithDilation) {
  string padding_type = "VALID";
  initializer_list<int32> input_size_NCHW = {1, 2, 7, 6};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  vector<int64> output_del_size_same_NCHW = {1, 2, 1, 1};
  std::vector<int> stride_NCHW = {1, 1, 2, 2};
  // Conv2DBackpropInput has static input of index 0
  vector<int> static_input_indexes = {0};

  Scope ngraph_scope = Scope::NewRootScope();
  ops::Conv2DBackpropInput::Attrs op_attr_nchw;
  op_attr_nchw = op_attr_nchw.DataFormat("NCHW");
  op_attr_nchw = op_attr_nchw.Dilations({1, 1, 3, 2});

  auto input_data_NCHW = ops::Const(ngraph_scope, input_size_NCHW);

  Tensor output_delta(DT_FLOAT, TensorShape(output_del_size_same_NCHW));
  int size_vect = std::accumulate(output_del_size_same_NCHW.begin(),
                                  output_del_size_same_NCHW.end(), 1,
                                  std::multiplies<int64>());
  std::vector<float> output_vector(size_vect);
  std::iota(output_vector.begin(), output_vector.end(), 0);
  AssignInputValues<float>(output_delta, output_vector);

  Tensor filter(DT_FLOAT, TensorShape(filter_size_HWIO));
  size_vect = std::accumulate(filter_size_HWIO.begin(), filter_size_HWIO.end(),
                              1, std::multiplies<int64>());
  std::vector<float> filter_vector(size_vect);
  std::iota(filter_vector.begin(), filter_vector.end(), 0);
  AssignInputValues<float>(filter, filter_vector);

  auto r_ngraph = ops::Conv2DBackpropInput(ngraph_scope, input_data_NCHW,
                                           filter, output_delta, stride_NCHW,
                                           padding_type, op_attr_nchw);

  vector<Output> sess_run_fetchoutputs = {r_ngraph};
  vector<DataType> output_datatypes = {DT_FLOAT};
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropInput",
                               static_input_indexes, output_datatypes,
                               sess_run_fetchoutputs);
  vector<Tensor> ngraph_outputs;
  opexecuter_ngraph.ExecuteOnNGraph(ngraph_outputs);

  // Construct tf_outputs using gathered values
  vector<Tensor> tf_outputs;
  Tensor tf_result(DT_FLOAT, TensorShape({1, 2, 7, 6}));
  std::vector<float> tf_output_vector{
      1.f,  0.f, 5.f,  0.f, 9.f,  0.f, 0.f,  0.f, 0.f,  0.f, 0.f,  0.f,
      0.f,  0.f, 0.f,  0.f, 0.f,  0.f, 13.f, 0.f, 17.f, 0.f, 21.f, 0.f,
      0.f,  0.f, 0.f,  0.f, 0.f,  0.f, 0.f,  0.f, 0.f,  0.f, 0.f,  0.f,
      25.f, 0.f, 29.f, 0.f, 33.f, 0.f, 3.f,  0.f, 7.f,  0.f, 11.f, 0.f,
      0.f,  0.f, 0.f,  0.f, 0.f,  0.f, 0.f,  0.f, 0.f,  0.f, 0.f,  0.f,
      15.f, 0.f, 19.f, 0.f, 23.f, 0.f, 0.f,  0.f, 0.f,  0.f, 0.f,  0.f,
      0.f,  0.f, 0.f,  0.f, 0.f,  0.f, 27.f, 0.f, 31.f, 0.f, 35.f, 0.f};
  AssignInputValues(tf_result, tf_output_vector);
  tf_outputs.push_back(tf_result);

  // Compare NGraph and TF Outputs
  Compare(tf_outputs, ngraph_outputs);
}  // end of op Conv2DBackpropInputNCHWValidWithDilation

// Conv2DBackpropInput op : compute the graidents of conv with respects to input
// Test case for TF default data format: NHWC
TEST(NNOps, Conv2DBackpropInputNHWC) {
  std::initializer_list<int> input_size_NHWC = {1, 7, 6, 2};

  // Filter :[filter_height, filter_width, in_channels, out_channels]
  vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  vector<int64> output_del_size_valid = {1, 3, 2, 2};
  vector<int64> output_del_size_same = {1, 4, 3, 2};

  std::vector<int> stride = {1, 2, 2, 1};

  std::map<std::string, vector<int64>> out_delta_size_map = {
      {"VALID", output_del_size_valid}, {"SAME", output_del_size_same}};

  // Conv2DBackpropInput has static input of index 0
  vector<int> static_input_indexes = {0};

  for (auto map_iterator : out_delta_size_map) {
    Scope root = Scope::NewRootScope();
    auto padding_type = map_iterator.first;
    auto output_delta_size = out_delta_size_map[padding_type];

    auto input_sizes = ops::Const(root, input_size_NHWC);

    Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
    AssignInputValuesRandom<float>(output_delta, -10.0f, 15.0f);

    Tensor filter(DT_FLOAT, TensorShape(filter_size_HWIO));
    AssignInputValuesRandom<float>(filter, -1.1f, 10.0f);

    auto R = ops::Conv2DBackpropInput(root, input_sizes, filter, output_delta,
                                      stride, padding_type);

    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Conv2DBackpropInput", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op Conv2DBackpropInputNHWC

// Test for Conv2DBackpropInput NWHC case with non-trivial dilation parameter
TEST(NNOps, Conv2DBackpropInputNHWCWithDilation) {
  std::initializer_list<int> input_size_NHWC = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  vector<int64> output_del_size_valid = {1, 1, 1, 2};
  vector<int64> output_del_size_same = {1, 4, 3, 2};
  std::vector<int> stride = {1, 2, 2, 1};

  // TF GPU results gathered running with same values of parameters
  std::map<std::string, std::vector<float>> tf_output_map = {
      {"VALID",
       {1.f,  3.f,  0.f, 0.f, 5.f,  7.f,  0.f, 0.f, 9.f,  11.f, 0.f, 0.f,
        0.f,  0.f,  0.f, 0.f, 0.f,  0.f,  0.f, 0.f, 0.f,  0.f,  0.f, 0.f,
        0.f,  0.f,  0.f, 0.f, 0.f,  0.f,  0.f, 0.f, 0.f,  0.f,  0.f, 0.f,
        13.f, 15.f, 0.f, 0.f, 17.f, 19.f, 0.f, 0.f, 21.f, 23.f, 0.f, 0.f,
        0.f,  0.f,  0.f, 0.f, 0.f,  0.f,  0.f, 0.f, 0.f,  0.f,  0.f, 0.f,
        0.f,  0.f,  0.f, 0.f, 0.f,  0.f,  0.f, 0.f, 0.f,  0.f,  0.f, 0.f,
        25.f, 27.f, 0.f, 0.f, 29.f, 31.f, 0.f, 0.f, 33.f, 35.f, 0.f, 0.f}},
      {"SAME",
       {0.f,    0.f,    80.f,   92.f,   0.f,    0.f,    217.f,  247.f,  0.f,
        0.f,    252.f,  280.f,  0.f,    0.f,    128.f,  236.f,  0.f,    0.f,
        361.f,  535.f,  0.f,    0.f,    396.f,  520.f,  0.f,    0.f,    428.f,
        488.f,  0.f,    0.f,    811.f,  913.f,  0.f,    0.f,    696.f,  772.f,
        0.f,    0.f,    340.f,  508.f,  0.f,    0.f,    920.f,  1196.f, 0.f,
        0.f,    972.f,  1172.f, 0.f,    0.f,    776.f,  884.f,  0.f,    0.f,
        1405.f, 1579.f, 0.f,    0.f,    1140.f, 1264.f, 0.f,    0.f,    788.f,
        848.f,  0.f,    0.f,    1423.f, 1525.f, 0.f,    0.f,    1152.f, 1228.f,
        0.f,    0.f,    1124.f, 1280.f, 0.f,    0.f,    1999.f, 2245.f, 0.f,
        0.f,    1584.f, 1756.f}}};
  std::map<std::string, vector<int64>> out_delta_size_map = {
      {"VALID", output_del_size_valid}, {"SAME", output_del_size_same}};

  // Conv2DBackpropInput has static input of index 0
  vector<int> static_input_indexes = {0};
  // changet the dilation attribute
  ops::Conv2DBackpropInput::Attrs op_attr;
  op_attr = op_attr.Dilations({1, 3, 2, 1});

  for (auto map_iterator : out_delta_size_map) {
    Scope root = Scope::NewRootScope();
    auto padding_type = map_iterator.first;
    auto output_delta_size = out_delta_size_map[padding_type];

    auto input_sizes = ops::Const(root, input_size_NHWC);

    Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
    int size_vect =
        std::accumulate(output_delta_size.begin(), output_delta_size.end(), 1,
                        std::multiplies<int64>());
    std::vector<float> output_vector(size_vect);
    std::iota(output_vector.begin(), output_vector.end(), 0);
    AssignInputValues<float>(output_delta, output_vector);

    Tensor filter(DT_FLOAT, TensorShape(filter_size_HWIO));
    size_vect =
        std::accumulate(filter_size_HWIO.begin(), filter_size_HWIO.end(), 1,
                        std::multiplies<int64>());
    std::vector<float> filter_vector(size_vect);
    std::iota(filter_vector.begin(), filter_vector.end(), 0);
    AssignInputValues<float>(filter, filter_vector);

    auto R = ops::Conv2DBackpropInput(root, input_sizes, filter, output_delta,
                                      stride, padding_type, op_attr);

    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Conv2DBackpropInput", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);
    vector<Tensor> ngraph_outputs;
    opexecuter.ExecuteOnNGraph(ngraph_outputs);

    // Construct tf_outputs using gathered values
    vector<Tensor> tf_outputs;
    Tensor tf_result(DT_FLOAT, TensorShape({1, 7, 6, 2}));
    auto tf_output_vector = tf_output_map[padding_type];
    AssignInputValues(tf_result, tf_output_vector);
    tf_outputs.push_back(tf_result);

    Compare(tf_outputs, ngraph_outputs);
  }
}  // end of op Conv2DBackpropInputNHWCWithDilation

// FusedBatchNormGrad : Gradient for batch normalization
// On TF CPU: only supports NHWC
TEST(NNOps, FusedBatchNormGradNHWC) {
  Scope root = Scope::NewRootScope();

  // 4D tensor for the gradient with respect to y
  Tensor y_backprop(DT_FLOAT, TensorShape({5, 4, 3, 2}));
  // 4D tensor for input data
  Tensor x(DT_FLOAT, TensorShape({5, 4, 3, 2}));
  // 1D tensor for scaling the normalized x
  Tensor scale(DT_FLOAT, TensorShape({2}));
  // 1D tensor for population mean
  Tensor reserve_space_1_mean(DT_FLOAT, TensorShape({2}));
  // 1D tensor for population variance
  Tensor reserve_space_2_variance(DT_FLOAT, TensorShape({2}));

  AssignInputValuesRandom<float>(y_backprop, -5.0f, 10.0f);
  AssignInputValuesRandom<float>(x, -10.0f, 10.0f);
  AssignInputValuesRandom<float>(scale, -1.6f, 1.6f);
  AssignInputValuesRandom<float>(reserve_space_1_mean, 1.1f, 1.5f);
  AssignInputValuesRandom<float>(reserve_space_2_variance, 0.5f, 1.5f);

  auto attrs = ops::FusedBatchNormGrad::Attrs();
  attrs.is_training_ =
      true;  // doesn't support is_training_= false case on ngraph
  attrs.epsilon_ = 0.0001f;
  attrs.data_format_ = "NHWC";

  // test grab the first three outputs from the FusedBatchNormGrad op
  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  auto R =
      ops::FusedBatchNormGrad(root, y_backprop, x, scale, reserve_space_1_mean,
                              reserve_space_2_variance, attrs);
  std::vector<Output> sess_run_fetchoutputs = {R.x_backprop, R.scale_backprop,
                                               R.offset_backprop};
  OpExecuter opexecuter(root, "FusedBatchNormGrad", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);
  opexecuter.RunTest();

  // test grab all the outputs from the FusedBatchNormGrad op
  Scope all_output_test = Scope::NewRootScope();
  vector<DataType> output_datatypes_all = {DT_FLOAT, DT_FLOAT, DT_FLOAT,
                                           DT_FLOAT, DT_FLOAT};
  R = ops::FusedBatchNormGrad(all_output_test, y_backprop, x, scale,
                              reserve_space_1_mean, reserve_space_2_variance,
                              attrs);
  std::vector<Output> sess_run_fetchoutputs_all = {
      R.x_backprop, R.scale_backprop, R.offset_backprop, R.reserve_space_3,
      R.reserve_space_4};
  OpExecuter opexecuter_all_output(all_output_test, "FusedBatchNormGrad",
                                   static_input_indexes, output_datatypes_all,
                                   sess_run_fetchoutputs_all);
  opexecuter_all_output.RunTest();
}

// Test Op :"L2Loss"
TEST(NNOps, L2Loss) {
  std::vector<std::vector<int64>> input_sizes;
  input_sizes.push_back({2, 3, 4});
  input_sizes.push_back({0});

  vector<int> static_input_indexes = {};

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -10, 10);

    auto R = ops::L2Loss(root, input_data);
    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "L2Loss", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

// Note: TF only supports QUINT8 for QMP in CPU
// Source:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantized_pooling_ops.cc#L127
// Computes Quantized Maxpool
TEST(NNOps, QuantizedMaxPool) {
  int dim1 = 2;
  int dim2 = 3;
  int channels = 2;

  for (int windowsize1 = 1; windowsize1 < 3; windowsize1++) {
    for (int windowsize2 = 1; windowsize2 < 3; windowsize2++) {
      for (int stride1 = 1; stride1 < 3; stride1++) {
        for (int stride2 = 1; stride2 < 3; stride2++) {
          for (auto padding_mode : {"SAME", "VALID"}) {
            Scope root = Scope::NewRootScope();
            auto quant_type = DT_QUINT8;
            Tensor A(quant_type, TensorShape({1, dim1, dim2, channels}));
            AssignInputValues<quint8>(
                A, {50, 242, 14, 0, 17, 22, 100, 250, 34, 60, 79, 255});
            vector<int> ksize = {1, windowsize1, windowsize2, 1};
            vector<int> strides = {1, stride1, stride2, 1};

            vector<int> static_input_indexes = {1, 2};
            auto R = ops::QuantizedMaxPool(root, A, -10.0f, 10.99f, ksize,
                                           strides, padding_mode);

            vector<DataType> output_datatypes = {quant_type, DT_FLOAT,
                                                 DT_FLOAT};

            std::vector<Output> sess_run_fetchoutputs = {R.output, R.min_output,
                                                         R.max_output};
            OpExecuter opexecuter(root, "QuantizedMaxPool",
                                  static_input_indexes, output_datatypes,
                                  sess_run_fetchoutputs);

            opexecuter.RunTest();
          }
        }
      }
    }
  }
}
// TODO: add a quantized maxpool test, where min-max are equal or close together

// Computes softmax cross entropy cost and gradients to backpropagate.
TEST(NNOps, SparseSoftmaxCrossEntropyWithLogits) {
  Scope root = Scope::NewRootScope();
  int batch = 10;
  int num_of_classes = 2;

  Tensor A(DT_FLOAT, TensorShape({batch, num_of_classes}));
  Tensor B(DT_INT32, TensorShape({batch}));

  AssignInputValuesRandom<float>(A, -2.0f, 2.0f);
  AssignInputValuesRandom<int>(B, 0, num_of_classes - 1);

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
