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
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/opexecuter.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Test(TestCaseName, TestName)
// Please ensure
// Neither TestCaseName nor TestName should contain underscore
// https://github.com/google/googletest/blob/master/googletest/docs/primer.md
// Use only Tensors and ops::Const() to provide input to the test op

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
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropInput",
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
                           sess_run_fetchoutputs_tf);

  vector<Tensor> tf_outputs;
  opexecuter_tf.ExecuteOnTF(tf_outputs);

  // Compare NGraph and TF Outputs
  Compare(tf_outputs, ngraph_outputs, 1e-05, 1e-05);
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
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropInput",
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
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropInput",
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
                           sess_run_fetchoutputs_tf);

  vector<Tensor> tf_outputs;
  opexecuter_tf.ExecuteOnTF(tf_outputs);

  // Compare NGraph and TF Outputs
  Compare(tf_outputs, ngraph_outputs, 1e-05, 1e-05);
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
  OpExecuter opexecuter_ngraph(ngraph_scope, "Conv2DBackpropInput",
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

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Conv2DBackpropInput", sess_run_fetchoutputs);

    opexecuter.RunTest(1e-05, 1e-05);
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

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Conv2DBackpropInput", sess_run_fetchoutputs);
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

// Conv3D Op Tests

TEST(NNOps, Conv3DNDHWCSame) {
  vector<int64> input_size_NDHWC = {1, 5, 6, 7, 10};
  Tensor input_data_NDHWC(DT_FLOAT, TensorShape(input_size_NDHWC));
  AssignInputValuesRandom<float>(input_data_NDHWC, -15.0f, 15.0f);

  // Filter :[filter_depth, filter_height, filter_width, in_channels,
  // out_channels]
  vector<int64> filter_size_DHWIO = {3, 3, 3, 10, 2};

  std::vector<int> stride = {1, 2, 2, 2, 1};

  // Dilation rates > 1 not supported by TF on CPU
  ops::Conv3D::Attrs op_attr_ndhwc;
  op_attr_ndhwc = op_attr_ndhwc.DataFormat("NDHWC");
  op_attr_ndhwc = op_attr_ndhwc.Dilations({1, 1, 1, 1, 1});

  Scope root = Scope::NewRootScope();
  string padding_type = "SAME";

  Tensor filter(DT_FLOAT, TensorShape(filter_size_DHWIO));
  AssignInputValuesRandom<float>(filter, -1.1f, 10.0f);

  auto R = ops::Conv3D(root, input_data_NDHWC, filter, stride, padding_type,
                       op_attr_ndhwc);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Conv3D", sess_run_fetchoutputs);

  opexecuter.RunTest(1e-03, 1e-03);
}

// FusedBatchNormV2 op test with only DT_FLOAT datatype
TEST(NNOps, FusedBatchNormV2NHWCInference) {
  Scope root = Scope::NewRootScope();

  // 4D tensor for input data
  Tensor x(DT_FLOAT, TensorShape({10, 128, 128, 3}));
  // 1D tensor for scaling the normalized x
  Tensor scale(DT_FLOAT, TensorShape({3}));
  // 1D tensor for offset, to shift to the normalized x
  Tensor offset(DT_FLOAT, TensorShape({3}));
  // 1D tensor for population mean
  // used for inference only, must be empty for training
  Tensor mean(DT_FLOAT, TensorShape({3}));
  // 1D tensor for population variance
  // used for inference only, must be empty for training
  Tensor variance(DT_FLOAT, TensorShape({3}));

  AssignInputValuesRandom<float>(x, -30.0f, 50.0f);
  AssignInputValuesRandom<float>(scale, 0.2f, 3.f);
  AssignInputValuesRandom<float>(offset, 1.1f, 1.5f);
  AssignInputValuesRandom<float>(mean, -3.5f, 3.5f);
  AssignInputValuesRandom<float>(variance, 0.5f, 3.5f);

  auto attrs = ops::FusedBatchNormV2::Attrs();
  attrs.is_training_ = false;
  attrs.epsilon_ = 0.0001f;
  attrs.data_format_ = "NHWC";

  auto R = ops::FusedBatchNormV2(root, x, scale, offset, mean, variance, attrs);

  // In inference case, y is the only output tensor
  std::vector<Output> sess_run_fetchoutputs = {R.y};
  OpExecuter opexecuter(root, "FusedBatchNormV2", sess_run_fetchoutputs);

  opexecuter.RunTest(1e-05, 1e-06);
}  // end of FusedBatchNormV2NHWCInference

// FusedBatchNormV3 op test with only DT_FLOAT datatype
TEST(NNOps, FusedBatchNormV3NHWCInference) {
  Scope root = Scope::NewRootScope();

  // 4D tensor for input data
  Tensor x(DT_FLOAT, TensorShape({10, 128, 128, 3}));
  // 1D tensor for scaling the normalized x
  Tensor scale(DT_FLOAT, TensorShape({3}));
  // 1D tensor for offset, to shift to the normalized x
  Tensor offset(DT_FLOAT, TensorShape({3}));
  // 1D tensor for population mean
  // used for inference only, must be empty for training
  Tensor mean(DT_FLOAT, TensorShape({3}));
  // 1D tensor for population variance
  // used for inference only, must be empty for training
  Tensor variance(DT_FLOAT, TensorShape({3}));

  AssignInputValuesRandom<float>(x, -30.0f, 50.0f);
  AssignInputValuesRandom<float>(scale, 0.2f, 3.f);
  AssignInputValuesRandom<float>(offset, 1.1f, 1.5f);
  AssignInputValuesRandom<float>(mean, -3.5f, 3.5f);
  AssignInputValuesRandom<float>(variance, 0.5f, 3.5f);

  auto attrs = ops::FusedBatchNormV3::Attrs();
  attrs.is_training_ = false;
  attrs.epsilon_ = 0.0001f;
  attrs.data_format_ = "NHWC";

  auto R = ops::FusedBatchNormV3(root, x, scale, offset, mean, variance, attrs);

  // In inference case, y is the only output tensor
  std::vector<Output> sess_run_fetchoutputs = {R.y};
  OpExecuter opexecuter(root, "FusedBatchNormV3", sess_run_fetchoutputs);

  opexecuter.RunTest(1e-05, 1e-06);
}  // end of FusedBatchNormV3NHWCInference

// Test Op :"L2Loss"
TEST(NNOps, L2Loss) {
  std::vector<std::vector<int64>> input_sizes;
  input_sizes.push_back({2, 3, 4});
  input_sizes.push_back({0});

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -10, 10);

    auto R = ops::L2Loss(root, input_data);
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "L2Loss", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

// Test Op :"LogSoftmax"
TEST(NNOps, LogSoftmax) {
  std::vector<std::vector<int64>> input_sizes = {
      {3}, {3, 2}, {5, 6}, {3, 4, 5}, {2, 3, 4, 5}};

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -2, 2);

    auto R = ops::LogSoftmax(root, input_data);
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "LogSoftmax", sess_run_fetchoutputs);
    float rtol = static_cast<float>(1e-05);
    float atol = static_cast<float>(1e-05);
    opexecuter.RunTest(rtol, atol);
  }
}

// Test Op :"LRN"
TEST(NNOps, LRN) {
  Scope root = Scope::NewRootScope();

  Tensor A(DT_FLOAT, TensorShape({2, 2, 2, 2}));

  AssignInputValues(A, 2.1f);

  auto R = ops::LRN(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "LRN", sess_run_fetchoutputs);
  opexecuter.RunTest();
}

TEST(NNOps, LRNattr) {
  Scope root = Scope::NewRootScope();

  Tensor A(DT_FLOAT, TensorShape({12, 13, 1, 14}));
  AssignInputValuesRandom<float>(A, -1, 9);
  auto attrs = ops::LRN::Attrs();
  attrs.alpha_ = 0.3222;
  attrs.beta_ = 0.6875;
  attrs.bias_ = 1.0059;
  attrs.depth_radius_ = 1;

  auto R = ops::LRN(root, A, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "LRN", sess_run_fetchoutputs);
  float rtol = static_cast<float>(1e-02);
  float atol = static_cast<float>(1e-02);
  opexecuter.RunTest(rtol, atol);
}

// Test Op :"MaxPool3D"
TEST(NNOps, MaxPool3DNDHWCSame) {
  std::vector<std::vector<int64>> input_sizes;
  input_sizes.push_back({2, 3, 4, 4, 3});
  input_sizes.push_back({10, 30, 15, 20, 3});

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -10, 10);

    vector<int> filter = {1, 2, 2, 2, 1};
    vector<int> stride = {1, 1, 1, 1, 1};

    auto R = ops::MaxPool3D(root, input_data, filter, stride, "SAME");
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "MaxPool3D", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of MaxPool3DNDHWCSame op

// Test Op :"MaxPool3D"
TEST(NNOps, MaxPool3DNDHWCValid) {
  std::vector<std::vector<int64>> input_sizes;
  input_sizes.push_back({2, 3, 4, 4, 3});
  input_sizes.push_back({10, 30, 15, 20, 3});

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -10, 10);

    vector<int> filter = {1, 2, 2, 2, 1};
    vector<int> stride = {1, 1, 1, 1, 1};

    auto R = ops::MaxPool3D(root, input_data, filter, stride, "VALID");
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "MaxPool3D", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of MaxPool3DNDHWCValid op

// Softmax on 2D tensor
TEST(NNOps, Softmax2D) {
  Scope root = Scope::NewRootScope();
  int batch = 10;
  int num_of_classes = 2;

  Tensor A(DT_FLOAT, TensorShape({batch, num_of_classes}));
  AssignInputValuesRandom<float>(A, -2.0f, 2.0f);

  auto R = ops::Softmax(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Softmax", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// Softmax on 3D tensor
TEST(NNOps, Softmax3D) {
  Scope root = Scope::NewRootScope();
  Tensor A(DT_FLOAT, TensorShape({2, 3, 4}));

  AssignInputValuesRandom<float>(A, -2.0f, 2.0f);
  auto R = ops::Softmax(root, A);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Softmax", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// The non softmax (non last) dim is zero
TEST(NNOps, SoftmaxZeroDimTest1) {
  Scope root = Scope::NewRootScope();

  Tensor A(DT_FLOAT, TensorShape({3, 0, 2}));

  AssignInputValuesRandom<float>(A, -2.0f, 2.0f);
  auto R = ops::Softmax(root, A);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Softmax", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// The softmax (last) dim is zero
TEST(NNOps, SoftmaxZeroDimTest2) {
  Scope root = Scope::NewRootScope();

  Tensor A(DT_FLOAT, TensorShape({3, 2, 0}));

  AssignInputValuesRandom<float>(A, -2.0f, 2.0f);
  auto R = ops::Softmax(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Softmax", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// Test Op :"Softplus"
TEST(NNOps, Softplus) {
  std::vector<std::vector<int64>> input_sizes = {
      {3}, {3, 2}, {5, 6}, {3, 4, 5}, {2, 3, 4, 5}};

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -2, 2);

    auto R = ops::Softplus(root, input_data);
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Softplus", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

// Test Op :"BiasAdd", also see ./python/test_biasadd.py
// Run .../ngraph-bridge/build_cmake/test$ ./gtest_ngtf
// --gtest_filter="NNOps.BiasAdd"
TEST(NNOps, BiasAdd) {
  {
    Tensor A(DT_FLOAT, TensorShape({2, 2, 2, 2}));
    AssignInputValues<float>(A,
                             {0, 1, 0, 1, 2, 1, 1, 0, 3, 1, 1, 0, 4, 4, 5, 4});
    Tensor B(DT_FLOAT, TensorShape({2}));
    AssignInputValues<float>(B, {100, -100});
    ops::BiasAdd::Attrs attrs;
    std::vector<std::string> formats{"NHWC", "NCHW"};

    for (auto& format : formats) {
      NGRAPH_VLOG(2) << "BiasAdd testing with format: " << format;
      Scope root = Scope::NewRootScope();
      attrs = attrs.DataFormat(format);
      // see TF file .../tensorflow/cc/ops/nn_ops.h
      auto R = ops::BiasAdd(root, A, B, attrs);
      std::vector<Output> sess_run_fetchoutputs = {R};
      OpExecuter opexecuter(root, "BiasAdd", sess_run_fetchoutputs);
      opexecuter.RunTest();
    }
  }

  {
    Tensor A(DT_FLOAT, TensorShape({2, 3, 2, 2}));  // NCHW
    AssignInputValues<float>(A, {0, 1, 0, 1, 2, 1, 1, 0, 3, 1, 1, 0,
                                 4, 4, 5, 4, 3, 5, 1, 2, 0, 4, 0, 1});
    Tensor B(DT_FLOAT, TensorShape({3}));
    AssignInputValues<float>(B, {100, -100, 50});  // channels = 3
    ops::BiasAdd::Attrs attrs;
    std::string format("NCHW");
    NGRAPH_VLOG(2) << "BiasAdd testing with format: " << format;
    Scope root = Scope::NewRootScope();
    attrs = attrs.DataFormat(format);
    // see TF file .../tensorflow/cc/ops/nn_ops.h
    auto R = ops::BiasAdd(root, A, B, attrs);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "BiasAdd", sess_run_fetchoutputs);
    opexecuter.RunTest();
  }

  {
    Tensor A(DT_FLOAT, TensorShape({2, 2, 2, 3}));  // NHWC
    AssignInputValues<float>(A, {0, 1, 0, 1, 2, 1, 1, 0, 3, 1, 1, 0,
                                 4, 4, 5, 4, 3, 5, 1, 2, 0, 4, 0, 1});
    Tensor B(DT_FLOAT, TensorShape({3}));
    AssignInputValues<float>(B, {100, -100, 50});  // channels = 3
    ops::BiasAdd::Attrs attrs;
    std::string format("NHWC");
    NGRAPH_VLOG(2) << "BiasAdd testing with format: " << format;
    Scope root = Scope::NewRootScope();
    attrs = attrs.DataFormat(format);
    // see TF file .../tensorflow/cc/ops/nn_ops.h
    auto R = ops::BiasAdd(root, A, B, attrs);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "BiasAdd", sess_run_fetchoutputs);
    opexecuter.RunTest();
  }
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
