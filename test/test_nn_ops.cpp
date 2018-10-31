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

  // Dialtion rates > 1 not supported by TF
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

  // Dialtion rates > 1 not supported by TF
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

TEST(NNOps, Conv2DBackpropFilterNHWCSame) {
  // TF Default formats
  // Input NHWC :[batch, in_height, in_width, in_channels]
  vector<int64> input_size = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size = {3, 3, 2, 2};
  // Out_delta :[batch, out_height, out_width, out_channels]
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

TEST(NNOps, Conv2DBackpropFilterNHWCValid) {
  // TF Default formats
  // Input NHWC :[batch, in_height, in_width, in_channels]
  vector<int64> input_size = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  initializer_list<int> filter_size = {3, 3, 2, 2};
  // Out_delta :[batch, out_height, out_width, out_channels]
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

// NCHW data format not supported on the test framework now
TEST(NNOps, DISABLED_Conv2DBackpropFilterNCHW) {
  // Input NCHW :[batch, in_height, in_width, in_channels]
  vector<int64> input_size_NCHW = {1, 2, 7, 6};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  vector<int64> filter_size_HWIO = {3, 3, 2, 2};

  // Out_delta :[batch, out_height, out_width, out_channels]
  vector<int64> output_del_size_valid = {1, 3, 2, 2};
  vector<int64> output_del_size_same = {1, 4, 3, 2};
  std::vector<int> stride = {1, 2, 2, 1};

  std::map<std::string, vector<int64>> out_delta_size_map = {
      {"VALID", output_del_size_valid}, {"SAME", output_del_size_same}};

  vector<int> static_input_indexes = {1};
  auto attrs = ops::Conv2DBackpropFilter::Attrs();
  attrs.data_format_ = "NCHW";

  // TEST NCHW
  for (auto map_iterator : out_delta_size_map) {
    Scope root = Scope::NewRootScope();
    auto padding_type = map_iterator.first;
    auto output_delta_size = out_delta_size_map[padding_type];

    Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
    AssignInputValues(output_delta, -1.1f);

    auto filter_sizes = ops::Const(root, {3, 3, 2, 2});

    Tensor input_data(DT_FLOAT, TensorShape(input_size_NCHW));
    AssignInputValues(input_data, -1.1f);

    auto R =
        ops::Conv2DBackpropFilter(root, input_data, filter_sizes, output_delta,
                                  stride, padding_type, attrs);

    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

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
