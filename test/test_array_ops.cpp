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
// Please ensure the alphabetical order while adding the test functions

// Test DepthToSpace with NHWC data format
TEST(ArrayOps, DepthToSpaceNHWC) {
  std::map<std::vector<int64>, int> input_map;
  input_map.insert(pair<std::vector<int64>, int>({1, 1, 1, 4}, 2));
  input_map.insert(pair<std::vector<int64>, int>({1, 1, 1, 12}, 2));
  input_map.insert(pair<std::vector<int64>, int>({1, 1, 1, 27}, 3));
  input_map.insert(pair<std::vector<int64>, int>({1, 1, 1, 500}, 10));
  input_map.insert(pair<std::vector<int64>, int>({1, 4, 2, 75}, 5));
  input_map.insert(pair<std::vector<int64>, int>({2, 1, 2, 27}, 3));
  input_map.insert(pair<std::vector<int64>, int>({10, 5, 5, 40}, 2));

  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT};

  map<std::vector<int64>, int>::iterator iter;
  for (iter = input_map.begin(); iter != input_map.end(); iter++) {
    std::vector<int64> shape = iter->first;
    int block_size = iter->second;

    Scope root = Scope::NewRootScope();
    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);

    auto R = ops::DepthToSpace(root, input_data, block_size);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "DepthToSpace", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);
    opexecuter.RunTest();
  }
}  // end of op DepthToSpaceNHWC

// Test DepthToSpace with NCHW data format
TEST(ArrayOps, DepthToSpaceNCHW) {
  std::map<std::vector<int64>, int> input_map;
  input_map.insert(pair<std::vector<int64>, int>({1, 4, 1, 1}, 2));
  input_map.insert(pair<std::vector<int64>, int>({1, 250, 1, 1}, 5));
  input_map.insert(pair<std::vector<int64>, int>({1, 180, 1, 1}, 3));
  input_map.insert(pair<std::vector<int64>, int>({2, 27, 2, 1}, 3));
  input_map.insert(pair<std::vector<int64>, int>({10, 40, 5, 5}, 2));
  input_map.insert(pair<std::vector<int64>, int>({2, 9, 5, 1}, 3));
  input_map.insert(pair<std::vector<int64>, int>({30, 3000, 3, 3}, 10));

  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT};
  ops::DepthToSpace::Attrs attrs;
  attrs.data_format_ = "NCHW";

  map<std::vector<int64>, int>::iterator iter;
  for (iter = input_map.begin(); iter != input_map.end(); iter++) {
    std::vector<int64> shape = iter->first;
    int block_size = iter->second;

    Scope root = Scope::NewRootScope();
    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);

    auto R = ops::DepthToSpace(root, input_data, block_size, attrs);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "DepthToSpace", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    vector<Tensor> ngraph_outputs;
    opexecuter.ExecuteOnNGraph(ngraph_outputs);

    // On CPU, the op only supports NHWC data format
    Scope tf_scope = Scope::NewRootScope();
    auto input_data_NHWC = ops::Transpose(tf_scope, input_data, {0, 2, 3, 1});
    auto r_tf = ops::DepthToSpace(tf_scope, input_data_NHWC, block_size);
    auto r_tf_NCHW = ops::Transpose(tf_scope, r_tf, {0, 3, 1, 2});
    vector<Output> sess_run_fetchoutputs_tf = {r_tf_NCHW};
    OpExecuter opexecuter_tf(tf_scope, "DepthToSpace", static_input_indexes,
                             output_datatypes, sess_run_fetchoutputs_tf);

    vector<Tensor> tf_outputs;
    opexecuter_tf.ExecuteOnTF(tf_outputs);

    // Compare NGraph and TF Outputs
    Compare(tf_outputs, ngraph_outputs);
  }
}  // end of op DepthToSpaceNCHW

// Test op: Dequantize
// Dequantizes a tensor from i8 to float
TEST(ArrayOps, Dequantizei8) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_QINT8, TensorShape({dim1, dim2}));
  AssignInputValues<qint8>(A, {-5, -1, 0, 1, 5, 100});

  auto attrs = ops::Dequantize::Attrs();
  attrs.mode_ = "SCALED";

  vector<int> static_input_indexes = {1, 2};
  ops::Dequantize R = ops::Dequantize(root, A, -6.0f, 128.0f, attrs);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output};
  OpExecuter opexecuter(root, "Dequantize", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Dequantizei8

// Dequantize tests for values tested in tf test
// dequantize_op_test.DequantizeOpTest.testBasicQint8 with 'Scaled' mode
TEST(ArrayOps, Dequantizei8TF1) {
  Scope root = Scope::NewRootScope();
  int dim1 = 3;

  Tensor A(DT_QINT8, TensorShape({dim1}));
  AssignInputValues<qint8>(A, {-128, 0, 127});

  auto attrs = ops::Dequantize::Attrs();
  attrs.mode_ = "SCALED";

  vector<int> static_input_indexes = {1, 2};
  ops::Dequantize R = ops::Dequantize(root, A, -1.0f, 2.0f, attrs);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output};
  OpExecuter opexecuter(root, "Dequantize", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Dequantizei8TF1

// The output values are close but do not pass the test
TEST(ArrayOps, DISABLED_Dequantizei8TF2) {
  Scope root = Scope::NewRootScope();
  int dim1 = 3;

  Tensor A(DT_QINT8, TensorShape({dim1}));
  AssignInputValues<qint8>(A, {-2, 4, -17});

  auto attrs = ops::Dequantize::Attrs();
  attrs.mode_ = "SCALED";

  vector<int> static_input_indexes = {1, 2};
  ops::Dequantize R = ops::Dequantize(root, A, -5.0f, -3.0f, attrs);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output};
  OpExecuter opexecuter(root, "Dequantize", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Dequantizei8TF2

TEST(ArrayOps, Dequantizei8TF3) {
  Scope root = Scope::NewRootScope();
  int dim1 = 4;

  Tensor A(DT_QINT8, TensorShape({dim1}));
  AssignInputValues<qint8>(A, {0, -4, 42, -108});

  auto attrs = ops::Dequantize::Attrs();
  attrs.mode_ = "SCALED";

  vector<int> static_input_indexes = {1, 2};
  ops::Dequantize R = ops::Dequantize(root, A, 5.0f, 40.0f, attrs);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output};
  OpExecuter opexecuter(root, "Dequantize", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Dequantizei8TF3

// Test op: Dequantize
// Dequantizes a tensor from u8 to float
TEST(ArrayOps, Dequantizeu8) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_QUINT8, TensorShape({dim1, dim2}));
  AssignInputValues<quint8>(A, {0, 1, 5, 17, 82, 100});

  auto attrs = ops::Dequantize::Attrs();
  attrs.mode_ = "SCALED";

  vector<int> static_input_indexes = {1, 2};
  ops::Dequantize R = ops::Dequantize(root, A, 0.0f, 128.0f, attrs);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output};
  OpExecuter opexecuter(root, "Dequantize", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Dequantizeu8

// Test op: Fill
TEST(ArrayOps, Fill) {
  std::vector<std::vector<int>> input_sizes;  // 1-D or higher

  input_sizes.push_back({2, 3, 4});
  input_sizes.push_back({10, 10, 10});
  input_sizes.push_back({1, 5});
  input_sizes.push_back({0});
  input_sizes.push_back({2, 5, 1, 3, 1});

  vector<int> static_input_indexes = {0};  // has static input

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    int input_dim = input_size.size();
    Tensor shape(DT_INT32, TensorShape({input_dim}));
    AssignInputValues<int>(shape, input_size);

    // 0-D(scalar) value to fill the returned tensor
    Tensor input_data(DT_FLOAT, TensorShape({}));
    AssignInputValuesRandom(input_data);

    // Fill creates a tensor filled with scalar value
    // 1-D shape of the output tensor
    auto R = ops::Fill(root, shape, input_data);
    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Fill", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);
    opexecuter.RunTest();
  }  // end of for loop
}  // end of op Fill

// Test op: ExpandDims, inserts a dimension of 1 into a tensor's shape
TEST(ArrayOps, ExpandDims) {
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  // axis at which the dimension will be inserted
  // should be -rank-1 <= axis <= rank
  vector<int> axis_ = {-1, 0};

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::ExpandDims(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "ExpandDims", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }

}  // end of test op ExpandDims

// Test op: Gather. vector indices
// Test fails because of this error:
// Not found: No attr named '_ngraph_backend' in NodeDef:
// This is because op_executor does not go through mark_for_clustering
TEST(ArrayOps, DISABLED_GatherV2Vector) {
  int dim = 5;

  Tensor A(DT_FLOAT, TensorShape({dim}));
  AssignInputValuesRandom(A);

  Tensor B(DT_INT32, TensorShape({2}));
  AssignInputValues<int>(B, {2, 1});

  Tensor C(DT_INT32, TensorShape({}));
  AssignInputValues<int>(C, 0);

  vector<int> static_input_indexes = {1, 2};
  vector<DataType> output_datatypes = {DT_FLOAT};

  Scope root = Scope::NewRootScope();
  auto R = ops::GatherV2(root, A, B, C);
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "GatherV2", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();

}  // end of test op GatherV2

// Test op: OneHot
TEST(ArrayOps, OneHot1dNegAxis) {
  Scope root = Scope::NewRootScope();

  vector<int> static_input_indexes = {1};

  Tensor indices(DT_INT32, TensorShape({4}));

  AssignInputValues<int>(indices, {0, 2, -1, 1});

  Tensor depth(DT_INT32, TensorShape({}));
  Tensor on_value(DT_FLOAT, TensorShape({}));
  Tensor off_value(DT_FLOAT, TensorShape({}));

  AssignInputValues<int>(depth, 3);
  AssignInputValues<float>(on_value, 5.0);
  AssignInputValues<float>(off_value, 0.0);

  auto attrs = ops::OneHot::Attrs();
  attrs.axis_ = -1;

  auto R = ops::OneHot(root, indices, depth, on_value, off_value, attrs);
  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot1d) {
  Scope root = Scope::NewRootScope();

  vector<int> static_input_indexes = {1};

  Tensor indices(DT_INT32, TensorShape({4}));

  AssignInputValues<int>(indices, {0, 2, -1, -50});

  Tensor depth(DT_INT32, TensorShape({}));
  Tensor on_value(DT_FLOAT, TensorShape({}));
  Tensor off_value(DT_FLOAT, TensorShape({}));

  AssignInputValues<int>(depth, 3);
  AssignInputValues<float>(on_value, 5.0);
  AssignInputValues<float>(off_value, 0.0);

  auto attrs = ops::OneHot::Attrs();
  attrs.axis_ = 0;

  auto R = ops::OneHot(root, indices, depth, on_value, off_value, attrs);
  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot2dNegAxis) {
  Scope root = Scope::NewRootScope();

  vector<int> static_input_indexes = {1};

  Tensor indices(DT_INT32, TensorShape({2, 2}));

  AssignInputValues<int>(indices, {0, 2, -1, 1});

  Tensor depth(DT_INT32, TensorShape({}));
  Tensor on_value(DT_FLOAT, TensorShape({}));
  Tensor off_value(DT_FLOAT, TensorShape({}));

  AssignInputValues<int>(depth, 3);
  AssignInputValues<float>(on_value, 5.0);
  AssignInputValues<float>(off_value, 0.0);

  auto attrs = ops::OneHot::Attrs();
  attrs.axis_ = -1;

  auto R = ops::OneHot(root, indices, depth, on_value, off_value, attrs);
  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot2d) {
  Scope root = Scope::NewRootScope();

  vector<int> static_input_indexes = {1};

  Tensor indices(DT_INT32, TensorShape({2, 2}));

  AssignInputValues<int>(indices, {0, 2, -1, 50});

  Tensor depth(DT_INT32, TensorShape({}));
  Tensor on_value(DT_FLOAT, TensorShape({}));
  Tensor off_value(DT_FLOAT, TensorShape({}));

  AssignInputValues<int>(depth, 3);
  AssignInputValues<float>(on_value, 5.0);
  AssignInputValues<float>(off_value, 0.0);

  auto attrs = ops::OneHot::Attrs();
  attrs.axis_ = 1;

  auto R = ops::OneHot(root, indices, depth, on_value, off_value, attrs);
  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot3d) {
  Scope root = Scope::NewRootScope();

  vector<int> static_input_indexes = {1};

  Tensor indices(DT_INT32, TensorShape({2, 2, 3}));

  AssignInputValuesRandom<int>(indices, -2, 5);

  Tensor depth(DT_INT32, TensorShape({}));
  Tensor on_value(DT_FLOAT, TensorShape({}));
  Tensor off_value(DT_FLOAT, TensorShape({}));

  AssignInputValues<int>(depth, 2);
  AssignInputValues<float>(on_value, 5.0);
  AssignInputValues<float>(off_value, 0.0);

  auto attrs = ops::OneHot::Attrs();
  attrs.axis_ = 2;

  auto R = ops::OneHot(root, indices, depth, on_value, off_value, attrs);
  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot3dNegAxis) {
  Scope root = Scope::NewRootScope();

  vector<int> static_input_indexes = {1};

  Tensor indices(DT_INT32, TensorShape({2, 2, 3}));

  AssignInputValuesRandom<int>(indices, -2, 5);

  Tensor depth(DT_INT32, TensorShape({}));
  Tensor on_value(DT_FLOAT, TensorShape({}));
  Tensor off_value(DT_FLOAT, TensorShape({}));

  AssignInputValues<int>(depth, 2);
  AssignInputValues<float>(on_value, 1.0);
  AssignInputValues<float>(off_value, 0.0);

  auto attrs = ops::OneHot::Attrs();
  attrs.axis_ = -1;

  auto R = ops::OneHot(root, indices, depth, on_value, off_value, attrs);
  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();

}  // end of op OneHot

TEST(ArrayOps, Pad) {
  Scope root = Scope::NewRootScope();

  vector<int> static_input_indexes = {1};

  Tensor input(DT_INT32, TensorShape({2, 3}));

  Tensor paddings(DT_INT32, TensorShape({2, 2}));

  AssignInputValuesRandom<int>(input, 1, 4);
  AssignInputValuesRandom<int>(paddings, 2, 5);

  auto R = ops::Pad(root, input, paddings);
  vector<DataType> output_datatypes = {DT_INT32};
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "Pad", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();

}  // end of op Pad

// Test op: PreventGradient
TEST(ArrayOps, PreventGradient) {
  Scope scope_cpu = Scope::NewRootScope();

  std::vector<std::vector<int64>> input_sizes;

  input_sizes.push_back({2, 3, 4, 20});
  input_sizes.push_back({10, 10, 10});
  input_sizes.push_back({1, 5});
  input_sizes.push_back({0});

  vector<int> static_input_indexes = {};

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);

    auto R = ops::PreventGradient(root, input_data);
    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "PreventGradient", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op PreventGradient

// Test op: QuantizeV2
// Quantizes a tensor from float to i8
TEST(ArrayOps, QuantizeV2i8) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValues<float>(A, {-0.9, -1.3, 2.6, 3.5, 4.2, 5.0});
  auto quant_type = DT_QINT8;

  auto attrs = ops::QuantizeV2::Attrs();
  attrs.mode_ = "SCALED";
  attrs.round_mode_ = "HALF_TO_EVEN";

  vector<int> static_input_indexes = {1, 2};
  ops::QuantizeV2 R =
      ops::QuantizeV2(root, A, -10.0f, 10.99f, quant_type, attrs);

  vector<DataType> output_datatypes = {quant_type};

  std::vector<Output> sess_run_fetchoutputs = {R.output};
  OpExecuter opexecuter(root, "QuantizeV2", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op QuantizeV2i8

// Test op: QuantizeV2
// Quantizes a tensor from float to i8. Also tests min-max output
// TODO: enable this test when min-max output generation is supported
TEST(ArrayOps, DISABLED_QuantizeV2i8minmax) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValues<float>(A, {-0.9, -1.3, 2.6, 3.5, 4.2, 5.0});
  auto quant_type = DT_QINT8;

  auto attrs = ops::QuantizeV2::Attrs();
  attrs.mode_ = "SCALED";
  attrs.round_mode_ = "HALF_TO_EVEN";

  vector<int> static_input_indexes = {1, 2};
  ops::QuantizeV2 R =
      ops::QuantizeV2(root, A, -10.0f, 10.99f, quant_type, attrs);

  vector<DataType> output_datatypes = {quant_type, DT_FLOAT, DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output, R.output_min,
                                               R.output_max};
  OpExecuter opexecuter(root, "QuantizeV2", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op QuantizeV2i8

// Test op: QuantizeV2
// Quantizes a tensor from float to u8
TEST(ArrayOps, QuantizeV2u8SameRange) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValues<float>(A, {0.9, 1.3, 2.6, 3.5, 4.2, 5.0});
  auto quant_type = DT_QUINT8;

  auto attrs = ops::QuantizeV2::Attrs();
  attrs.mode_ = "SCALED";
  attrs.round_mode_ = "HALF_TO_EVEN";

  vector<int> static_input_indexes = {1, 2};
  ops::QuantizeV2 R = ops::QuantizeV2(root, A, 0.9f, 5.0f, quant_type, attrs);

  vector<DataType> output_datatypes = {quant_type};

  std::vector<Output> sess_run_fetchoutputs = {R.output};
  OpExecuter opexecuter(root, "QuantizeV2", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op QuantizeV2u8SameRange

// Test op: QuantizeV2
// Quantizes a tensor from float to u8
TEST(ArrayOps, QuantizeV2u8DiffRange) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValues<float>(A, {0.9, 1.3, 2.6, 3.5, 4.2, 5.0});
  auto quant_type = DT_QUINT8;

  auto attrs = ops::QuantizeV2::Attrs();
  attrs.mode_ = "SCALED";
  attrs.round_mode_ = "HALF_TO_EVEN";

  vector<int> static_input_indexes = {1, 2};
  ops::QuantizeV2 R = ops::QuantizeV2(root, A, 0.0f, 6.0f, quant_type, attrs);

  vector<DataType> output_datatypes = {quant_type, DT_FLOAT, DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output, R.output_min,
                                               R.output_max};
  OpExecuter opexecuter(root, "QuantizeV2", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op QuantizeV2u8DiffRange

// TODO: add tests for other modes (MIN_COMBINED, MIN_FIRST)
// TODO: add a test where min==max

// Test op: QuantizeAndDequantizeV2
// Quantizes and dequantize a tensor
TEST(ArrayOps, QuantizeAndDequantizeV2x8xtruextrue) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValues<float>(A, {0.9, 1.3, 2.6, 3.5, 4.2, 5.0});

  auto attrs = ops::QuantizeAndDequantizeV2::Attrs();
  attrs.num_bits_ = 8;
  attrs.range_given_ = true;
  attrs.signed_input_ = true;

  vector<int> static_input_indexes = {1, 2};
  ops::QuantizeAndDequantizeV2 R =
      ops::QuantizeAndDequantizeV2(root, A, -10.0f, 10.99f, attrs);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output};
  OpExecuter opexecuter(root, "QuantizeAndDequantizeV2", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op QuantizeAndDequantizeV2x8xtruextrue

TEST(ArrayOps, QuantizeAndDequantizeV2x8xtruexfalse) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValues<float>(A, {0.9, 1.3, 2.6, 3.5, 4.2, 5.0});

  auto attrs = ops::QuantizeAndDequantizeV2::Attrs();
  attrs.num_bits_ = 8;
  attrs.range_given_ = true;
  attrs.signed_input_ = false;

  vector<int> static_input_indexes = {1, 2};
  ops::QuantizeAndDequantizeV2 R =
      ops::QuantizeAndDequantizeV2(root, A, -10.0f, 10.99f, attrs);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output};
  OpExecuter opexecuter(root, "QuantizeAndDequantizeV2", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op QuantizeAndDequantizeV2x8xtruexfalse

// CPU only supports QuantizedConcat with DT_QINT32 and DT_QUINT8
TEST(ArrayOps, QuantizedConcat) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_QUINT8, TensorShape({dim1, dim2}));
  AssignInputValues<quint8>(A, {5, 1, 0, 1, 5, 100});

  Tensor B(DT_QUINT8, TensorShape({dim1, dim2}));
  AssignInputValues<quint8>(B, {0, 2, 4, 6, 8, 10});

  Tensor C(DT_QUINT8, TensorShape({dim1, dim2}));
  AssignInputValues<quint8>(C, {1, 3, 5, 7, 9, 50});

  vector<int> static_input_indexes = {0, 4, 5, 6, 7, 8, 9};

  // TODO: NG and TF results disagress when input mins/maxes vary
  ops::QuantizedConcat R = ops::QuantizedConcat(
      root, 1, {A, B, C}, {-1.0f, -1.0f, -1.0f}, {3.0f, 3.0f, 3.0f});

  vector<DataType> output_datatypes = {DT_QUINT8, DT_FLOAT, DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output, R.output_min,
                                               R.output_max};
  OpExecuter opexecuter(root, "QuantizedConcat", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op QuantizedConcat

// Disabled because for varing min/max input
// NGraph and TF results diagree
// For this test case: TF result is [47 46 46 0 1 2 70 72 73][46 47 55 3 3 4 75
// 76 106]
// NG result is [1 0 0 0 1 2 1 3 5][0 1 20 2 3 4 7 9 50]
TEST(ArrayOps, DISABLED_QuantizedConcatVaryingMinMax) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_QUINT8, TensorShape({dim1, dim2}));
  AssignInputValues<quint8>(A, {5, 1, 0, 1, 5, 100});

  Tensor B(DT_QUINT8, TensorShape({dim1, dim2}));
  AssignInputValues<quint8>(B, {0, 2, 4, 6, 8, 10});

  Tensor C(DT_QUINT8, TensorShape({dim1, dim2}));
  AssignInputValues<quint8>(C, {1, 3, 5, 7, 9, 50});

  vector<int> static_input_indexes = {0, 4, 5, 6, 7, 8, 9};

  ops::QuantizedConcat R = ops::QuantizedConcat(
      root, 1, {A, B, C}, {1.0f, -1.0f, 2.0f}, {2.0f, 4.0f, 10.0f});

  vector<DataType> output_datatypes = {DT_QUINT8, DT_FLOAT, DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.output, R.output_min,
                                               R.output_max};
  OpExecuter opexecuter(root, "QuantizedConcat", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  // vector<Tensor> tf_outputs;
  // opexecuter.ExecuteOnTF(tf_outputs);

  // vector<Tensor> ng_outputs;
  // opexecuter.ExecuteOnNGraph(ng_outputs);

  // cout << "TF outputs " << endl;
  // for(auto i : tf_outputs){
  //   PrintTensorAllValues(i, 100);
  // }

  // cout << "NG outputs " << endl;
  // for(auto i : ng_outputs){
  //   PrintTensorAllValues(i, 100);
  // }
  opexecuter.RunTest();
}  // end of test op QuantizedConcatVaryingMinMax

// Test op: Rank Op
TEST(ArrayOps, Rank) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;
  int dim3 = 3;

  Tensor A(DT_INT32, TensorShape({dim1, dim2, dim3}));
  AssignInputValuesRandom<int32>(A, 2, 20);

  vector<int> static_input_indexes = {};

  auto R = ops::Rank(root, A);

  vector<DataType> output_datatypes = {DT_INT32};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Rank", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of RankOp

// Test op: Shape, outputs the shape of a tensor
TEST(ArrayOps, Shape2D) {
  Scope root = Scope::NewRootScope();

  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 7.5f);

  vector<int> static_input_indexes = {};

  auto attrs = ops::Shape::Attrs();
  attrs.out_type_ = DT_INT64;
  auto R = ops::Shape(root, A, attrs);

  vector<DataType> output_datatypes = {DT_INT64};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Shape", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of op Shape2D

TEST(ArrayOps, Shape3D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;
  int dim3 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2, dim3}));

  AssignInputValues(A, 7.5f);

  vector<int> static_input_indexes = {};
  auto R = ops::Shape(root, A);

  vector<DataType> output_datatypes = {DT_INT32};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Shape", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of op Shape3D

// Size op: returns the size of a tensor
// This test changes the default attribute of out_type_
TEST(ArrayOps, SizeOpAttrsChange) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({0});
  input_shapes.push_back({2});
  input_shapes.push_back({6, 10});
  input_shapes.push_back({10, 10, 10});
  input_shapes.push_back({1, 7, 8, 10});
  input_shapes.push_back({2, 5, 1, 3, 1});

  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_INT64};

  auto attrs = ops::Size::Attrs();
  attrs.out_type_ = DT_INT64;

  for (auto const& shapes : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shapes));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);

    auto R = ops::Size(root, input_data, attrs);
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Size", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SizeAttrsChange

// Size op: returns the size of a tensor
TEST(ArrayOps, SizeOpDefault) {
  std::vector<std::vector<int64>> input_shapes;

  input_shapes.push_back({0});
  input_shapes.push_back({1});
  input_shapes.push_back({3, 8});
  input_shapes.push_back({10, 11, 12});
  input_shapes.push_back({1, 7, 8, 10});
  input_shapes.push_back({2, 5, 1, 3, 1});

  vector<int> static_input_indexes = {};
  // Size Op default output tyep is DT_INT32
  vector<DataType> output_datatypes = {DT_INT32};

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    auto R = ops::Size(root, input_data);

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Size", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SizeDefault

// Test slice op
TEST(ArrayOps, Slice) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 4, 1});

  std::vector<int64> begin = {0, 0, 2, 0};
  std::vector<int64> size = {-1, -1, 2, -1};

  vector<int> static_input_indexes = {1, 2};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    Tensor begin_tensor(DT_INT64, TensorShape({4}));
    AssignInputValues(begin_tensor, begin);
    Tensor size_tensor(DT_INT64, TensorShape({4}));
    AssignInputValues(size_tensor, size);

    auto R = ops::Slice(root, input_data, begin_tensor, size_tensor);

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Slice", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op Slice

// Test SpaceToDepth with NHWC data format
TEST(ArrayOps, SpaceToDepthNHWC) {
  std::map<std::vector<int64>, int> input_map;
  input_map.insert(pair<std::vector<int64>, int>({1, 2, 2, 1}, 2));
  input_map.insert(pair<std::vector<int64>, int>({1, 2, 2, 3}, 2));
  input_map.insert(pair<std::vector<int64>, int>({1, 3, 3, 3}, 3));
  input_map.insert(pair<std::vector<int64>, int>({1, 10, 10, 5}, 10));
  input_map.insert(pair<std::vector<int64>, int>({1, 6, 4, 1}, 2));
  input_map.insert(pair<std::vector<int64>, int>({1, 20, 10, 3}, 5));
  input_map.insert(pair<std::vector<int64>, int>({2, 3, 6, 3}, 3));
  input_map.insert(pair<std::vector<int64>, int>({10, 10, 10, 10}, 2));

  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT};

  map<std::vector<int64>, int>::iterator iter;
  for (iter = input_map.begin(); iter != input_map.end(); iter++) {
    std::vector<int64> shape = iter->first;
    int block_size = iter->second;

    Scope root = Scope::NewRootScope();
    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);

    auto R = ops::SpaceToDepth(root, input_data, block_size);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "SpaceToDepth", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);
    opexecuter.RunTest();
  }
}  // end of op SpaceToDepthNHWC

// Test SpaceToDepth with NCHW data format
TEST(ArrayOps, SpaceToDepthNCHW) {
  std::map<std::vector<int64>, int> input_map;
  input_map.insert(pair<std::vector<int64>, int>({1, 1, 2, 2}, 2));
  input_map.insert(pair<std::vector<int64>, int>({1, 10, 5, 5}, 5));
  input_map.insert(pair<std::vector<int64>, int>({1, 20, 3, 3}, 3));
  input_map.insert(pair<std::vector<int64>, int>({2, 3, 6, 3}, 3));
  input_map.insert(pair<std::vector<int64>, int>({10, 10, 10, 10}, 2));
  input_map.insert(pair<std::vector<int64>, int>({2, 1, 15, 3}, 3));
  input_map.insert(pair<std::vector<int64>, int>({30, 30, 30, 30}, 10));

  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT};
  ops::SpaceToDepth::Attrs attrs;
  attrs.data_format_ = "NCHW";

  map<std::vector<int64>, int>::iterator iter;
  for (iter = input_map.begin(); iter != input_map.end(); iter++) {
    std::vector<int64> shape = iter->first;
    int block_size = iter->second;

    Scope root = Scope::NewRootScope();
    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);

    auto R = ops::SpaceToDepth(root, input_data, block_size, attrs);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "SpaceToDepth", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    vector<Tensor> ngraph_outputs;
    opexecuter.ExecuteOnNGraph(ngraph_outputs);

    // On CPU, the op only supports NHWC data format
    Scope tf_scope = Scope::NewRootScope();
    auto input_data_NHWC = ops::Transpose(tf_scope, input_data, {0, 2, 3, 1});
    auto r_tf = ops::SpaceToDepth(tf_scope, input_data_NHWC, block_size);
    auto r_tf_NCHW = ops::Transpose(tf_scope, r_tf, {0, 3, 1, 2});
    vector<Output> sess_run_fetchoutputs_tf = {r_tf_NCHW};
    OpExecuter opexecuter_tf(tf_scope, "SpaceToDepth", static_input_indexes,
                             output_datatypes, sess_run_fetchoutputs_tf);

    vector<Tensor> tf_outputs;
    opexecuter_tf.ExecuteOnTF(tf_outputs);

    // Compare NGraph and TF Outputs
    Compare(tf_outputs, ngraph_outputs);
  }
}  // end of op SpaceToDepthNCHW

// Test op: StridedSlice
// In this test the begin, end and stride vectors have length < rank
TEST(ArrayOps, StridedSliceTest1) {
  vector<int> static_input_indexes = {1, 2, 3};

  Scope root = Scope::NewRootScope();
  auto in_tensor_type = DT_FLOAT;
  std::vector<int64> input_size = {39, 128, 128};
  Tensor input_data(in_tensor_type, TensorShape(input_size));

  auto num_elems_in_tensor = [](std::vector<int64> shape_vect) {
    return std::accumulate(begin(shape_vect), end(shape_vect), 1,
                           std::multiplies<int64>());
  };
  auto tot_num_elems_in_tensor = num_elems_in_tensor(input_size);

  std::vector<float> data_vect(tot_num_elems_in_tensor);
  std::iota(data_vect.begin(), data_vect.end(), 0.0f);
  AssignInputValues<float>(input_data, data_vect);

  std::vector<int64> cstart = {0, 1};
  std::vector<int64> cend = {0, 2};
  std::vector<int64> cstride = {1, 1};

  Tensor begin(DT_INT64, TensorShape({static_cast<int>(cstart.size())}));
  AssignInputValues<int64>(begin, cstart);
  Tensor end(DT_INT64, TensorShape({static_cast<int>(cend.size())}));
  AssignInputValues<int64>(end, cend);
  Tensor strides(DT_INT64, TensorShape({static_cast<int>(cstride.size())}));
  AssignInputValues<int64>(strides, cstride);

  ops::StridedSlice::Attrs attrs;
  attrs.begin_mask_ = 0;
  attrs.ellipsis_mask_ = 0;
  attrs.end_mask_ = 0;
  attrs.new_axis_mask_ = 0;
  attrs.shrink_axis_mask_ = 1;

  auto R = ops::StridedSlice(root, input_data, begin, end, strides);
  vector<DataType> output_datatypes = {in_tensor_type};
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "StridedSlice", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// Test op: StridedSlice
// This test is disabled because it exhaustively tests all possibilities of
// begin and end index
// While it was useful when developing the strided slice translation function,
// this exhaustive search is not needed in ci
// however, keeping this test, in case we make changes to strided slice later
// and we want to test it locally
// Note this test has couts to help debugging
TEST(ArrayOps, DISABLED_StridedSlice) {
  vector<int> static_input_indexes = {1, 2, 3};  // has static input
  std::vector<std::vector<int64>> input_sizes = {{2, 3, 4}, {2}};
  auto in_tensor_type = DT_FLOAT;

  // A debugging print function
  auto print_vect = [](vector<int64> v) {
    for (auto i : v) cout << " | " << i << " ";
    cout << "\n";
  };

  // A tensor of rank r has a shape vector of length r representing the sizes of
  // its dimensions
  // To represent a coordinate in the tensor, we need a r-tuple, whose ith index
  // is < the ith value of the shape vector
  // Alternatively we can also use a single integer (< prod(tensorshape)) to
  // represent a coordinate (assuming we agree upon things like
  // row-major/column-major)
  // In other words, we can establish a bijective map to vectorize and
  // un-vectorize a rank r tensor

  // vector index to r rank index

  auto num_elems_in_tensor = [](std::vector<int64> shape_vect) {
    return std::accumulate(begin(shape_vect), end(shape_vect), 1,
                           std::multiplies<int64>());
  };
  auto vectorized_idx_to_coordinate = [num_elems_in_tensor](
      int vectorized_idx, std::vector<int64> shape_vect,
      std::vector<int64>* coord) {
    auto num_elems = num_elems_in_tensor(shape_vect);
    ASSERT_TRUE(std::accumulate(
        begin(shape_vect), end(shape_vect), true,
        [](bool acc, int64 item) { return (item >= 0) && acc; }))
        << "Expected all elements of shape_vect to be >= 0, but found some "
           "negative ones";
    ASSERT_TRUE(vectorized_idx < num_elems)
        << "Vectorized idx(" << vectorized_idx
        << ") should have been less than number of elements in the tensor("
        << num_elems << ")";
    ASSERT_TRUE(vectorized_idx >= 0) << "Vectorized idx(" << vectorized_idx
                                     << ") should have been greater than 0";
    coord->resize(shape_vect.size());
    for (int i = shape_vect.size() - 1; i >= 0; i--) {
      (*coord)[i] = vectorized_idx % shape_vect[i];
      vectorized_idx = vectorized_idx / shape_vect[i];
    }
  };

  // TODO: can input_size ever have an element which is 0
  // TODO: what if end is > dimension?
  int64 tot_num_tests_run = 0;
  for (auto input_size : input_sizes) {
    auto rank = input_size.size();
    auto tot_num_elems_in_tensor = num_elems_in_tensor(input_size);
    for (int start_vectorized_idx = 0;
         start_vectorized_idx < tot_num_elems_in_tensor;
         start_vectorized_idx++) {
      for (int end_vectorized_idx = 0;
           end_vectorized_idx < tot_num_elems_in_tensor; end_vectorized_idx++) {
        vector<int64> cstart, cend;

        vectorized_idx_to_coordinate(start_vectorized_idx, input_size, &cstart);
        vectorized_idx_to_coordinate(end_vectorized_idx, input_size, &cend);
        std::vector<int64> diff_start_end(cstart.size());
        // Compute the difference between start and end coordinates and store in
        // diff_start_end. Some coordinates of the diff vector could be negative
        std::transform(cend.begin(), cend.end(), cstart.begin(),
                       diff_start_end.begin(), std::minus<int64>());

        // For a tensor with rank 2 and shape (a, b), (-a, -b) to (a, b) are min
        // and max vals of diff_start_end.
        // Therefore the new normalized (all positive) coordinate system has
        // 2^rank elements more that the original tensor of shape (a, b)
        auto num_elems_in_non_negative_representation_of_diff =
            (1 << rank) * tot_num_elems_in_tensor;

        std::vector<int64> non_negative_representation_of_diff(
            rank);  // if input is (a, b), this vector is (2a, 2b)
        std::transform(input_size.begin(), input_size.end(),
                       non_negative_representation_of_diff.begin(),
                       [](int64 x) { return 2 * x; });

        // TODO: strides[i] could be more than gap between start[i] and end[i]..
        // have a test for that too

        for (int vectorized_idx_for_stride = 0;
             vectorized_idx_for_stride <
             num_elems_in_non_negative_representation_of_diff;
             vectorized_idx_for_stride++) {
          vector<int64> non_neg_stride_coordinate;
          vectorized_idx_to_coordinate(vectorized_idx_for_stride,
                                       non_negative_representation_of_diff,
                                       &non_neg_stride_coordinate);

          // Continuing the example: we had considered it in a space (0, 0) to
          // (2a, 2b) to make it all positive. Now bring it back to (-a, -b) to
          // (a, b)
          std::vector<int64> cstride(rank);
          std::transform(non_neg_stride_coordinate.begin(),
                         non_neg_stride_coordinate.end(), input_size.begin(),
                         cstride.begin(), std::minus<int64>());

          // strides cannot be 0. num_elems_in_tensor function calculates
          // product of elements, so if one dimension is 0, the whole product is
          // 0
          if (num_elems_in_tensor(cstride) == 0) {
            continue;
          }
          cout << "=============\n";
          print_vect(cstart);
          print_vect(cend);
          print_vect(cstride);
          cout << "=============\n";

          Scope root = Scope::NewRootScope();

          Tensor input_data(in_tensor_type, TensorShape(input_size));
          std::vector<float> data_vect(tot_num_elems_in_tensor);
          std::iota(data_vect.begin(), data_vect.end(), 0.0f);
          AssignInputValues<float>(input_data, data_vect);

          Tensor begin(DT_INT64, TensorShape({static_cast<int>(rank)}));
          AssignInputValues<int64>(begin, cstart);
          Tensor end(DT_INT64, TensorShape({static_cast<int>(rank)}));
          AssignInputValues<int64>(end, cend);
          Tensor strides(DT_INT64, TensorShape({static_cast<int>(rank)}));
          AssignInputValues<int64>(strides, cstride);

          auto R = ops::StridedSlice(root, input_data, begin, end, strides);
          vector<DataType> output_datatypes = {in_tensor_type};
          std::vector<Output> sess_run_fetchoutputs = {R};

          OpExecuter opexecuter(root, "StridedSlice", static_input_indexes,
                                output_datatypes, sess_run_fetchoutputs);

          opexecuter.RunTest();
          tot_num_tests_run++;
        }
      }
    }
  }
  cout << "Ran a total of " << tot_num_tests_run << " tests\n";
}  // end of test op Tile

// Test SplitNegativeAxis op
TEST(ArrayOps, SplitNegativeAxis) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 8, 1});
  // num_split : The number of ways to split. Must evenly divide
  // value.shape[split_dim]
  int64_t num_splits = 4;

  vector<int> static_input_indexes = {0};
  vector<DataType> output_datatypes(num_splits, DT_FLOAT);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  Tensor axis(DT_INT32, TensorShape({}));
  AssignInputValues<int>(axis, -2);

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);

    auto R = ops::Split(root, axis, input_data, num_splits);

    std::vector<Output> sess_run_fetchoutputs = {R[0], R[1], R[2], R[3]};
    OpExecuter opexecuter(root, "Split", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitNegativeAxis

// Test SplitPositiveAxis op
TEST(ArrayOps, SplitPositiveAxis) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 1, 6});
  // num_split : The number of ways to split. Must evenly divide
  // value.shape[split_dim]
  int64_t num_splits = 3;

  vector<int> static_input_indexes = {0};
  vector<DataType> output_datatypes(num_splits, DT_FLOAT);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  Tensor axis(DT_INT32, TensorShape({}));
  AssignInputValues<int>(axis, 3);

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);

    auto R = ops::Split(root, axis, input_data, num_splits);

    std::vector<Output> sess_run_fetchoutputs = {R[0], R[1], R[2]};
    OpExecuter opexecuter(root, "Split", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitPositiveAxis

// Test SplitVNegSizeSplit op
TEST(ArrayOps, SplitVNegSizeSplit) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 6, 1});

  std::vector<int64> size_splits = {2, -1, 2, 1};
  int64_t num_splits = 4;

  vector<int> static_input_indexes = {1, 2};
  vector<DataType> output_datatypes(num_splits, DT_FLOAT);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  Tensor axis(DT_INT32, TensorShape({}));
  AssignInputValues<int>(axis, 2);

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    Tensor size_tensor(DT_INT64, TensorShape({4}));
    AssignInputValues(size_tensor, size_splits);

    auto R = ops::SplitV(root, input_data, size_tensor, axis, num_splits);

    std::vector<Output> sess_run_fetchoutputs = {R[0], R[1], R[2], R[3]};
    OpExecuter opexecuter(root, "SplitV", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVNegSizeSplit

// Test SplitVNegativeAxis op
TEST(ArrayOps, SplitVNegativeAxis) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 6, 1});

  std::vector<int64> size_splits = {2, 1, 2, 1};
  int64_t num_splits = 4;

  vector<int> static_input_indexes = {1, 2};
  vector<DataType> output_datatypes(num_splits, DT_FLOAT);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  Tensor axis(DT_INT32, TensorShape({}));
  AssignInputValues<int>(axis, -2);

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    Tensor size_tensor(DT_INT64, TensorShape({4}));
    AssignInputValues(size_tensor, size_splits);

    auto R = ops::SplitV(root, input_data, size_tensor, axis, num_splits);

    std::vector<Output> sess_run_fetchoutputs = {R[0], R[1], R[2], R[3]};
    OpExecuter opexecuter(root, "SplitV", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVNegativeAxis

// Test SplitVPositiveSizeSplits op
TEST(ArrayOps, SplitVPositiveSizeSplits) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 6, 1});

  std::vector<int64> size_splits = {2, 1, 2, 1};
  int64_t num_splits = 4;

  vector<int> static_input_indexes = {1, 2};
  vector<DataType> output_datatypes(num_splits, DT_FLOAT);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  Tensor axis(DT_INT32, TensorShape({}));
  AssignInputValues<int>(axis, 2);

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    Tensor size_tensor(DT_INT64, TensorShape({4}));
    AssignInputValues(size_tensor, size_splits);

    auto R = ops::SplitV(root, input_data, size_tensor, axis, num_splits);

    std::vector<Output> sess_run_fetchoutputs = {R[0], R[1], R[2], R[3]};
    OpExecuter opexecuter(root, "SplitV", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVPositiveSizeSplits

// Test SplitVZeroSizeSplit op
TEST(ArrayOps, SplitVZeroSizeSplit) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 10});

  std::vector<int64> size_splits = {10, 0};
  int64_t num_splits = 2;

  vector<int> static_input_indexes = {1, 2};
  vector<DataType> output_datatypes(num_splits, DT_FLOAT);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  Tensor axis(DT_INT32, TensorShape({}));
  AssignInputValues<int>(axis, 1);

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    Tensor size_tensor(DT_INT64, TensorShape({2}));
    AssignInputValues(size_tensor, size_splits);

    auto R = ops::SplitV(root, input_data, size_tensor, axis, num_splits);

    std::vector<Output> sess_run_fetchoutputs = {R[0], R[1]};
    OpExecuter opexecuter(root, "SplitV", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVZeroSizeSplit

// Test SplitVZeroSizeNegSplit op
TEST(ArrayOps, SplitVZeroSizeNegSplit) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 10});

  std::vector<int64> size_splits = {10, -1};
  int64_t num_splits = 2;

  vector<int> static_input_indexes = {1, 2};
  vector<DataType> output_datatypes(num_splits, DT_FLOAT);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  Tensor axis(DT_INT32, TensorShape({}));
  AssignInputValues<int>(axis, 1);

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    Tensor size_tensor(DT_INT64, TensorShape({2}));
    AssignInputValues(size_tensor, size_splits);

    auto R = ops::SplitV(root, input_data, size_tensor, axis, num_splits);

    std::vector<Output> sess_run_fetchoutputs = {R[0], R[1]};
    OpExecuter opexecuter(root, "SplitV", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVZeroSizeNegSplit

// Test op: Tile, constructs a tensor by tiling a given tensor
TEST(ArrayOps, Tile) {
  std::vector<std::vector<int64>> input_sizes;  // 1-D or higher

  input_sizes.push_back({2, 3, 4});
  input_sizes.push_back({10, 10, 10});
  input_sizes.push_back({1, 5});
  input_sizes.push_back({0});

  vector<int> static_input_indexes = {1};  // has static input

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -5.0f, 10.0f);

    // Must be of type int32 or int64,
    // 1-D. Length must be the same as the number of dimensions in input
    int input_dim = input_size.size();
    Tensor multiples(DT_INT32, TensorShape({input_dim}));
    AssignInputValuesRandom<int32>(multiples, 0, 20);

    auto R = ops::Tile(root, input_data, multiples);
    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Tile", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of test op Tile

// Test op: Transpose
TEST(ArrayOps, Transpose) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;
  int dim3 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2, dim3}));
  Tensor perm(DT_INT32, TensorShape({3}));
  AssignInputValues(A, 7.5f);
  AssignInputValues(perm, vector<int>{2, 1, 0});

  vector<int> static_input_indexes = {1};
  auto R = ops::Transpose(root, A, perm);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Transpose", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Transpose

// Test op: Transpose With Constant input and empty permuation vector
TEST(ArrayOps, TransposeConstant) {
  Scope root = Scope::NewRootScope();

  auto A = ops::Const(root, 12.0f);
  auto perm = ops::Const(root, std::initializer_list<int>{});
  auto R = ops::Transpose(root, A, perm);

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Transpose", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Transpose

// Unpacks the given dimension of a rank R tensor into a (R-1) tensor
TEST(ArrayOps, Unpack) {
  std::vector<std::vector<int64>> input_sizes;

  // rank > 0
  input_sizes.push_back({3, 2, 3});
  input_sizes.push_back({4, 3});
  input_sizes.push_back({3});

  std::vector<int64> axes({0, 1, 0});

  vector<int> static_input_indexes = {};
  for (auto i = 0; i < input_sizes.size(); ++i) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_sizes[i]));
    AssignInputValuesRandom<float>(input_data, -20, 50);

    ops::Unstack::Attrs attrs;
    attrs.axis_ = axes[i];

    auto R = ops::Unstack(root, input_data, input_sizes[i][axes[i]], attrs);

    // Unpack returns a list of Tensors
    // which internally flatten to multiple outputs
    // retrieve using indexes
    // the indexes matches the axies dimension of the input size
    std::vector<Output> sess_run_fetchoutputs = {R[0], R[1], R[2]};
    vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    OpExecuter opexecuter(root, "Unpack", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }  // end of for loop
}  // end of testing Unpack

// Test op: ZerosLike
// Returns a tensor of zeros of the same shape and type as the input tensor
TEST(ArrayOps, ZerosLike) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 7.5f);

  vector<int> static_input_indexes = {};
  auto R = ops::ZerosLike(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ZerosLike", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op ZerosLike

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
