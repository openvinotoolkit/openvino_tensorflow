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
    AssignInputValuesFromVector<int>(shape, input_size);

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

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
