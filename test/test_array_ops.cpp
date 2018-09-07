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

TEST(ArrayOps, PreventGradient) {
  Scope scope_cpu = Scope::NewRootScope();

  std::vector<std::vector<int64>> input_sizes;

  input_sizes.push_back({2, 3, 4, 20});
  input_sizes.push_back({10, 10, 10});
  input_sizes.push_back({1, 5});
  input_sizes.push_back({0});

  vector<int> static_input_indexes = {};  // has static input

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom(input_data);

    auto R = ops::PreventGradient(root, input_data);
    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "PreventGradient", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op PreventGradient

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
    AssignInputValuesRandom(input_data);

    // placeholder, will be modified accordingly by the if loop
    auto mul = ops::Const(root, {int64(1)}, TensorShape({1}));

    if (input_size.size() == 3) {
      mul = ops::Const(root, {int32(10), int32(0), int32(1)}, TensorShape({3}));
    } else if (input_size.size() == 2) {
      mul = ops::Const(root, {int32(2), int32(3)}, TensorShape({2}));
    } else if (input_size.size() == 1) {
      mul = ops::Const(root, {int64(5)}, TensorShape({1}));
    }

    auto R = ops::Tile(root, input_data, mul);
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
    AssignInputValuesRandom(input_data);

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
