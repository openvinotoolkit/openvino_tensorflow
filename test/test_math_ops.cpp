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

TEST(MathOps, Add) {
  // Create a tf graph
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 2.1f);
  AssignInputValues(B, 4.1f);

  vector<int> static_input_indexes = {};
  auto R = ops::Add(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Add", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, RealDiv) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 2.0f);
  AssignInputValues(B, 7.0f);

  vector<int> static_input_indexes = {};
  auto R = ops::RealDiv(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "RealDiv", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, RealDivBroadcasting) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 2.0f);
  AssignInputValues(B, 7.0f);

  vector<int> static_input_indexes = {};
  auto R = ops::RealDiv(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "RealDiv", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Sqrt) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.0f);

  vector<int> static_input_indexes = {};
  auto R = ops::Sqrt(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Sqrt", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Negate) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 16.5f);

  vector<int> static_input_indexes = {};
  auto R = ops::Negate(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Neg", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Square) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.0f);

  vector<int> static_input_indexes = {};
  auto R = ops::Square(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Square", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Reciprocal) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 2.0f);

  vector<int> static_input_indexes = {};
  auto R = ops::Reciprocal(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Reciprocal", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, AddN) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor C(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.5f);
  AssignInputValues(B, 3.2f);
  AssignInputValues(C, 2.3f);

  vector<int> static_input_indexes = {};
  auto R = ops::AddN(root, {A, B, C});

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "AddN", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, FloorDiv) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.5f);
  AssignInputValues(B, 3.2f);

  vector<int> static_input_indexes = {};
  auto R = ops::FloorDiv(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorDiv", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, FloorDivBroadcasting) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 4.5f);
  AssignInputValues(B, 3.2f);

  vector<int> static_input_indexes = {};
  auto R = ops::FloorDiv(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorDiv", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, FloorMod) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 7.5f);
  AssignInputValues(B, 5.2f);

  vector<int> static_input_indexes = {};
  auto R = ops::FloorMod(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorMod", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, FloorModBroadcasting) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 7.5f);
  AssignInputValues(B, 5.2f);

  vector<int> static_input_indexes = {};
  auto R = ops::FloorMod(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorMod", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Rsqrt) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.0f);

  vector<int> static_input_indexes = {};
  auto R = ops::Rsqrt(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Rsqrt", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, SquaredDifference) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 7.5f);
  AssignInputValues(B, 5.2f);

  vector<int> static_input_indexes = {};
  auto R = ops::SquaredDifference(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "SquaredDifference", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, SquaredDifferenceBroadcasting) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 7.5f);
  AssignInputValues(B, 5.2f);

  vector<int> static_input_indexes = {};
  auto R = ops::SquaredDifference(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "SquaredDifference", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// Cast float to int
TEST(MathOps, Cast1D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValuesRandom(A);

  vector<int> static_input_indexes = {};
  auto R = ops::Cast(root, A, DT_INT32);

  vector<DataType> output_datatypes = {DT_INT32};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Cast", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// Cast float to int
TEST(MathOps, Cast2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);

  vector<int> static_input_indexes = {};
  auto R = ops::Cast(root, A, DT_INT32);

  vector<DataType> output_datatypes = {DT_INT32};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Cast", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Exp1D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 2.5);

  vector<int> static_input_indexes = {};
  auto R = ops::Exp(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Exp", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Exp2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 3.6);

  vector<int> static_input_indexes = {};
  auto R = ops::Exp(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Exp", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}
}  // namespace testing

}  // namespace ngraph_bridge
}  // namespace tensorflow
