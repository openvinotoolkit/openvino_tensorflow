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

// Test op: Abs
TEST(MathOps, Abs1D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 1;
  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValuesRandom(A);

  vector<int> static_input_indexes = {};
  auto R = ops::Abs(root, A);
  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Abs", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Abs2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);

  vector<int> static_input_indexes = {};
  auto R = ops::Abs(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Abs", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Abs

// Test op: Add
TEST(MathOps, Add) {
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
}  // end of test op Add

// Test op: AddN
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
}  // end of test op AddN

// Test op: Any
// Any with attribute KeepDims set to true
TEST(MathOps, AnyKeepDims) {
  int dim1 = 2;
  int dim2 = 2;
  std::vector<bool> v = {true, true, true, true};

  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  auto keep_dims = ops::Any::Attrs().KeepDims(true);
  AssignInputValuesFromVector<bool>(A, v);
  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = 0;
  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_BOOL};

  Scope root = Scope::NewRootScope();
  auto R = ops::Any(root, A, axis, keep_dims);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Any", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, AnyNegativeAxis) {
  int dim1 = 2;
  int dim2 = 3;
  std::vector<bool> v = {true, true, true, true, false, false};

  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  AssignInputValuesFromVector<bool>(A, v);
  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = -1;
  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_BOOL};

  Scope root = Scope::NewRootScope();
  auto R = ops::Any(root, A, axis);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Any", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);
  opexecuter.RunTest();
}

TEST(MathOps, AnyPositiveAxis) {
  int dim1 = 3;
  int dim2 = 3;
  std::vector<bool> v = {true,  true, true,  true, false,
                         false, true, false, false};

  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  AssignInputValuesFromVector<bool>(A, v);
  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = 1;
  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_BOOL};

  Scope root = Scope::NewRootScope();
  auto R = ops::Any(root, A, axis);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Any", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Any

// Test op: All
// All with attribute KeepDims set to true
TEST(MathOps, AllKeepDims) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  std::vector<bool> v = {true, true, true, false};
  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  auto keep_dims = ops::All::Attrs().KeepDims(true);

  AssignInputValuesFromVector<bool>(A, v);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = 0;

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_BOOL};

  auto R = ops::All(root, A, axis, keep_dims);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "All", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, AllNegativeAxis) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  std::vector<bool> v = {true, true, true, true, false, false};
  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));

  AssignInputValuesFromVector<bool>(A, v);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = -1;

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_BOOL};

  auto R = ops::All(root, A, axis);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "All", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, AllPositiveAxis) {
  Scope root = Scope::NewRootScope();
  int dim1 = 3;
  int dim2 = 3;

  std::vector<bool> v = {true,  true, true,  true, false,
                         false, true, false, false};
  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));

  AssignInputValuesFromVector<bool>(A, v);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = 1;

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_BOOL};

  auto R = ops::All(root, A, axis);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "All", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op All

// ArgMax test for negative dimension
TEST(MathOps, ArgMax_Neg) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  int dim = -1;

  vector<int> static_input_indexes = {1};

  auto R = ops::ArgMax(root, A, dim);

  vector<DataType> output_datatypes = {DT_INT64};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ArgMax", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);
  opexecuter.RunTest();
}

// ArgMax test for positive dimension
TEST(MathOps, ArgMax_Pos) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  int dim = 1;

  vector<int> static_input_indexes = {1};

  auto attrs = ops::ArgMax::Attrs();
  attrs.output_type_ = DT_INT32;

  auto R = ops::ArgMax(root, A, dim, attrs);

  vector<DataType> output_datatypes = {DT_INT32};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ArgMax", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op ArgMax

// Test op: BatchMatMul
TEST(MathOps, BatchMatMul2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 2.0f);
  AssignInputValues(B, 7.0f);

  vector<int> static_input_indexes = {};
  auto R = ops::BatchMatMul(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "BatchMatMul", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, BatchMatMul3D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;
  int dim3 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2, dim3}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2, dim3}));

  AssignInputValues(A, 2.0f);
  AssignInputValues(B, 7.0f);
  AssignInputValues(B, 5.0f);

  vector<int> static_input_indexes = {};

  auto R = ops::BatchMatMul(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "BatchMatMul", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// BatchMatMul 3D with attributes AdjX set to true
TEST(MathOps, BatchMatMul3DAdjX) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;
  int dim3 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2, dim3}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2, dim3}));

  auto attrs_x = ops::BatchMatMul::Attrs().AdjX(true);

  AssignInputValues(A, 2.0f);
  AssignInputValues(B, 7.0f);

  vector<int> static_input_indexes = {};

  auto R = ops::BatchMatMul(root, A, B, attrs_x);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "BatchMatMul", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// BatchMatMul 3D with attributes AdjY set to true
TEST(MathOps, BatchMatMul3DAdjY) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;
  int dim3 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2, dim3}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2, dim3}));

  auto attrs_y = ops::BatchMatMul::Attrs().AdjY(true);

  AssignInputValues(A, 2.0f);
  AssignInputValues(B, 7.0f);
  AssignInputValues(B, 5.0f);

  vector<int> static_input_indexes = {};

  auto R = ops::BatchMatMul(root, A, B, attrs_y);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "BatchMatMul", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op BatchMatMul

// Test op: Cast : float to int
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
}  // end of test op Cast

// Test op: Exp
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
}  // end of test op Exp

// Test op: FloorDiv
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
}  // end of test op FloorDiv

// Test op: FloorDivBroadcasting
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
}  // end of test op FloorDivBroadcasting

// Test op: FloorMod
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
}  // end of test op FloorMod

// Test op: FloorModBroadcasting
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
}  // end of test op FloorModBroadcasting

// Test op: Log
TEST(MathOps, Log1D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 1.4f);

  vector<int> static_input_indexes = {};
  auto R = ops::Log(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Log", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Log2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 3.5f);

  vector<int> static_input_indexes = {};
  auto R = ops::Log(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Log", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Log

// Test op: Max
TEST(MathOps, MaxNegativeAxis) {
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  vector<int> axis_ = {-1};

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::Max(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Max", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

TEST(MathOps, MaxPositiveAxis) {
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  vector<int> axis_ = {0};

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::Max(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Max", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }

}  // end of test op Max

// Test op: Min
TEST(MathOps, MinNegativeAxis) {
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  vector<int> axis_ = {-1};

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::Min(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Min", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

TEST(MathOps, MinPositiveAxis) {
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  vector<int> axis_ = {0};

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::Min(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Min", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }

}  // end of test op Min

// Test op: Minimum
TEST(MathOps, Minimum) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);
  AssignInputValuesRandom(B);

  vector<int> static_input_indexes = {};
  auto R = ops::Minimum(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Minimum", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Minimum

// Test op: MinimumBroadcasting
TEST(MathOps, MinimumBroadcasting) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 7.5f);
  AssignInputValues(B, 5.2f);

  vector<int> static_input_indexes = {};
  auto R = ops::Minimum(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Minimum", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op MinimumBroadcasting

// Test op: Negate
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
}  // end of Test op Negate

// Test op: Pow
TEST(MathOps, Pow1D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 4;
  Tensor A(DT_FLOAT, TensorShape({dim1}));
  Tensor B(DT_FLOAT, TensorShape({dim1}));
  AssignInputValues(A, 1.4f);
  AssignInputValues(B, 0.5f);
  vector<int> static_input_indexes = {};
  auto R = ops::Pow(root, A, B);
  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Pow", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);
  opexecuter.RunTest();
}

TEST(MathOps, Pow2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;
  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValues(A, -2.5f);
  AssignInputValues(B, 4.0f);
  vector<int> static_input_indexes = {};
  auto R = ops::Pow(root, A, B);
  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Pow", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Pow

// Test op: RealDiv
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
}  // end of test op RealDiv

// Test op: RealDivBroadcasting
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
}  // end of test op RealDivBroadcasting

// Test op: Reciprocal
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
}  // end of test op Reciprocal

// Test op: Rsqrt
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
}  // end of test op Rsqrt

// Test op: Square
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
}  // end of test op Square

// Test op: Sqrt
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
}  // end of test op Sqrt

// Test op: SquareDifference
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
}  // end of test op SquaredDifference

// Test op: SquaredDifferenceBroadcasting
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
}  // end of test op SquaredDifferenceBroadcasting

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
