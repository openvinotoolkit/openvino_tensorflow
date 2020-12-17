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
#include <map>

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

#include <cmath>
#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/opexecuter.h"
#include "test/test_utilities.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

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

  auto R = ops::Abs(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Abs", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Abs2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);

  auto R = ops::Abs(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Abs", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Abs

// Test op: Acos
TEST(MathOps, Acos2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);

  auto R = ops::Acos(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Acos", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Acos

// Test op: Acosh
TEST(MathOps, Acosh) {
  Scope root = Scope::NewRootScope();
  int dim1 = 3;
  int dim2 = 5;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);

  auto R = ops::Acosh(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Acosh", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Acosh

// Test op: Add
TEST(MathOps, Add) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 2.1f);
  AssignInputValues(B, 4.1f);

  auto R = ops::Add(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Add", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Add

// Test op: AddV2
TEST(MathOps, AddV2) {
  // Run a bunch of sub-test combinations to check shape broadcasting
  vector<TensorShape> tensors_combs = {
      // A-size, B-size
      {2, 4}, {2, 4},  // sub-test# 1
      {2, 4}, {2, 1},  // sub-test# 2
      {2, 4}, {1, 4},  // sub-test# 3
      {2, 1}, {2, 4},  // sub-test# 4
      {1, 4}, {2, 4},  // sub-test# 5
      {2, 4}, {1, 1},  // sub-test# 6
      {1, 1}, {2, 4},  // sub-test# 7
  };

  for (int i = 0; i < tensors_combs.size(); i += 2) {
    NGRAPH_VLOG(5) << "========>> Running AddV2 sub-test# " << (int)(i / 2 + 1)
                   << " ...";

    Scope root = Scope::NewRootScope();

    Tensor A(DT_FLOAT, TensorShape(tensors_combs[i]));
    Tensor B(DT_FLOAT, TensorShape(tensors_combs[i + 1]));

    AssignInputValues(A, 2.1f);
    AssignInputValues(B, 4.1f);

    auto R = ops::AddV2(root, A, B);

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "AddV2", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }

}  // end of test op AddV2

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

  auto R = ops::AddN(root, {A, B, C});

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "AddN", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op AddN

// Test op: Any
// Any with attribute KeepDims set to true
// Fails with opset3 upgrade because there is no opset0
// downgrade available for it in nGraph
TEST(MathOps, AnyKeepDims) {
  int dim1 = 2;
  int dim2 = 2;
  std::vector<bool> v = {true, true, true, true};

  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  auto keep_dims = ops::Any::Attrs().KeepDims(true);
  AssignInputValues<bool>(A, v);
  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = 0;

  Scope root = Scope::NewRootScope();
  auto R = ops::Any(root, A, axis, keep_dims);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Any", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, AnyNegativeAxis) {
  int dim1 = 2;
  int dim2 = 3;
  std::vector<bool> v = {true, true, true, true, false, false};

  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  AssignInputValues<bool>(A, v);
  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = -1;

  Scope root = Scope::NewRootScope();
  auto R = ops::Any(root, A, axis);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Any", sess_run_fetchoutputs);
  opexecuter.RunTest();
}

TEST(MathOps, AnyPositiveAxis) {
  int dim1 = 3;
  int dim2 = 3;
  std::vector<bool> v = {true,  true, true,  true, false,
                         false, true, false, false};

  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  AssignInputValues<bool>(A, v);
  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = 1;

  Scope root = Scope::NewRootScope();
  auto R = ops::Any(root, A, axis);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Any", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Any

// Test op: All
// All with attribute KeepDims set to true
// Fails with opset3 upgrade because there is no opset0
// downgrade available for it in nGraph
TEST(MathOps, AllKeepDims) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  std::vector<bool> v = {true, true, true, false};
  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  auto keep_dims = ops::All::Attrs().KeepDims(true);

  AssignInputValues<bool>(A, v);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = 0;

  auto R = ops::All(root, A, axis, keep_dims);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "All", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, AllNegativeAxis) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  std::vector<bool> v = {true, true, true, true, false, false};
  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));

  AssignInputValues<bool>(A, v);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = -1;

  auto R = ops::All(root, A, axis);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "All", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, AllPositiveAxis) {
  Scope root = Scope::NewRootScope();
  int dim1 = 3;
  int dim2 = 3;

  std::vector<bool> v = {true,  true, true,  true, false,
                         false, true, false, false};
  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));

  AssignInputValues<bool>(A, v);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  int axis = 1;

  auto R = ops::All(root, A, axis);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "All", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op All

// Test op: Asin
TEST(MathOps, Asin) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);

  auto R = ops::Asin(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Asin", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Asin

// Test op: Asinh
TEST(MathOps, Asinh) {
  Scope root = Scope::NewRootScope();
  int dim1 = 3;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);

  auto R = ops::Asinh(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Asinh", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Asinh

// Test op: Atan
TEST(MathOps, Atan) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValuesRandom(A);

  auto R = ops::Atan(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Atan", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Atan

// Test op: Atanh
TEST(MathOps, Atanh) {
  Scope root = Scope::NewRootScope();
  int dim1 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValuesRandom(A);

  auto R = ops::Atanh(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Atanh", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Atanh

// Test op: Cumsum
TEST(MathOps, Cumsum) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_INT32, TensorShape({}));

  AssignInputValues(A, 2.1f);
  AssignInputValues(B, 0);

  auto attrs = ops::Cumsum::Attrs();
  attrs.exclusive_ = true;
  attrs.reverse_ = true;
  auto R = ops::Cumsum(root, A, B, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Cumsum", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Cumsum

// Test op: Sum with & without keep dims & with both positive & negative axis
TEST(MathOps, Sum) {
  int dim1 = 2;
  int dim2 = 2;

  std::vector<int> v = {1, 2, 3, 4};
  Tensor A(DT_INT32, TensorShape({dim1, dim2}));
  vector<bool> v_keep_dims = {true, false};
  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  vector<int> v_axis = {-1, 0, 1};
  for (auto axis : v_axis) {
    for (auto keep_dims : v_keep_dims) {
      Scope root = Scope::NewRootScope();
      auto keep_dims_attr = ops::Sum::Attrs().KeepDims(keep_dims);

      AssignInputValues<int>(A, v);

      auto R = ops::Sum(root, A, axis, keep_dims_attr);
      std::vector<Output> sess_run_fetchoutputs = {R};
      OpExecuter opexecuter(root, "Sum", sess_run_fetchoutputs);

      opexecuter.RunTest();
    }
  }
}

// BEGIN MathOpsSumFixture
class MathOpsSumFixture : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
  void TestWith(std::vector<int64> shape, std::vector<int64> axis,
                bool keep_dims) {
    std::cout << ">> Running test with: tensor shape=[" << ngraph::join(shape)
              << "], axis=[" << ngraph::join(axis)
              << "], keep_dims=" << keep_dims << std::endl;
    Scope root = Scope::NewRootScope();

    std::vector<int> vals = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                             1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    Tensor A(DT_INT32, TensorShape(shape));
    AssignInputValues<int>(A, vals);

    Tensor B(DT_INT64, TensorShape({static_cast<int64>(axis.size())}));
    AssignInputValues<int64>(B, axis);

    auto keep_dims_attr = ops::Sum::Attrs().KeepDims(keep_dims);
    auto R = ops::Sum(root, A, B, keep_dims_attr);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Sum", sess_run_fetchoutputs);
    opexecuter.RunTest();
  }
};

TEST_F(MathOpsSumFixture, LimitedSet1) {
  TestWith({2, 3}, {1}, false);
  TestWith({6, 2}, {0}, false);
  TestWith({2, 4, 3}, {0}, false);
}

TEST_F(MathOpsSumFixture, FullSet) {
  vector<vector<int64>> shapes = {{2, 3},         {2, 2, 3},    {6, 2},
                                  {2, 4, 3},      {4, 3, 2, 1}, {1, 2, 3, 4},
                                  {3, 1, 2, 4, 1}};
  vector<bool> v_keep_dims = {true, false};
  for (auto shape : shapes) {
    for (int64 i = 0; i < shape.size(); i++) {
      for (auto keep_dims : v_keep_dims) {
        TestWith(shape, {i}, keep_dims);
      }
    }
  }
  // Add few more tests with axes-combos
  TestWith({6, 2}, {0, 1}, false);
  TestWith({6, 2}, {0, 1}, true);
  TestWith({2, 4, 3}, {0, 1, 2}, false);
}
// END MathOpsSumFixture

// Test op: Mean with & without keep dims & with both positive & negative axis
TEST(MathOps, Mean) {
  int dim1 = 2;
  int dim2 = 2;

  std::vector<int> v = {1, 2, 3, 4};
  Tensor A(DT_INT32, TensorShape({dim1, dim2}));
  vector<bool> v_keep_dims = {true, false};
  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  vector<int> v_axis = {-1, 0, 1};
  for (auto axis : v_axis) {
    for (auto keep_dims : v_keep_dims) {
      Scope root = Scope::NewRootScope();
      auto keep_dims_attr = ops::Mean::Attrs().KeepDims(keep_dims);

      AssignInputValues<int>(A, v);

      auto R = ops::Mean(root, A, axis, keep_dims_attr);
      std::vector<Output> sess_run_fetchoutputs = {R};
      OpExecuter opexecuter(root, "Mean", sess_run_fetchoutputs);

      opexecuter.RunTest();
    }
  }
}

// Test op: Prod with & without keep dims & with both positive & negative axis
TEST(MathOps, Prod) {
  int dim1 = 2;
  int dim2 = 2;

  std::vector<int> v = {1, 2, 3, 4};
  Tensor A(DT_INT32, TensorShape({dim1, dim2}));
  vector<bool> v_keep_dims = {true, false};
  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  vector<int> v_axis = {-1, 0, 1};
  for (auto axis : v_axis) {
    for (auto keep_dims : v_keep_dims) {
      Scope root = Scope::NewRootScope();
      auto keep_dims_attr = ops::Prod::Attrs().KeepDims(keep_dims);

      AssignInputValues<int>(A, v);

      auto R = ops::Prod(root, A, axis, keep_dims_attr);
      std::vector<Output> sess_run_fetchoutputs = {R};
      OpExecuter opexecuter(root, "Prod", sess_run_fetchoutputs);

      opexecuter.RunTest();
    }
  }
}

// ArgMax test for negative dimension
TEST(MathOps, ArgMaxNeg) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  int dim = -1;
  auto attrs = ops::ArgMax::Attrs();
  attrs.output_type_ = DT_INT32;

  auto R = ops::ArgMax(root, A, dim, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ArgMax", sess_run_fetchoutputs);
  opexecuter.RunTest();
}

// ArgMax test for positive dimension
TEST(MathOps, ArgMaxPos) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  int dim = 1;

  auto attrs = ops::ArgMax::Attrs();
  attrs.output_type_ = DT_INT32;

  auto R = ops::ArgMax(root, A, dim, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ArgMax", sess_run_fetchoutputs);
  opexecuter.RunTest();
}
// ArgMax test for 3D
TEST(MathOps, ArgMax3D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;
  int dim3 = 1;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2, dim3}));
  AssignInputValuesRandom(A);

  int dim = -1;
  auto attrs = ops::ArgMax::Attrs();
  attrs.output_type_ = DT_INT32;

  auto R = ops::ArgMax(root, A, dim, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ArgMax", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op ArgMax

// ArgMin test for negative dimension
TEST(MathOps, ArgMinNeg) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  int dim = -1;
  auto attrs = ops::ArgMin::Attrs();
  attrs.output_type_ = DT_INT32;

  auto R = ops::ArgMin(root, A, dim, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ArgMin", sess_run_fetchoutputs);
  opexecuter.RunTest();
}

// ArgMin test for positive dimension
TEST(MathOps, ArgMinPos) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  int dim = 1;

  auto attrs = ops::ArgMin::Attrs();
  attrs.output_type_ = DT_INT32;

  auto R = ops::ArgMin(root, A, dim, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ArgMin", sess_run_fetchoutputs);
  opexecuter.RunTest();
}

// ArgMin test for 3D
TEST(MathOps, ArgMin3D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;
  int dim3 = 1;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2, dim3}));
  AssignInputValuesRandom(A);

  int dim = 1;
  auto attrs = ops::ArgMin::Attrs();
  attrs.output_type_ = DT_INT32;

  auto R = ops::ArgMin(root, A, dim, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ArgMin", sess_run_fetchoutputs);
  opexecuter.RunTest();
}

// ArgMin test for empty output
TEST(MathOps, ArgMinEmpty) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValues<float>(A, {0, 0, 0, 0, 0, 0});

  int dim = 1;

  auto attrs = ops::ArgMin::Attrs();
  attrs.output_type_ = DT_INT32;

  auto R = ops::ArgMin(root, A, dim, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ArgMin", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op ArgMin

// Test op: Atan2
TEST(MathOps, Atan2) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 5;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues<float>(A, {0, -0, 3, -3.5, 1.2, 3, 5, -4.5, 1.0, -7.0});
  AssignInputValues<float>(B, {0, -0, 3, 2.5, -0.7, 2, 3.4, -5.6, 30, 0.06});

  auto R = ops::Atan2(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Atan2", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Atan2

// Test op: MatMul
TEST(MathOps, MatMul) {
  Scope root = Scope::NewRootScope();

  Tensor A(DT_FLOAT, TensorShape({2, 3}));
  Tensor B(DT_FLOAT, TensorShape({3, 4}));

  AssignInputValues(A, 2.0f);
  AssignInputValues(B, 7.0f);

  auto R = ops::MatMul(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "MatMul", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// Test op: Cast : float to int
TEST(MathOps, Cast1D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValuesRandom(A);

  auto R = ops::Cast(root, A, DT_INT32);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Cast", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Cast2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);

  auto R = ops::Cast(root, A, DT_INT32);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Cast", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Cast

// Test op: Ceil
TEST(MathOps, Ceil) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 5;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValuesRandom(A);

  auto R = ops::Ceil(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Ceil", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Ceil

// Test op: Cos
TEST(MathOps, Cos) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 5;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues<float>(
      A, {0, -0, M_PI / 2, M_PI, 1.0, 3.8, 4.2, -3.9, -4.2, -1.0});

  auto R = ops::Cos(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Cos", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Cos

// Test op: Cosh
TEST(MathOps, Cosh) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValuesRandom(A);

  auto R = ops::Cosh(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Cosh", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Cosh

// Test op: Exp
TEST(MathOps, Exp1D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 2.5f);

  auto R = ops::Exp(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Exp", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Exp2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 3.6f);

  auto R = ops::Exp(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Exp", sess_run_fetchoutputs);

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

  auto R = ops::FloorDiv(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorDiv", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op FloorDiv

TEST(MathOps, FloorDivInt) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_INT32, TensorShape({dim1, dim2}));
  Tensor B(DT_INT32, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4);
  AssignInputValues(B, 3);

  auto R = ops::FloorDiv(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorDiv", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op FloorDivInt

// Test op: FloorDivBroadcasting
TEST(MathOps, FloorDivBroadcasting) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 4.5f);
  AssignInputValues(B, 3.2f);

  auto R = ops::FloorDiv(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorDiv", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op FloorDivBroadcasting

// Test op: FloorDivNegInt
TEST(MathOps, FloorDivNegInt) {
  Scope root = Scope::NewRootScope();

  Tensor A(DT_INT32, TensorShape({1}));
  Tensor B(DT_INT32, TensorShape({1}));

  AssignInputValues(A, -1);
  AssignInputValues(B, 3);

  auto R = ops::FloorDiv(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorDiv", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op FloorDivNegInt

// For FloorDiv op, the input and output data type should match
TEST(MathOps, FloorDivNegFloat) {
  Scope root = Scope::NewRootScope();

  Tensor A(DT_FLOAT, TensorShape({1}));
  Tensor B(DT_FLOAT, TensorShape({1}));

  AssignInputValues(A, -1.f);
  AssignInputValues(B, 3.f);

  auto R = ops::FloorDiv(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorDiv", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op FloorDivNegFloat

// Test op: FloorMod
TEST(MathOps, FloorMod) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_INT32, TensorShape({dim1, dim2}));
  Tensor B(DT_INT32, TensorShape({dim1, dim2}));

  AssignInputValues(A, 7);
  AssignInputValues(B, 5);

  auto R = ops::FloorMod(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorMod", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op FloorMod

// Test op: FloorModBroadcasting
TEST(MathOps, FloorModBroadcasting) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_INT32, TensorShape({dim1, dim2}));
  Tensor B(DT_INT32, TensorShape({dim1}));

  AssignInputValues(A, 7);
  AssignInputValues(B, 5);

  auto R = ops::FloorMod(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "FloorMod", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op FloorModBroadcasting

// Test op: FloorModNegInt
// Currently failing with TF produces {2,2}, NG produces {-8,-3}
// Should enable when NGraph fixes the FloorMod
TEST(MathOps, FloorModNegInt) {
  Scope root = Scope::NewRootScope();
  vector<int> nums = {-8, -8};
  vector<int> divs = {10, 5};

  Tensor A(DT_INT32, TensorShape({1, 2}));
  Tensor B(DT_INT32, TensorShape({1, 2}));

  AssignInputValues(A, nums);
  AssignInputValues(B, divs);

  auto R = ops::FloorMod(root, A, B);
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "FloorMod", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op FloorModNegInt

TEST(MathOps, FloorModNegFloat) {
  Scope root = Scope::NewRootScope();

  vector<float> nums = {-8.f, -8.f};
  vector<float> divs = {10.f, 5.f};

  Tensor A(DT_FLOAT, TensorShape({1, 2}));
  Tensor B(DT_FLOAT, TensorShape({1, 2}));

  AssignInputValues(A, nums);
  AssignInputValues(B, divs);

  auto R = ops::FloorMod(root, A, B);
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "FloorMod", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op FloorModNegFloat

// Test op: IsFinite
TEST(MathOps, IsFinite) {
  Scope root = Scope::NewRootScope();
  int dim1 = 8;

  Tensor A(DT_FLOAT, TensorShape({dim1}));
  std::vector<float> values{0.f,
                            1.f,
                            2.f,
                            -2.f,
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::signaling_NaN()};
  AssignInputValues(A, values);
  auto R = ops::IsFinite(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "IsFinite", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op IsFinite

// Test op: Log
TEST(MathOps, Log1D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 1.4f);

  auto R = ops::Log(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Log", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, Log2D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 3.5f);

  auto R = ops::Log(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Log", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Log

TEST(MathOps, Log1p) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 4;
  vector<float> vals = {-2, -1, 0, 0.25, 0.5, 1, 5, 10};

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, vals);

  auto R = ops::Log1p(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Log1p", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Log1p

// Test Op:LogicalOr
TEST(MathOps, LogicalOr) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;
  std::vector<bool> v1 = {true, true, true, true, false, false};
  std::vector<bool> v2 = {false, true, false, true, false, false};

  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  AssignInputValues(A, v1);

  Tensor B(DT_BOOL, TensorShape({dim1, dim2}));
  AssignInputValues(B, v2);

  auto R = ops::LogicalOr(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "LogicalOr", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of LogicalOr

// Test Op:LogicalNot
TEST(MathOps, LogicalNot) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;
  std::vector<bool> v1 = {true, true, true, true, false, false};

  Tensor A(DT_BOOL, TensorShape({dim1, dim2}));
  AssignInputValues(A, v1);

  auto R = ops::LogicalNot(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "LogicalNot", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of LogicalNot

// Test op: Max
TEST(MathOps, MaxNegativeAxis) {
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  // axis at which the dimension will be inserted
  // should be -rank <= axis < rank
  vector<int> axis_ = {-1};

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::Max(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Max", sess_run_fetchoutputs);

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

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::Max(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Max", sess_run_fetchoutputs);

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

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::Min(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Min", sess_run_fetchoutputs);

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

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::Min(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Min", sess_run_fetchoutputs);

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

  auto R = ops::Minimum(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Minimum", sess_run_fetchoutputs);

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

  auto R = ops::Minimum(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Minimum", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op MinimumBroadcasting

// Test op: MaximumBroadcasting
TEST(MathOps, MaximumBroadcasting) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1}));

  AssignInputValues(A, 7.5f);
  AssignInputValues(B, 5.2f);

  auto R = ops::Maximum(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Maximum", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op MaximumBroadcasting

// Test op: Negate
TEST(MathOps, Negate) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 16.5f);

  auto R = ops::Negate(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Neg", sess_run_fetchoutputs);

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
  auto R = ops::Pow(root, A, B);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Pow", sess_run_fetchoutputs);
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
  auto R = ops::Pow(root, A, B);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Pow", sess_run_fetchoutputs);
  opexecuter.RunTest();
}

// Broadcasting
TEST(MathOps, Pow0D1D) {
  Scope root = Scope::NewRootScope();
  Tensor A(DT_FLOAT, TensorShape({}));   // scalar == rank 0 (no axes)
  Tensor B(DT_FLOAT, TensorShape({5}));  // vector == rank 1 (1 axis)
  AssignInputValues(A, 2.1f);
  AssignInputValues(B, 4.1f);
  auto R = ops::Pow(root, A, B);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Pow", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Pow

// Test op: Range
TEST(MathOps, RangeFloat) {
  Scope root = Scope::NewRootScope();

  Tensor start(DT_FLOAT, TensorShape({}));  // scalar == rank 0 (no axes)
  Tensor limit(DT_FLOAT, TensorShape({}));  // scalar == rank 0 (no axes)
  Tensor delta(DT_FLOAT, TensorShape({}));  // scalar == rank 0 (no axes)

  AssignInputValues(start, 2.0f);
  AssignInputValues(limit, 7.5f);
  AssignInputValues(delta, 1.5f);

  auto R = ops::Range(root, start, limit, delta);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Range", sess_run_fetchoutputs);

  setenv("NGRAPH_TF_CONSTANT_FOLDING", "1", true);
  opexecuter.RunTest();
  unsetenv("NGRAPH_TF_CONSTANT_FOLDING");
}

TEST(MathOps, RangeInt) {
  Scope root = Scope::NewRootScope();

  Tensor start(DT_INT32, TensorShape({}));  // scalar == rank 0 (no axes)
  Tensor limit(DT_INT32, TensorShape({}));  // scalar == rank 0 (no axes)
  Tensor delta(DT_INT32, TensorShape({}));  // scalar == rank 0 (no axes)

  AssignInputValues(start, 2);
  AssignInputValues(limit, 7);
  AssignInputValues(delta, 1);

  auto R = ops::Range(root, start, limit, delta);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Range", sess_run_fetchoutputs);

  setenv("NGRAPH_TF_CONSTANT_FOLDING", "1", true);
  opexecuter.RunTest();
  unsetenv("NGRAPH_TF_CONSTANT_FOLDING");
}  // end of test op Range

// Test op: RealDiv
TEST(MathOps, RealDiv) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 2.0f);
  AssignInputValues(B, 7.0f);

  auto R = ops::RealDiv(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "RealDiv", sess_run_fetchoutputs);

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

  auto R = ops::RealDiv(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "RealDiv", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op RealDivBroadcasting

// Test op: RealDiv for nan, inf case
TEST(MathOps, RealDivNonfinite) {
  Scope root = Scope::NewRootScope();
  int dim = 3;

  Tensor A(DT_FLOAT, TensorShape({dim}));
  Tensor B(DT_FLOAT, TensorShape({dim}));

  auto inf = std::numeric_limits<float>::infinity();

  vector<float> dividend_vals = {0, -inf, inf};
  vector<float> divisor_vals = {0, 1.0, 1.0};

  AssignInputValues(A, dividend_vals);
  AssignInputValues(B, divisor_vals);

  auto R = ops::RealDiv(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "RealDiv", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test RealDivNonfinite

// Test op: Reciprocal
TEST(MathOps, Reciprocal) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 2.0f);

  auto R = ops::Reciprocal(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Reciprocal", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Reciprocal

// Test op: Relu
TEST(MathOps, Relu) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 2.0f);

  auto R = ops::Relu(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Relu", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Relu

// Test op: Rsqrt
TEST(MathOps, Rsqrt) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.0f);

  auto R = ops::Rsqrt(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Rsqrt", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Rsqrt

// Test op: Sign
TEST(MathOps, Sign) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom<float>(A, -50, 50);

  auto R = ops::Sign(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Sign", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Sign

// Test op: Sin
TEST(MathOps, Sin) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 5;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues<float>(
      A, {0, -0, M_PI / 2, M_PI, 1.0, 3.8, 4.2, -3.9, -4.2, -1.0});

  auto R = ops::Sin(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Sin", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Sin

// Test op: Sinh
TEST(MathOps, Sinh) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 5;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues<float>(
      A, {0, -0, M_PI / 2, M_PI, 1.0, 3.8, 4.2, -3.9, -4.2, -1.0});

  auto R = ops::Sinh(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Sinh", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Sinh

// Test op: Square
TEST(MathOps, Square) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.0f);

  auto R = ops::Square(root, A);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Square", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Square

// Test op: SqueezeNoAttributes
TEST(MathOps, SqueezeNoAttributes) {
  vector<vector<int64>> shape_vector;
  shape_vector.push_back({1, 10, 2, 3});
  shape_vector.push_back({2, 2, 3, 4});
  shape_vector.push_back({10, 1, 5, 1});
  shape_vector.push_back({1, 1, 1, 1});

  for (auto shape : shape_vector) {
    Scope root = Scope::NewRootScope();

    Tensor input(DT_INT32, TensorShape(shape));
    AssignInputValuesRandom<int32>(input, -50, 50);

    auto R = ops::Squeeze(root, input);

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Squeeze", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of test op SqueezeNoAttributes

// Test op: SqueezeWithAttributes
TEST(MathOps, SqueezeWithAttributes) {
  // construct a map to store input shape and squeeze dimension attributes
  map<vector<int64>, gtl::ArraySlice<int>> shape_attributes_map;
  shape_attributes_map.insert(
      pair<vector<int64>, gtl::ArraySlice<int>>({1, 10, 2, 3}, {0}));
  shape_attributes_map.insert(
      pair<vector<int64>, gtl::ArraySlice<int>>({10, 1, 5, 1}, {-1, -3}));
  shape_attributes_map.insert(
      pair<vector<int64>, gtl::ArraySlice<int>>({1, 1, 1, 1}, {-1, -2}));
  shape_attributes_map.insert(
      pair<vector<int64>, gtl::ArraySlice<int>>({1, 1, 1, 1}, {0, 1, -2, -3}));

  for (auto itr : shape_attributes_map) {
    Scope root = Scope::NewRootScope();

    auto input_shape = itr.first;
    auto squeeze_dim = itr.second;

    Tensor input(DT_FLOAT, TensorShape(input_shape));
    AssignInputValuesRandom<float>(input, -50, 50);

    auto attrs = ops::Squeeze::Attrs();
    attrs.axis_ = squeeze_dim;

    auto R = ops::Squeeze(root, input, attrs);

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Squeeze", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of test op SqueezeWithAttributes

// Test op: Sqrt
TEST(MathOps, Sqrt) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;
  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.0f);

  auto R = ops::Sqrt(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Sqrt", sess_run_fetchoutputs);
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

  auto R = ops::SquaredDifference(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "SquaredDifference", sess_run_fetchoutputs);

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

  auto R = ops::SquaredDifference(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "SquaredDifference", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op SquaredDifferenceBroadcasting

// Test op: Xdivy
TEST(MathOps, Xdivy) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.3f);
  AssignInputValues(B, 3.7f);

  auto R = ops::Xdivy(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Xdivy", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, XdivyZeroX) {
  Scope root = Scope::NewRootScope();
  int dim1 = 3;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, std::vector<float>{0.0f, 1.1f, 5.1f, 3.2f, 8.1f, 1.0f,
                                          -1.0f, 2.0f, 0.0f});
  AssignInputValues(B, std::vector<float>{2.0f, 1.2f, 4.2f, 8.9f, 0.0f, 0.0f,
                                          0.0f, 0.0f, 0.0f});

  auto R = ops::Xdivy(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Xdivy", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(MathOps, XdivyZeroXZeroY) {
  Scope root = Scope::NewRootScope();

  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 0.0f);
  AssignInputValues(B, 0.0f);

  auto R = ops::Xdivy(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Xdivy", sess_run_fetchoutputs);
  opexecuter.RunTest();
}  // end of test op Xdivy

// Test op: Tan
TEST(MathOps, Tan) {
  Scope root = Scope::NewRootScope();

  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 3.5f);

  auto R = ops::Tan(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Tan", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Tan

// Test op: Tanh
TEST(MathOps, Tanh) {
  Scope root = Scope::NewRootScope();

  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 7.5f);

  auto R = ops::Tanh(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Tanh", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Tanh

// Test op: NotEqual
TEST(MathOps, NotEqual) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 2.1f);
  AssignInputValues(B, 4.1f);

  auto R = ops::NotEqual(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "NotEqual", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op NotEqual

// Test op: Mod
TEST(MathOps, Mod) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 4.1f);
  AssignInputValues(B, 2.0f);

  auto R = ops::Mod(root, A, B);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Mod", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Mod

}  // namespace testing
}  // namespace ngraph_bridge
}
