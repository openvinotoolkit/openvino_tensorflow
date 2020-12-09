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

  map<std::vector<int64>, int>::iterator iter;
  for (iter = input_map.begin(); iter != input_map.end(); iter++) {
    std::vector<int64> shape = iter->first;
    int block_size = iter->second;

    Scope root = Scope::NewRootScope();
    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);

    auto R = ops::DepthToSpace(root, input_data, block_size);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "DepthToSpace", sess_run_fetchoutputs);
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
    OpExecuter opexecuter(root, "DepthToSpace", sess_run_fetchoutputs);

    vector<Tensor> ngraph_outputs;
    opexecuter.ExecuteOnNGraph(ngraph_outputs);

    // On CPU, the op only supports NHWC data format
    Scope tf_scope = Scope::NewRootScope();
    auto input_data_NHWC = ops::Transpose(tf_scope, input_data, {0, 2, 3, 1});
    auto r_tf = ops::DepthToSpace(tf_scope, input_data_NHWC, block_size);
    auto r_tf_NCHW = ops::Transpose(tf_scope, r_tf, {0, 3, 1, 2});
    vector<Output> sess_run_fetchoutputs_tf = {r_tf_NCHW};
    OpExecuter opexecuter_tf(tf_scope, "DepthToSpace",
                             sess_run_fetchoutputs_tf);

    vector<Tensor> tf_outputs;
    opexecuter_tf.ExecuteOnTF(tf_outputs);

    // Compare NGraph and TF Outputs
    Compare(tf_outputs, ngraph_outputs);
  }
}  // end of op DepthToSpaceNCHW

// Test op: Fill
TEST(ArrayOps, Fill) {
  std::vector<std::vector<int>> input_sizes;  // 1-D or higher

  input_sizes.push_back({2, 3, 4});
  input_sizes.push_back({10, 10, 10});
  input_sizes.push_back({1, 5});
  input_sizes.push_back({0});
  input_sizes.push_back({2, 5, 1, 3, 1});

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
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Fill", sess_run_fetchoutputs);
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

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::ExpandDims(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "ExpandDims", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }

}  // end of test op ExpandDims

TEST(ArrayOps, GatherVector) {
  Tensor A(DT_FLOAT, TensorShape({5}));  // input
  AssignInputValues<float>(A, {10.1, 20.2, 30.3, 40.4, 50.5});

  Tensor B(DT_INT32, TensorShape({2}));  // indices
  AssignInputValues<int>(B, {2, 1});

  Scope root = Scope::NewRootScope();
  auto R = ops::Gather(root, A, B);
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "Gather", sess_run_fetchoutputs);
  opexecuter.RunTest();
}

TEST(ArrayOps, GatherTensor) {
  Tensor A(DT_FLOAT, TensorShape({5, 5, 5, 5}));
  AssignInputValuesRandom(A);

  Tensor B(DT_INT32, TensorShape({10}));
  AssignInputValues<int>(B, {0, 4, 2, 2, 3, 1, 3, 0, 3, 3});

  Scope root = Scope::NewRootScope();
  auto R = ops::Gather(root, A, B);
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "Gather", sess_run_fetchoutputs);
  opexecuter.RunTest();
}

// Test op: Gather. vector indices
TEST(ArrayOps, GatherV2Vector) {
  int dim = 5;

  Tensor A(DT_FLOAT, TensorShape({dim}));
  AssignInputValuesRandom(A);

  Tensor B(DT_INT32, TensorShape({2}));
  AssignInputValues<int>(B, {2, 1});

  Tensor C(DT_INT32, TensorShape({}));
  AssignInputValues<int>(C, 0);

  Scope root = Scope::NewRootScope();
  auto R = ops::GatherV2(root, A, B, C);
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "GatherV2", sess_run_fetchoutputs);

  opexecuter.RunTest();

}  // end of test op GatherV2

TEST(ArrayOps, GatherV2Tensor) {
  Tensor A(DT_FLOAT, TensorShape({5, 5, 5, 5}));
  AssignInputValuesRandom(A);

  Tensor B(DT_INT32, TensorShape({10}));
  AssignInputValues<int>(B, {0, 4, 2, 2, 3, 1, 3, 0, 3, 3});

  Tensor C(DT_INT32, TensorShape({}));
  AssignInputValues<int>(C, 0);

  Scope root = Scope::NewRootScope();
  auto R = ops::GatherV2(root, A, B, C);
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "GatherV2", sess_run_fetchoutputs);

  opexecuter.RunTest();

}  // end of test op GatherV2

TEST(ArrayOps, GatherV2TensorAxis2) {
  Tensor A(DT_FLOAT, TensorShape({5, 5, 5, 5}));
  AssignInputValuesRandom(A);

  Tensor B(DT_INT32, TensorShape({10}));
  AssignInputValues<int>(B, {0, 4, 2, 2, 3, 1, 3, 0, 3, 3});

  Tensor C(DT_INT32, TensorShape({}));
  AssignInputValues<int>(C, 2);

  Scope root = Scope::NewRootScope();
  auto R = ops::GatherV2(root, A, B, C);
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "GatherV2", sess_run_fetchoutputs);

  opexecuter.RunTest();

}  // end of test op GatherV2

// Test op: OneHot
TEST(ArrayOps, OneHot1dNegAxis) {
  Scope root = Scope::NewRootScope();

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
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot1d) {
  Scope root = Scope::NewRootScope();

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
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot2dNegAxis) {
  Scope root = Scope::NewRootScope();

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
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot2d) {
  Scope root = Scope::NewRootScope();

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
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot3d) {
  Scope root = Scope::NewRootScope();

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
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(ArrayOps, OneHot3dNegAxis) {
  Scope root = Scope::NewRootScope();

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
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "OneHot", sess_run_fetchoutputs);

  opexecuter.RunTest();

}  // end of op OneHot

TEST(ArrayOps, Pad) {
  Scope root = Scope::NewRootScope();

  Tensor input(DT_INT32, TensorShape({2, 3}));

  Tensor paddings(DT_INT32, TensorShape({2, 2}));

  AssignInputValuesRandom<int>(input, 1, 4);
  AssignInputValuesRandom<int>(paddings, 2, 5);

  auto R = ops::Pad(root, input, paddings);
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "Pad", sess_run_fetchoutputs);

  opexecuter.RunTest();

}  // end of op Pad

// Test op: Rank Op
TEST(ArrayOps, Rank) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;
  int dim3 = 3;

  Tensor A(DT_INT32, TensorShape({dim1, dim2, dim3}));
  AssignInputValuesRandom<int32>(A, 2, 20);

  auto R = ops::Rank(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Rank", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of RankOp

// Test op: Shape, outputs the shape of a tensor
TEST(ArrayOps, Shape2D) {
  Scope root = Scope::NewRootScope();

  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 7.5f);

  auto attrs = ops::Shape::Attrs();
  attrs.out_type_ = DT_INT64;
  auto R = ops::Shape(root, A, attrs);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Shape", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of op Shape2D

TEST(ArrayOps, Shape3D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;
  int dim3 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2, dim3}));

  AssignInputValues(A, 7.5f);

  auto R = ops::Shape(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Shape", sess_run_fetchoutputs);

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

  auto attrs = ops::Size::Attrs();
  attrs.out_type_ = DT_INT64;

  for (auto const& shapes : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shapes));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);

    auto R = ops::Size(root, input_data, attrs);
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Size", sess_run_fetchoutputs);

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

  // Size Op default output tyep is DT_INT32

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    auto R = ops::Size(root, input_data);

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Size", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SizeDefault

// Test slice op
TEST(ArrayOps, Slice) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 4, 1});

  std::vector<int64> begin = {0, 0, 2, 0};
  std::vector<int64> size = {-1, -1, 2, -1};

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
    OpExecuter opexecuter(root, "Slice", sess_run_fetchoutputs);

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

  map<std::vector<int64>, int>::iterator iter;
  for (iter = input_map.begin(); iter != input_map.end(); iter++) {
    std::vector<int64> shape = iter->first;
    int block_size = iter->second;

    Scope root = Scope::NewRootScope();
    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);

    auto R = ops::SpaceToDepth(root, input_data, block_size);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "SpaceToDepth", sess_run_fetchoutputs);
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
    OpExecuter opexecuter(root, "SpaceToDepth", sess_run_fetchoutputs);

    vector<Tensor> ngraph_outputs;
    opexecuter.ExecuteOnNGraph(ngraph_outputs);

    // On CPU, the op only supports NHWC data format
    Scope tf_scope = Scope::NewRootScope();
    auto input_data_NHWC = ops::Transpose(tf_scope, input_data, {0, 2, 3, 1});
    auto r_tf = ops::SpaceToDepth(tf_scope, input_data_NHWC, block_size);
    auto r_tf_NCHW = ops::Transpose(tf_scope, r_tf, {0, 3, 1, 2});
    vector<Output> sess_run_fetchoutputs_tf = {r_tf_NCHW};
    OpExecuter opexecuter_tf(tf_scope, "SpaceToDepth",
                             sess_run_fetchoutputs_tf);

    vector<Tensor> tf_outputs;
    opexecuter_tf.ExecuteOnTF(tf_outputs);

    // Compare NGraph and TF Outputs
    Compare(tf_outputs, ngraph_outputs);
  }
}  // end of op SpaceToDepthNCHW

// Test op: StridedSlice
// In this test the begin, end and stride vectors have length < rank
TEST(ArrayOps, StridedSliceTest1) {
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
  std::vector<int64> cend = {1, 2};
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
  std::vector<Output> sess_run_fetchoutputs = {R};

  OpExecuter opexecuter(root, "StridedSlice", sess_run_fetchoutputs);

  opexecuter.RunTest();
}

// Test SplitNegativeAxis op
TEST(ArrayOps, SplitNegativeAxis) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 8, 1});
  // num_split : The number of ways to split. Must evenly divide
  // value.shape[split_dim]
  int64_t num_splits = 4;

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
    OpExecuter opexecuter(root, "Split", sess_run_fetchoutputs);

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
    OpExecuter opexecuter(root, "Split", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitPositiveAxis

// Test SplitVNegSizeSplit op
TEST(ArrayOps, SplitVNegSizeSplit) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 6, 1});

  std::vector<int64> size_splits = {2, -1, 2, 1};
  int64_t num_splits = 4;

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
    OpExecuter opexecuter(root, "SplitV", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVNegSizeSplit

// Test SplitVNegativeAxis op
TEST(ArrayOps, SplitVNegativeAxis) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 6, 1});

  std::vector<int64> size_splits = {2, 1, 2, 1};
  int64_t num_splits = 4;

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
    OpExecuter opexecuter(root, "SplitV", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVNegativeAxis

// Test SplitVPositiveSizeSplits op
TEST(ArrayOps, SplitVPositiveSizeSplits) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 2, 6, 1});

  std::vector<int64> size_splits = {2, 1, 2, 1};
  int64_t num_splits = 4;

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
    OpExecuter opexecuter(root, "SplitV", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVPositiveSizeSplits

// Test SplitVZeroSizeSplit op
// fails with opset3 upgrade on CPU/Interpreter
TEST(ArrayOps, SplitVZeroSizeSplit) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 10});

  std::vector<int64> size_splits = {10, 0};
  int64_t num_splits = 2;

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
    OpExecuter opexecuter(root, "SplitV", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVZeroSizeSplit

// Test SplitVZeroSizeNegSplit op
// fails with opset3 upgrade on CPU/Interpreter
TEST(ArrayOps, SplitVZeroSizeNegSplit) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1, 10});

  std::vector<int64> size_splits = {10, -1};
  int64_t num_splits = 2;

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
    OpExecuter opexecuter(root, "SplitV", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SplitVZeroSizeNegSplit

// Test op: Tile, constructs a tensor by tiling a given tensor
TEST(ArrayOps, Tile) {
  std::vector<std::vector<int64>> input_sizes;  // 1-D or higher

  input_sizes.push_back({2, 3, 4});
  input_sizes.push_back({10, 10, 10});
  input_sizes.push_back({0});

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
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Tile", sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

TEST(ArrayOps, Tile2) {  // Not working with OV GPU
  std::vector<int64> input_size{1, 3};
  Tensor input_data(DT_FLOAT, TensorShape(input_size));
  AssignInputValues(input_data, 2.1f);

  // Must be of type int32 or int64,
  // 1-D. Length must be the same as the number of dimensions in input
  int input_dim = input_size.size();
  Tensor multiples(DT_INT32, TensorShape({input_dim}));
  AssignInputValues(multiples, vector<int>{1, 2});

  Scope root = Scope::NewRootScope();
  auto R = ops::Tile(root, input_data, multiples);
  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Tile", sess_run_fetchoutputs);
  opexecuter.RunTest();
}
// end of test op Tile

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

  auto R = ops::Transpose(root, A, perm);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Transpose", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Transpose

// Test op: Transpose With Constant input and empty permuation vector
TEST(ArrayOps, TransposeConstant) {
  Scope root = Scope::NewRootScope();

  auto A = ops::Const(root, 12.0f);
  auto perm = ops::Const(root, std::initializer_list<int>{});
  auto R = ops::Transpose(root, A, perm);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Transpose", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op Transpose

// Unpacks the given dimension of a rank R tensor into a (R-1) tensor
TEST(ArrayOps, Unpack) {
  std::vector<std::vector<int64>> input_sizes;

  // rank > 0
  input_sizes.push_back({3, 2, 3});
  input_sizes.push_back({4, 3});
  // input_sizes.push_back({3});

  std::vector<int64> axes({0, 1, 0});

  for (size_t i = 0; i < input_sizes.size(); ++i) {
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

    OpExecuter opexecuter(root, "Unpack", sess_run_fetchoutputs);

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

  auto R = ops::ZerosLike(root, A);

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ZerosLike", sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op ZerosLike

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
