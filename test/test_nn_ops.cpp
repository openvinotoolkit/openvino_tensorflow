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

TEST(NNOps, SparseSoftmaxCrossEntropyWithLogits) {
  Scope root = Scope::NewRootScope();
  int batch = 10;
  int num_of_classes = 2;

  Tensor A(DT_FLOAT, TensorShape({batch, num_of_classes}));
  Tensor B(DT_INT32, TensorShape({batch}));

  AssignInputValues(A, 2.0f);
  AssignInputIntValues(B, num_of_classes);

  vector<int> static_input_indexes = {};
  auto R = ops::SparseSoftmaxCrossEntropyWithLogits(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.loss, R.backprop};
  OpExecuter opexecuter(root, "SparseSoftmaxCrossEntropyWithLogits",
                        static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}

TEST(NNOps, Conv2DBackpropFilter_NHWC) {
  // TF Default formats
  // Input NHWC :[batch, in_height, in_width, in_channels]
  vector<int64> input_size_NHWC = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  // Out_delta :[batch, out_height, out_width, out_channels]
  vector<int64> output_del_size_valid = {1, 3, 2, 2};
  vector<int64> output_del_size_same = {1, 4, 3, 2};

  std::vector<int> stride = {1, 2, 2, 1};

  std::map<std::string, vector<int64>> out_delta_size_map = {
      {"VALID", output_del_size_valid}, {"SAME", output_del_size_same}};

  vector<int> static_input_indexes = {1};
  // TEST NHWC : default data format
  for (auto map_iterator : out_delta_size_map) {
    Scope root = Scope::NewRootScope();
    auto padding_type = map_iterator.first;
    auto output_delta_size = out_delta_size_map[padding_type];

    Tensor output_delta(DT_FLOAT, TensorShape(output_delta_size));
    AssignInputValues(output_delta, -1.1f);

    auto filter_sizes = ops::Const(root, {3, 3, 2, 2});

    Tensor input_data(DT_FLOAT, TensorShape(input_size_NHWC));
    AssignInputValues(input_data, -1.1f);

    auto R = ops::Conv2DBackpropFilter(root, input_data, filter_sizes,
                                       output_delta, stride, padding_type);

    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}

}  // namespace testing

}  // namespace ngraph_bridge
}  // namespace tensorflow
