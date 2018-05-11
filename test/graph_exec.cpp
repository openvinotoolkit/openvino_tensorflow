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

#include "ngraph_builder.h"
#include "ngraph_utils.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

using namespace std;
namespace tf = tensorflow;

namespace ngraph_bridge {

TEST(graph_exec, axpy) {
  tf::GraphDef gdef;
  // auto status = tf::ReadTextProto(tf::Env::Default(), "test_py.pbtxt",
  // &gdef);
  auto status =
      tf::ReadTextProto(tf::Env::Default(), "test_axpy_launchop.pbtxt", &gdef);
  ASSERT_TRUE(status == tf::Status::OK()) << "Can't read protobuf graph";

  tf::Graph input_graph(tf::OpRegistry::Global());

  tf::GraphConstructorOptions opts;
  // Set the allow_internal_ops to true so that graphs with node names such as
  // _arg_Placeholder_1_0_1_0_arg are allowed. These op names are generated
  // during the graph rewrite passes and considered internal
  opts.allow_internal_ops = true;

  ASSERT_EQ(tf::ConvertGraphDefToGraph(opts, gdef, &input_graph),
            tf::Status::OK());
  // Create the inputs for this graph
  // Inside TensorFlow execution, call this:
  // OpKernelContext->input(index).shape()
  auto ng_function = ngraph_bridge::Builder::TranslateGraph(&input_graph);
}
}  // namespace ngraph_bridge