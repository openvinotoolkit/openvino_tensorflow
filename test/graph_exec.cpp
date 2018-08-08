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
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

TEST(graph_exec, axpy) {
  GraphDef gdef;
  // auto status = ReadTextProto(Env::Default(), "test_py.pbtxt",
  // &gdef);
  auto status =
      ReadTextProto(Env::Default(), "test_axpy_launchop.pbtxt", &gdef);
  // ReadTextProto(Env::Default(), "test_launch_op.pbtxt", &gdef);
  ASSERT_TRUE(status == Status::OK()) << "Can't read protobuf graph";

  Graph input_graph(OpRegistry::Global());

  GraphConstructorOptions opts;
  // Set the allow_internal_ops to true so that graphs with node names such as
  // _arg_Placeholder_1_0_1_0_arg are allowed. These op names are generated
  // during the graph rewrite passes and considered internal
  opts.allow_internal_ops = true;

  ASSERT_EQ(ConvertGraphDefToGraph(opts, gdef, &input_graph), Status::OK());
  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  Tensor y(DT_FLOAT, TensorShape({2, 3}));

  std::vector<TensorShape> inputs;
  inputs.push_back(x.shape());
  inputs.push_back(y.shape());

  shared_ptr<ng::Function> ng_function;
  ASSERT_EQ(Status::OK(), ngraph_bridge::Builder::TranslateGraph(
                              inputs, &input_graph, ng_function));

  // Create the nGraph backend
  auto backend = ng::runtime::Backend::create("CPU");

  // Allocate tensors for arguments a, b, c
  ng::Shape ng_shape_x(x.shape().dims());
  for (int i = 0; i < x.shape().dims(); ++i) {
    ng_shape_x[i] = x.shape().dim_size(i);
  }

  ng::Shape ng_shape_y(y.shape().dims());
  for (int i = 0; i < y.shape().dims(); ++i) {
    ng_shape_y[i] = y.shape().dim_size(i);
  }

  auto t_x = backend->create_tensor(ng::element::f32, ng_shape_x);
  float v_x[2][3] = {{1, 1, 1}, {1, 1, 1}};
  t_x->write(&v_x, 0, sizeof(v_x));

  auto t_y = backend->create_tensor(ng::element::f32, ng_shape_y);
  t_y->write(&v_x, 0, sizeof(v_x));

  // Allocate tensor for the result(s)
  vector<shared_ptr<ng::runtime::TensorView>> outputs;
  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    auto shape = ng_function->get_output_shape(i);
    auto elem_type = ng_function->get_output_element_type(i);
    auto t_result = backend->create_tensor(elem_type, shape);
    outputs.push_back(t_result);
  }

  // Execute the nGraph function.
  cout << "Calling nGraph function\n";
  backend->call(ng_function, outputs, {t_x, t_y});

  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor(cout, ng_function->get_output_op(i)->get_name(), outputs[i]);
    cout << endl;
  }
  // Add the validation logic
  // TODO
}

}  // namespace ngraph_bridge

}  // namespace tensorflwo
