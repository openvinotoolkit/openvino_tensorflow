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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_executable.h"
#include "ngraph_bridge/ngraph_utils.h"

#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

class NGraphExecTest : public ::testing::Test {
 protected:
  // Loads the .pbtxt into a graph object
  Status LoadGraph(const string& graph_pbtxt_file, Graph* graph) {
    GraphDef gdef;
    TF_RETURN_IF_ERROR(ReadTextProto(Env::Default(), graph_pbtxt_file, &gdef));
    GraphConstructorOptions opts;

    // Set the allow_internal_ops to true so that graphs with node names such as
    // _arg_Placeholder_1_0_1_0_arg are allowed. These op names are generated
    // during the graph rewrite passes and considered internal
    opts.allow_internal_ops = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, gdef, graph));
    return Status::OK();
  }

  // Translates the TFGraph into NGFunction assumes no static inputs
  Status TranslateTFGraphNoStatic(const vector<TensorShape>& tf_input_shapes,
                                  const Graph& input_graph,
                                  shared_ptr<ngraph::Function>& ng_function) {
    // Translate the Graph: Create ng_function
    std::vector<const Tensor*> static_input_map(tf_input_shapes.size(),
                                                nullptr);
    TF_RETURN_IF_ERROR(ngraph_bridge::Builder::TranslateGraph(
        tf_input_shapes, static_input_map, &input_graph, ng_function));
    return Status::OK();
  }

  // Creates the tensor from backend of the same shape and data type as TF
  // Tensor
  // Copies data from TF Tensor to it
  Status CreateTensorFromBackend(
      const shared_ptr<Backend>& ng_backend, Tensor& tf_tensor,
      shared_ptr<ngraph::runtime::Tensor>& ng_tensor) {
    ng::element::Type ng_element_type;
    TF_RETURN_IF_ERROR(
        TFDataTypeToNGraphElementType(tf_tensor.dtype(), &ng_element_type));
    ng::Shape ng_shape(tf_tensor.shape().dims());
    for (int j = 0; j < tf_tensor.shape().dims(); ++j) {
      ng_shape[j] = tf_tensor.shape().dim_size(j);
    }
    TF_RETURN_IF_ERROR(CreateTensorFromBackend(ng_backend, ng_element_type,
                                               ng_shape, ng_tensor));
    WriteNGTensor(ng_tensor, &tf_tensor);
    return Status::OK();
  }

  // Creates the tensor from backend of the provided shape and data type
  Status CreateTensorFromBackend(
      const shared_ptr<Backend>& ng_backend,
      const ng::element::Type& ng_element_type, const ng::Shape& ng_shape,
      shared_ptr<ngraph::runtime::Tensor>& ng_tensor) {
    try {
      ng_tensor = ng_backend->create_tensor(ng_element_type, ng_shape);
    } catch (const std::exception& exp) {
      return errors::Internal(
          "Failed to create tensor from backend. Got ngraph exception ",
          exp.what());
    }
    return Status::OK();
  }

  // Executes the graph on TF
  Status RunGraphOnTF(const Graph& graph,
                      const vector<pair<string, Tensor>>& feed_dict,
                      const vector<string>& out_node_names,
                      vector<Tensor>& out_tensors) {
    // Create Session
    SessionOptions options;
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(tf::OptimizerOptions_Level_L0);
    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_constant_folding(tf::RewriterConfig::OFF);
    std::unique_ptr<Session> session(NewSession(options));

    // Attach Graph
    GraphDef gdef;
    graph.ToGraphDef(&gdef);
    TF_RETURN_IF_ERROR(session->Create(gdef));
    DeactivateNGraph();
    Status status = session->Run(feed_dict, out_node_names, {}, &out_tensors);
    ActivateNGraph();
    return status;
  }
};

TEST_F(NGraphExecTest, Axpy) {
  auto env_map = StoreEnv({"NGRAPH_TF_BACKEND"});
  SetBackendUsingEnvVar("CPU");

  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_axpy_launchop.pbtxt", &input_graph));

  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  Tensor y(DT_FLOAT, TensorShape({2, 3}));

  std::vector<TensorShape> input_shapes;
  input_shapes.push_back(x.shape());
  input_shapes.push_back(y.shape());

  shared_ptr<ng::Function> ng_function;
  ASSERT_OK(TranslateTFGraphNoStatic(input_shapes, input_graph, ng_function));

  // Create the nGraph backend
  auto backend = BackendManager::GetBackend();
  ASSERT_NE(backend, nullptr);

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
  t_x->write(&v_x, sizeof(v_x));

  auto t_y = backend->create_tensor(ng::element::f32, ng_shape_y);
  t_y->write(&v_x, sizeof(v_x));

  // Allocate tensor for the result(s)
  vector<shared_ptr<ng::runtime::Tensor>> outputs;
  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    auto shape = ng_function->get_output_shape(i);
    auto elem_type = ng_function->get_output_element_type(i);
    auto t_result = backend->create_tensor(elem_type, shape);
    outputs.push_back(t_result);
  }

  // Execute the nGraph function.
  cout << "Calling nGraph function\n";
  auto exec = backend->compile(ng_function);
  exec->call(outputs, {t_x, t_y});

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor<float>(cout, ng_function->get_output_op(i)->get_name(),
                        outputs[i]);
    cout << endl;
  }
  // Add the validation logic
  // TODO
  RestoreEnv(env_map);
}

TEST_F(NGraphExecTest, Axpy8bit) {
  auto env_map = StoreEnv({"NGRAPH_TF_BACKEND"});
  SetBackendUsingEnvVar("CPU");

  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_axpy_int8_launchop.pbtxt", &input_graph));

  // Create the inputs for this graph
  Tensor x(DT_INT8, TensorShape({2, 2}));
  Tensor y(DT_INT8, TensorShape({2, 2}));

  std::vector<TensorShape> input_shapes;
  input_shapes.push_back(x.shape());
  input_shapes.push_back(y.shape());

  shared_ptr<ng::Function> ng_function;
  ASSERT_OK(TranslateTFGraphNoStatic(input_shapes, input_graph, ng_function));

  // Create the nGraph backend
  auto backend = BackendManager::GetBackend();
  ASSERT_NE(backend, nullptr);

  // Allocate tensors for arguments a, b, c
  ng::Shape ng_shape_x(x.shape().dims());
  for (int i = 0; i < x.shape().dims(); ++i) {
    ng_shape_x[i] = x.shape().dim_size(i);
  }

  ng::Shape ng_shape_y(y.shape().dims());
  for (int i = 0; i < y.shape().dims(); ++i) {
    ng_shape_y[i] = y.shape().dim_size(i);
  }

  auto t_x = backend->create_tensor(ng::element::i8, ng_shape_x);
  int8 v_x[2][2] = {{1, 1}, {1, 1}};
  t_x->write(&v_x, sizeof(v_x));

  auto t_y = backend->create_tensor(ng::element::i8, ng_shape_y);
  t_y->write(&v_x, sizeof(v_x));

  // Allocate tensor for the result(s)
  vector<shared_ptr<ng::runtime::Tensor>> outputs;
  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    auto shape = ng_function->get_output_shape(i);
    auto elem_type = ng_function->get_output_element_type(i);
    auto t_result = backend->create_tensor(elem_type, shape);
    outputs.push_back(t_result);
  }

  // Execute the nGraph function.
  cout << "Calling nGraph function\n";
  auto exec = backend->compile(ng_function);
  exec->call(outputs, {t_x, t_y});

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor<int8>(cout, ng_function->get_output_op(i)->get_name(),
                       outputs[i]);
    cout << endl;
  }
  // Add the validation logic
  // TODO
  RestoreEnv(env_map);
}

TEST_F(NGraphExecTest, MixedTensors) {
  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_axpy_launchop.pbtxt", &input_graph));

  // Create the inputs for this graph
  DataType tf_dt = DT_FLOAT;
  TensorShape tf_shape = TensorShape({2, 3});
  int num_inputs = 2;

  // Run Graph on TF: Expected output
  Tensor x(tf_dt, tf_shape);
  Tensor y(tf_dt, tf_shape);
  AssignInputValues(x, 1.0f);
  AssignInputValues(y, 1.0f);
  std::vector<Tensor> tf_inputs = {x, y};
  vector<Tensor> expected_outputs;
  vector<pair<string, Tensor>> feed_dict = {{"x", x}, {"y", y}};
  vector<string> out_node_names = {"add", "mul"};
  ASSERT_OK(
      RunGraphOnTF(input_graph, feed_dict, out_node_names, expected_outputs));

  // Run on nGraph
  // Translate Graph
  std::vector<TensorShape> tf_input_shapes(num_inputs, tf_shape);
  shared_ptr<ng::Function> ng_function;
  ASSERT_OK(
      TranslateTFGraphNoStatic(tf_input_shapes, input_graph, ng_function));

  // Create the nGraph backend
  auto backend = BackendManager::GetBackend();
  ASSERT_NE(backend, nullptr);

  // Compile the nGraph function.
  auto exec = backend->compile(ng_function);

  // Allocate ng tensors for inputs
  vector<shared_ptr<ng::runtime::Tensor>> ng_inputs;
  for (int i = 0; i < 2; ++i) {
    shared_ptr<ng::runtime::Tensor> ng_input;
    if (i % 2 == 0) {
      ng_input = exec->create_input_tensor(i);
      WriteNGTensor(ng_input, &tf_inputs[i]);
    } else {
      ASSERT_OK(CreateTensorFromBackend(backend, tf_inputs[i], ng_input));
    }
    ng_inputs.push_back(ng_input);
  }

  // Allocate tensor for the result(s)
  vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    shared_ptr<ng::runtime::Tensor> ng_output;
    if (i % 2 == 0) {
      ng_output = exec->create_output_tensor(i);
    } else {
      auto shape = ng_function->get_output_shape(i);
      auto elem_type = ng_function->get_output_element_type(i);
      ASSERT_OK(CreateTensorFromBackend(backend, elem_type, shape, ng_output));
    }
    ng_outputs.push_back(ng_output);
  }

  // Execute the nGraph function.
  exec->call(ng_outputs, ng_inputs);

  // Actual Outputs
  // Allocating TF Tensors and reading into them to compare the outputs
  vector<Tensor> actual_outputs;

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    // Convert to tf tensor
    Tensor output_tensor(tf_dt, tf_shape);
    void* dst_ptr = DMAHelper::base(&output_tensor);
    ng_outputs[i]->read(dst_ptr, output_tensor.TotalBytes());
    actual_outputs.push_back(output_tensor);
  }

  // Comparing
  Compare(expected_outputs, actual_outputs);
}

TEST_F(NGraphExecTest, FindNumberOfNodesUtil1) {
  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_axpy_launchop.pbtxt", &input_graph));

  int number_of_args = FindNumberOfNodes(&input_graph, "_Arg");
  int number_of_retvals = FindNumberOfNodes(&input_graph, "_Retval");
  int number_of_const = FindNumberOfNodes(&input_graph, "Const");
  int number_of_xyz = FindNumberOfNodes(&input_graph, "XYZ");

  ASSERT_EQ(number_of_args, 2);
  ASSERT_EQ(number_of_retvals, 2);
  ASSERT_EQ(number_of_const, 1);
  ASSERT_EQ(number_of_xyz, 0);
}

TEST_F(NGraphExecTest, FindNumberOfNodesUtil2) {
  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_general_graph.pbtxt", &input_graph));

  int number_of_args = FindNumberOfNodes(&input_graph, "_Arg");
  int number_of_retvals = FindNumberOfNodes(&input_graph, "_Retval");
  int number_of_add = FindNumberOfNodes(&input_graph, "Add");
  int number_of_sub = FindNumberOfNodes(&input_graph, "Sub");

  ASSERT_EQ(number_of_args, 3);
  ASSERT_EQ(number_of_retvals, 3);
  ASSERT_EQ(number_of_add, 2);
  ASSERT_EQ(number_of_sub, 1);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
