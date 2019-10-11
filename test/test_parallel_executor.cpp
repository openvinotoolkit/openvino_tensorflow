/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include <memory>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_executor.h"
#include "ngraph_bridge/version.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

Status LoadGraphFromPbTxt(const string& pb_file,
                          unique_ptr<tf::Graph>& new_graph) {
  // Read the graph
  tensorflow::GraphDef graph_def;
  auto load_graph_status = ReadTextProto(Env::Default(), pb_file, &graph_def);
  if (!load_graph_status.ok()) {
    return load_graph_status;
  }

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  unique_ptr<tf::Graph> input_graph =
      unique_ptr<tf::Graph>(new tf::Graph(OpRegistry::Global()));

  auto status = ConvertGraphDefToGraph(opts, graph_def, input_graph.get());
  new_graph = move(input_graph);
  return status;
}

TEST(ParallelExecutor, Construction) {
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  unique_ptr<tf::Graph> input_graph;

  // First test with a backend not yet created
  unique_ptr<NGraphExecutor> executor;
  ASSERT_THROW(executor = unique_ptr<NGraphExecutor>(
                   new NGraphExecutor(100, 500, 600, input_graph, "bogus")),
               std::runtime_error);

  // Next test with a null graph not yet created
  ASSERT_THROW(executor = unique_ptr<NGraphExecutor>(
                   new NGraphExecutor(100, 500, 600, input_graph, "bogus")),
               std::runtime_error);

  // Now read the graph
  ASSERT_OK(LoadGraphFromPbTxt("test_axpy_launchop.pbtxt", input_graph));

  // Next test with a backend after creating
  tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");
  ASSERT_NO_THROW(executor = unique_ptr<NGraphExecutor>(new NGraphExecutor(
                      100, 500, 600, input_graph, "INTERPRETER")));

  // Now that the object has been cobstructed, test various internal parts
  // TODO: Create a Test Class and mark that as a friend of the Executor class
  ASSERT_EQ(executor->GetOpBackendName(), "INTERPRETER");
  ASSERT_TRUE(executor->IsTensorPipeliningSupported());
}

TEST(ParallelExecutor, CompilerTest) {
  // Read the graph
  unique_ptr<tf::Graph> input_graph;

  // We are using a graph with _Arg and _Retval
  // addded i.e., a PB that is saved after the initial processing of the
  // TF graph transformation.
  ASSERT_OK(LoadGraphFromPbTxt("test_axpy_launchop.pbtxt", input_graph));

  tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");
  NGraphExecutor executor(100, 500, 600, input_graph, "INTERPRETER");

  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  Tensor y(DT_FLOAT, TensorShape({2, 3}));
  auto y_flat = y.flat<float>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1.0;
  }

  std::vector<Tensor> tf_input_tensors{x, y};
  shared_ptr<ngraph::runtime::Executable> ng_exec;

  // Call the Executor to compile the funcion
  bool cache_hit = false;
  ASSERT_OK(executor.GetNgExecutable(tf_input_tensors, ng_exec, cache_hit));
  ASSERT_FALSE(cache_hit);

  // Now call again to test that the cache works
  ASSERT_OK(executor.GetNgExecutable(tf_input_tensors, ng_exec, cache_hit));

  // If the cache doesn't work then the following will fire
  ASSERT_TRUE(cache_hit);

  // Now validate that the nGraph function is available
  std::shared_ptr<ngraph::Function> ng_function;
  ASSERT_EQ(executor.GetNgFunction(ng_exec, ng_function),
            tensorflow::Status::OK());

  // Validate the nGraph Function
  const auto& parameters = ng_function->get_parameters();
  ASSERT_EQ(2, parameters.size());
}

TEST(ParallelExecutor, PipelinedTensorCreate) {
  // Read the graph
  // We are using a graph with _Arg and _Retval
  // addded i.e., a PB that is saved after the initial processing of the
  // TF graph transformation.
  unique_ptr<tf::Graph> input_graph;
  ASSERT_OK(LoadGraphFromPbTxt("test_axpy_launchop.pbtxt", input_graph));
  tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");
  NGraphExecutor executor(100, 500, 600, input_graph, "INTERPRETER");

  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  Tensor y(DT_FLOAT, TensorShape({2, 3}));
  auto y_flat = y.flat<float>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1.0;
  }

  std::vector<Tensor> tf_input_tensors{x, y};
  shared_ptr<ngraph::runtime::Executable> ng_exec;

  // Call the Executor to compile the funcion
  bool cache_hit = false;
  ASSERT_OK(executor.GetNgExecutable(tf_input_tensors, ng_exec, cache_hit));
  ASSERT_FALSE(cache_hit);
  ASSERT_EQ(2, executor.GetTensorPipelineDepth());

  // Get the pipelned tensors
  int pipeline_idx = -1;
  std::tuple<int, PipelinedTensorVector, PipelinedTensorVector> io_tensors;
  for (int i = 0; i < executor.GetTensorPipelineDepth(); i++) {
    ASSERT_OK(executor.GetTensorsFromPipeline(ng_exec, io_tensors));
    pipeline_idx = get<0>(io_tensors);
    ASSERT_EQ(i, pipeline_idx) << "GetTensorsFromPipeline() Returned: "
                               << pipeline_idx;
  }

  // Now we have exhausted all the tensors. So the next call fails
  ASSERT_NOT_OK(executor.GetTensorsFromPipeline(ng_exec, io_tensors));
}

TEST(ParallelExecutor, ExecuteOnSingleThread) {
  // Read the graph
  // We are using a graph with _Arg and _Retval
  // addded i.e., a PB that is saved after the initial processing of the
  // TF graph transformation.
  unique_ptr<tf::Graph> input_graph;
  ASSERT_OK(LoadGraphFromPbTxt("test_axpy_launchop.pbtxt", input_graph));
  tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");
  NGraphExecutor executor(100, 500, 600, input_graph, "INTERPRETER");

  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  Tensor y(DT_FLOAT, TensorShape({2, 3}));

  std::vector<Tensor> tf_input_tensors{x, y};
  shared_ptr<ngraph::runtime::Executable> ng_exec;

  // Call the Executor to compile the funcion
  bool cache_hit = false;
  ASSERT_OK(executor.GetNgExecutable(tf_input_tensors, ng_exec, cache_hit));
  ASSERT_FALSE(cache_hit);

  ASSERT_EQ(2, executor.GetTensorPipelineDepth());

  // Get the pipelned tensors
  std::tuple<int, PipelinedTensorVector, PipelinedTensorVector> io_tensors;
  ASSERT_OK(executor.GetTensorsFromPipeline(ng_exec, io_tensors));

  // Now Fill in the tensor - X
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  ng::element::Type ng_element_type;
  ASSERT_OK(TFDataTypeToNGraphElementType(x.dtype(), &ng_element_type));

  get<1>(io_tensors)[0]->write(
      &x_flat.data()[0], 0,
      get<1>(io_tensors)[0]->get_element_count() * ng_element_type.size());

  // Now Fill in the tensor - Y
  auto y_flat = y.flat<float>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1.0;
  }

  ASSERT_OK(TFDataTypeToNGraphElementType(y.dtype(), &ng_element_type));

  get<1>(io_tensors)[1]->write(
      &y_flat.data()[0], 0,
      get<1>(io_tensors)[1]->get_element_count() * ng_element_type.size());

  // Output
  vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;
  for (size_t i = 0; i < ng_exec->get_results().size(); i++) {
    ng_outputs.push_back(get<2>(io_tensors)[i]);
  }

  // And execute
  ng_exec->call(ng_outputs, {get<1>(io_tensors)[0], get<1>(io_tensors)[1]});

  // Pick up the output
  vector<tf::Tensor> ngraph_outputs;
  // Convert to tf tensor
  Tensor tf_output_tensor(DT_FLOAT, TensorShape({2, 3}));
  void* dst_ptr = DMAHelper::base(&tf_output_tensor);
  ng_outputs[0]->read(dst_ptr, 0, tf_output_tensor.TotalBytes());

  // And validate
  // z = a * x + y
  //   a ==> 5.0
  // TODO
  Tensor expected_val(DT_FLOAT, TensorShape({2, 3}));
  AssignInputValues(expected_val, 6.0f);
  Compare(tf_output_tensor, expected_val, 0.0f);
}

TEST(ParallelExecutor, ExecuteOnSingleThread8Bit) {
  // Read the graph
  // We are using a graph with _Arg and _Retval
  // addded i.e., a PB that is saved after the initial processing of the
  // TF graph transformation.
  unique_ptr<tf::Graph> input_graph;
  ASSERT_OK(LoadGraphFromPbTxt("test_axpy_int8_launchop.pbtxt", input_graph));

  string backend_name = "INTERPRETER";
  if (std::getenv("NGRAPH_TF_BACKEND") != nullptr) {
    backend_name = std::getenv("NGRAPH_TF_BACKEND");
  }

  tf::ngraph_bridge::BackendManager::CreateBackend(backend_name);
  NGraphExecutor executor(100, 500, 600, input_graph, backend_name);

  // Create the inputs for this graph
  Tensor x(DT_INT8, TensorShape({2, 2}));
  Tensor y(DT_INT8, TensorShape({2, 2}));

  std::vector<Tensor> tf_input_tensors{x, y};
  shared_ptr<ngraph::runtime::Executable> ng_exec;

  // Call the Executor to compile the funcion
  bool cache_hit = false;
  ASSERT_OK(executor.GetNgExecutable(tf_input_tensors, ng_exec, cache_hit));
  ASSERT_FALSE(cache_hit);

  ASSERT_EQ(2, executor.GetTensorPipelineDepth());

  // Get the pipelned tensors
  std::tuple<int, PipelinedTensorVector, PipelinedTensorVector> io_tensors;
  ASSERT_OK(executor.GetTensorsFromPipeline(ng_exec, io_tensors));

  // Now Fill in the tensor - X
  auto x_flat = x.flat<int8>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1;
  }

  ng::element::Type ng_element_type;
  ASSERT_OK(TFDataTypeToNGraphElementType(x.dtype(), &ng_element_type));

  get<1>(io_tensors)[0]->write(
      &x_flat.data()[0], 0,
      get<1>(io_tensors)[0]->get_element_count() * ng_element_type.size());

  // Now Fill in the tensor - Y
  auto y_flat = y.flat<int8>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1;
  }

  ASSERT_OK(TFDataTypeToNGraphElementType(y.dtype(), &ng_element_type));

  get<1>(io_tensors)[1]->write(
      &y_flat.data()[0], 0,
      get<1>(io_tensors)[1]->get_element_count() * ng_element_type.size());

  // Output
  vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;
  for (size_t i = 0; i < ng_exec->get_results().size(); i++) {
    ng_outputs.push_back(get<2>(io_tensors)[i]);
  }

  // And execute
  ng_exec->call(ng_outputs, {get<1>(io_tensors)[0], get<1>(io_tensors)[1]});

  // Pick up the output
  vector<tf::Tensor> ngraph_outputs;
  // Convert to tf tensor
  Tensor tf_output_tensor(DT_INT8, TensorShape({2, 2}));
  void* dst_ptr = DMAHelper::base(&tf_output_tensor);
  ng_outputs[0]->read(dst_ptr, 0, tf_output_tensor.TotalBytes());

  // And validate
  // z = a * x + y
  //   a ==> 5
  // TODO
  Tensor expected_val(DT_INT8, TensorShape({2, 2}));
  AssignInputValues(expected_val, (int8)6);
  // Compare(tf_output_tensor, expected_val, 0);
}

TEST(ParallelExecutor, ExecuteOnMultipleThreads8Bit) {
  // Read the graph
  // We are using a graph with _Arg and _Retval
  // addded i.e., a PB that is saved after the initial processing of the
  // TF graph transformation.
  unique_ptr<tf::Graph> input_graph;
  ASSERT_OK(LoadGraphFromPbTxt("test_axpy_int8_launchop.pbtxt", input_graph));

  string backend_name = "INTERPRETER";
  if (std::getenv("NGRAPH_TF_BACKEND") != nullptr) {
    backend_name = std::getenv("NGRAPH_TF_BACKEND");
  }

  tf::ngraph_bridge::BackendManager::CreateBackend(backend_name);
  NGraphExecutor executor(100, 500, 600, input_graph, backend_name);

  // Create the inputs for this graph
  Tensor x(DT_INT8, TensorShape({2, 2}));
  // Now Fill in the tensor - X
  auto x_flat = x.flat<int8>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1;
  }

  Tensor y(DT_INT8, TensorShape({2, 2}));

  std::vector<Tensor> tf_input_tensors{x, y};
  shared_ptr<ngraph::runtime::Executable> ng_exec;

  // Call the Executor to compile the funcion
  bool cache_hit = false;
  ASSERT_OK(executor.GetNgExecutable(tf_input_tensors, ng_exec, cache_hit));
  // ASSERT_FALSE(cache_hit);

  auto worker = [&](int8 worker_id) {

    ASSERT_EQ(2, executor.GetTensorPipelineDepth());

    // Get the pipelned tensors
    std::tuple<int, PipelinedTensorVector, PipelinedTensorVector> io_tensors;
    ASSERT_OK(executor.GetTensorsFromPipeline(ng_exec, io_tensors));

    cout << "PTS Index: " << get<0>(io_tensors) << endl;

    ng::element::Type ng_element_type;
    ASSERT_OK(TFDataTypeToNGraphElementType(x.dtype(), &ng_element_type));
    get<1>(io_tensors)[0]->write(
        &x_flat.data()[0], 0,
        get<1>(io_tensors)[0]->get_element_count() * ng_element_type.size());

    // Now Fill in the tensor - Y
    Tensor y_thread(DT_INT8, TensorShape({2, 2}));
    auto y_flat = y_thread.flat<int8>();
    for (int i = 0; i < y_flat.size(); i++) {
      y_flat.data()[i] = worker_id;
    }

    ASSERT_OK(
        TFDataTypeToNGraphElementType(y_thread.dtype(), &ng_element_type));
    get<1>(io_tensors)[1]->write(
        &y_flat.data()[0], 0,
        get<1>(io_tensors)[1]->get_element_count() * ng_element_type.size());

    // Output
    vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;
    for (size_t i = 0; i < ng_exec->get_results().size(); i++) {
      ng_outputs.push_back(get<2>(io_tensors)[i]);
    }

    // And execute
    ng_exec->call(ng_outputs, {get<1>(io_tensors)[0], get<1>(io_tensors)[1]});

    // Convert to tf tensor
    Tensor tf_output_tensor(DT_INT8, TensorShape({2, 2}));
    void* dst_ptr = DMAHelper::base(&tf_output_tensor);
    ng_outputs[0]->read(dst_ptr, 0, tf_output_tensor.TotalBytes());

    // And validate
    // z = a * x + y
    //   a ==> 5
    // TODO
    // Tensor expected_val(DT_INT8, TensorShape({2, 2}));
    // AssignInputValues(expected_val, (int8)6);
    // Compare<int8>(tf_output_tensor, 5+worker_id);
    cout << tf_output_tensor.DebugString() << endl;
  };

  std::thread thread0(worker, 0);
  std::thread thread1(worker, 1);
  // std::thread thread2(worker, 33);

  thread0.join();
  thread1.join();
  // thread2.join();
}

TEST(ParallelExecutor, ExecuteOnMultipleThreads) {
  // Read the graph
  // We are using a graph with _Arg and _Retval
  // addded i.e., a PB that is saved after the initial processing of the
  // TF graph transformation.
  unique_ptr<tf::Graph> input_graph;
  ASSERT_OK(LoadGraphFromPbTxt("test_axpy_launchop.pbtxt", input_graph));
  tf::ngraph_bridge::BackendManager::CreateBackend("INTERPRETER");
  NGraphExecutor executor(100, 500, 600, input_graph, "INTERPRETER");

  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  Tensor y(DT_FLOAT, TensorShape({2, 3}));

  std::vector<Tensor> tf_input_tensors{x, y};
  shared_ptr<ngraph::runtime::Executable> ng_exec;

  // Call the Executor to compile the funcion
  bool cache_hit = false;
  ASSERT_OK(executor.GetNgExecutable(tf_input_tensors, ng_exec, cache_hit));
  ASSERT_FALSE(cache_hit);
  ASSERT_EQ(2, executor.GetTensorPipelineDepth());

  // Now Fill in the tensor - X
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  ng::element::Type ng_element_type;
  ASSERT_OK(TFDataTypeToNGraphElementType(x.dtype(), &ng_element_type));

  auto worker = [&](size_t worker_id) {
    // Get the pipelned tensors
    std::tuple<int, PipelinedTensorVector, PipelinedTensorVector> io_tensors;
    ASSERT_OK(executor.GetTensorsFromPipeline(ng_exec, io_tensors));

    // Copy the tensors from TensorFlow Tensor to nGraph Tensor
    // First X
    get<1>(io_tensors)[0]->write(
        &x_flat.data()[0], 0,
        get<1>(io_tensors)[0]->get_element_count() * ng_element_type.size());

    // Fill in the tensor - Y
    auto y_flat = y.flat<float>();
    for (int i = 0; i < y_flat.size(); i++) {
      y_flat.data()[i] = worker_id;
    }
    ASSERT_OK(TFDataTypeToNGraphElementType(y.dtype(), &ng_element_type));

    // Copy the tensors from TensorFlow Tensor to nGraph Tensor
    // Next Y
    get<1>(io_tensors)[1]->write(
        &y_flat.data()[0], 0,
        get<1>(io_tensors)[1]->get_element_count() * ng_element_type.size());

    // Output
    vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;
    for (size_t i = 0; i < ng_exec->get_results().size(); i++) {
      ng_outputs.push_back(get<2>(io_tensors)[i]);
    }

    // And execute
    ng_exec->call(ng_outputs, {get<1>(io_tensors)[0], get<1>(io_tensors)[1]});

    // Pick up the output
    vector<tf::Tensor> ngraph_outputs;
    // Convert to tf tensor
    Tensor tf_output_tensor(DT_FLOAT, TensorShape({2, 3}));
    void* dst_ptr = DMAHelper::base(&tf_output_tensor);
    ng_outputs[0]->read(dst_ptr, 0, tf_output_tensor.TotalBytes());

    // And validate
    // z = a * x + y
    //   a ==> 5.0
    // TODO
    Tensor expected_val(DT_FLOAT, TensorShape({2, 3}));
    AssignInputValues(expected_val, 5.0f + worker_id);
    Compare(tf_output_tensor, expected_val, 0.0f);
  };

  std::thread thread0(worker, 0);
  std::thread thread1(worker, 1);

  thread0.join();
  thread1.join();
}

TEST(ParallelExecutor, E2E8Bit) {
  string graph_name = "test_axpy_8bit.pbtxt";

  string backend_name = "INTERPRETER";
  if (std::getenv("NGRAPH_TF_BACKEND") != nullptr) {
    backend_name = std::getenv("NGRAPH_TF_BACKEND");
  }

  unique_ptr<Session> session;
  ASSERT_OK(CreateSession(graph_name, backend_name, session));

  string inp_tensor_name_0{"x"};
  string inp_tensor_name_1{"y"};
  string out_tensor_name{"add_ngraph/_1"};

  Tensor inp_tensor_x_val(tensorflow::DT_INT8, tensorflow::TensorShape({2, 2}));
  AssignInputValues<int8>(inp_tensor_x_val, 1);

  auto worker = [&](int8 worker_id) {
    Tensor inp_tensor_y_val(tensorflow::DT_INT8,
                            tensorflow::TensorShape({2, 2}));
    AssignInputValues<int8>(inp_tensor_y_val, worker_id);

    Tensor out_tensor_expected_val(tensorflow::DT_INT8,
                                   tensorflow::TensorShape({2, 2}));
    AssignInputValues<int8>(out_tensor_expected_val, 5 + worker_id);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        {inp_tensor_name_0, inp_tensor_x_val},
        {inp_tensor_name_1, inp_tensor_y_val}};

    std::vector<Tensor> out_tensor_vals;
    ASSERT_OK(session->Run(inputs, {out_tensor_name}, {}, &out_tensor_vals));
    cout << "Worker: " << (int)worker_id << " Expected: " << 5 + worker_id
         << endl;

    cout << "Worker: " << (int)worker_id
         << " Input: " << inputs[0].second.DebugString() << endl;

    cout << "Worker: " << (int)worker_id
         << " Input: " << inputs[1].second.DebugString() << endl;

    cout << "Worker: " << (int)worker_id
         << " Output: " << out_tensor_vals[0].DebugString() << endl;
    Compare<int8>(out_tensor_vals[0], out_tensor_expected_val);
  };

  std::thread thread0(worker, 1);
  std::thread thread1(worker, 2);
  std::thread thread2(worker, 3);

  thread0.join();
  thread1.join();
  thread2.join();

  session->Close();
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
