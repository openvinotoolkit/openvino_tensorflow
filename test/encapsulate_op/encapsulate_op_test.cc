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
#include "tensorflow/core/graph/node_builder.h"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_encapsulate_impl.h"
#include "ngraph_bridge/ngraph_encapsulate_op.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

// Test: Create tensorflow input tensors and Compute Signature
TEST(EncapsulateOp, ComputeSignature) {
  NGraphEncapsulateImpl ng_encap_impl;

  std::vector<tensorflow::TensorShape> input_shapes;
  std::vector<tensorflow::Tensor> input_tensors;
  std::vector<const Tensor*> static_input_map;
  input_shapes.push_back({0});
  input_shapes.push_back({2});
  input_shapes.push_back({6, 10});
  input_shapes.push_back({10, 10, 10});

  // Create tensorflow tensors
  for (auto const& shapes : input_shapes) {
    Tensor input_data(DT_FLOAT, TensorShape(shapes));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);
    input_tensors.push_back(input_data);
  }
  int size = 4;
  ng_encap_impl.ResizeStaticInputVector(size);

  for (int i = 0; i < size; i++) {
    ng_encap_impl.SetStaticInputVector(i, false);
  }

  for (int i = 0; i < input_tensors.size(); i++) {
    const Tensor& input_tensor = input_tensors[i];
    if (ng_encap_impl.GetStaticInputVector()[i]) {
      static_input_map[i] = &input_tensor;
    }
  }
  std::stringstream signature_ss;
  ASSERT_OK(ng_encap_impl.ComputeSignature(input_tensors, input_shapes,
                                           static_input_map, signature_ss));
  ASSERT_EQ(signature_ss.str(), "0,;2,;6,10,;10,10,10,;/");
}

// Test: Create backend and get ngraph executable
TEST(EncapsulateOp, GetNgExecutable) {
  NGraphEncapsulateImpl ng_encap_impl;

  std::vector<tensorflow::TensorShape> input_shapes;
  std::vector<tensorflow::Tensor> input_tensors;
  std::vector<const Tensor*> static_input_map;
  input_shapes.push_back({0});
  input_shapes.push_back({2});
  input_shapes.push_back({6, 10});
  input_shapes.push_back({10, 10, 10});

  for (auto const& shapes : input_shapes) {
    Tensor input_data(DT_FLOAT, TensorShape(shapes));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);
    input_tensors.push_back(input_data);
  }
  int size = 4;
  ng_encap_impl.ResizeStaticInputVector(size);

  for (int i = 0; i < size; i++) {
    ng_encap_impl.SetStaticInputVector(i, false);
  }

  for (int i = 0; i < input_tensors.size(); i++) {
    const Tensor& input_tensor = input_tensors[i];
    if (ng_encap_impl.GetStaticInputVector()[i]) {
      static_input_map[i] = &input_tensor;
    }
  }

  ng_encap_impl.SetOpBackend("CPU");
  ASSERT_OK(BackendManager::CreateBackend(ng_encap_impl.GetOpBackend()));
  ng::runtime::Backend* op_backend;
  op_backend = BackendManager::GetBackend(ng_encap_impl.GetOpBackend());

  std::shared_ptr<ngraph::runtime::Executable> ng_exec;

  ASSERT_OK(ng_encap_impl.GetNgExecutable(
      input_tensors, input_shapes, static_input_map, op_backend, ng_exec));

  BackendManager::ReleaseBackend("CPU");
}

// Test: Allocating ngraph input tensors
TEST(EncapsulateOp, AllocateNGInputTensors) {
  NGraphEncapsulateImpl ng_encap_impl;
  ng::Shape shape{100};
  auto A = make_shared<ng::op::Parameter>(ng::element::f32, shape);
  auto B = make_shared<ng::op::Parameter>(ng::element::f32, shape);
  auto f = make_shared<ng::Function>(make_shared<ng::op::Add>(A, B),
                                     ng::ParameterVector{A, B});

  ng_encap_impl.SetOpBackend("CPU");
  ASSERT_OK(BackendManager::CreateBackend(ng_encap_impl.GetOpBackend()));
  ng::runtime::Backend* op_backend;
  op_backend = BackendManager::GetBackend(ng_encap_impl.GetOpBackend());
  auto ng_exec = op_backend->compile(f);

  std::vector<tensorflow::TensorShape> input_shapes;
  std::vector<tensorflow::Tensor> input_tensors;
  input_shapes.push_back({0});
  input_shapes.push_back({2});
  input_shapes.push_back({6, 10});
  input_shapes.push_back({10, 10, 10});

  // create tensorflow tensors
  for (auto const& shapes : input_shapes) {
    Tensor input_data(DT_FLOAT, TensorShape(shapes));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);
    input_tensors.push_back(input_data);
  }

  std::vector<shared_ptr<ng::runtime::Tensor>> ng_inputs;

  ASSERT_OK(ng_encap_impl.AllocateNGInputTensors(input_tensors, ng_exec, {},
                                                 op_backend, ng_inputs));
  BackendManager::ReleaseBackend("CPU");
}

// Test: Allocating ngraph output tensors
TEST(EncapsulateOp, AllocateNGOutputTensors) {
  NGraphEncapsulateImpl ng_encap_impl;
  ng::Shape shape{100};
  auto A = make_shared<ng::op::Parameter>(ng::element::f32, shape);
  auto B = make_shared<ng::op::Parameter>(ng::element::f32, shape);
  auto f = make_shared<ng::Function>(make_shared<ng::op::Add>(A, B),
                                     ng::ParameterVector{A, B});

  ng_encap_impl.SetOpBackend("CPU");
  ASSERT_OK(BackendManager::CreateBackend(ng_encap_impl.GetOpBackend()));
  ng::runtime::Backend* op_backend;
  op_backend = BackendManager::GetBackend(ng_encap_impl.GetOpBackend());
  auto ng_exec = op_backend->compile(f);

  std::vector<tensorflow::TensorShape> input_shapes;
  std::vector<tensorflow::Tensor> outputs;
  input_shapes.push_back({0});
  input_shapes.push_back({2});
  input_shapes.push_back({6, 10});
  input_shapes.push_back({10, 10, 10});

  // Create tensorflow tensors
  for (auto const& shapes : input_shapes) {
    Tensor input_data(DT_FLOAT, TensorShape(shapes));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);
    outputs.push_back(input_data);
  }

  std::vector<tensorflow::Tensor*> output_tensors;
  for (int i = 0; i < outputs.size(); i++) {
    Tensor* output_tensor = &outputs[i];
    output_tensors.push_back(output_tensor);
  }
  std::vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;

  ASSERT_OK(ng_encap_impl.AllocateNGOutputTensors(output_tensors, ng_exec, {},
                                                  op_backend, ng_outputs));

  BackendManager::ReleaseBackend("CPU");
}
}
}
}
