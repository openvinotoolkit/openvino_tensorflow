//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/util.hpp"

#include "test/dummy_backend.h"

using namespace std;
namespace ng = ngraph;

namespace ngraph {

using descriptor::layout::DenseTensorLayout;
runtime::dummy::DummyBackend::DummyBackend() {}

shared_ptr<runtime::Tensor> runtime::dummy::DummyBackend::create_tensor(
    const ng::element::Type& type, const ng::Shape& shape) {
  return make_shared<runtime::HostTensor>(type, shape, "external");
}

shared_ptr<runtime::Tensor> runtime::dummy::DummyBackend::create_tensor(
    const ng::element::Type& type, const ng::Shape& shape,
    void* memory_pointer) {
  return make_shared<runtime::HostTensor>(type, shape, memory_pointer,
                                          "external");
}

shared_ptr<runtime::Executable> runtime::dummy::DummyBackend::compile(
    shared_ptr<ng::Function> function, bool enable_performance_collection) {
  return make_shared<DummyExecutable>(function, enable_performance_collection);
}

bool runtime::dummy::DummyBackend::is_supported(const Node& node) const {
  return false;
}

runtime::dummy::DummyBackend::~DummyBackend() {}

runtime::dummy::DummyExecutable::DummyExecutable(
    shared_ptr<ng::Function> function,
    bool /* enable_performance_collection */) {
  pass::Manager pass_manager;
  pass_manager.register_pass<pass::AssignLayout<DenseTensorLayout>>();
  pass_manager.run_passes(function);

  set_parameters_and_results(*function);
}

bool runtime::dummy::DummyExecutable::call(
    const vector<shared_ptr<runtime::Tensor>>& /* outputs */,
    const vector<shared_ptr<runtime::Tensor>>& /* inputs */) {
  return true;
}

}  // namespace ngraph
