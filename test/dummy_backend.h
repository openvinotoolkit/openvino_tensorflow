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
#ifndef NGRAPH_TF_BRIDGE_DUMMYBACKEND_H_
#define NGRAPH_TF_BRIDGE_DUMMYBACKEND_H_

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"

using namespace std;
namespace ng = ngraph;

namespace ngraph {

namespace runtime {

namespace dummy {

class DummyBackend;
class DummyExecutable;
}
}
}

class ng::runtime::dummy::DummyBackend : public ng::runtime::Backend {
 public:
  DummyBackend();
  DummyBackend(const DummyBackend&) = delete;
  DummyBackend(DummyBackend&&) = delete;
  DummyBackend& operator=(const DummyBackend&) = delete;

  std::shared_ptr<ng::runtime::Tensor> create_tensor(
      const ng::element::Type& type, const ng::Shape& shape,
      void* memory_pointer) override;

  std::shared_ptr<ng::runtime::Tensor> create_tensor(
      const ng::element::Type& type, const ng::Shape& shape) override;

  std::shared_ptr<ng::runtime::Executable> compile(
      std::shared_ptr<ng::Function> function,
      bool enable_performance_data = false) override;

  bool is_supported(const ngraph::Node& node) const override;
  ~DummyBackend() override;
};

class ng::runtime::dummy::DummyExecutable : public ng::runtime::Executable {
 public:
  DummyExecutable(std::shared_ptr<ng::Function> function,
                  bool enable_performance_collection = false);
  bool call(
      const std::vector<std::shared_ptr<ng::runtime::Tensor>>& outputs,
      const std::vector<std::shared_ptr<ng::runtime::Tensor>>& inputs) override;
};

#endif  // NGRAPH_TF_BRIDGE_DUMMYBACKEND_H_