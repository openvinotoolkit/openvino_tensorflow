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

#include "backend.h"

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

Backend::Backend(const string& config) {
  string device = config.substr(0, config.find(":"));
  InferenceEngine::Core core;
  auto devices = core.GetAvailableDevices();
  // TODO: Handle multiple devices
  if (find(devices.begin(), devices.end(), device) == devices.end()) {
    stringstream ss;
    ss << "Device '" << config << "' not found.";
    throw runtime_error(ss.str());
  }
  m_device = config;
}

Backend::~Backend() { m_exec_map.clear(); }

shared_ptr<Executable> Backend::compile(shared_ptr<ngraph::Function> func,
                                        bool) {
  shared_ptr<Executable> rc;
  {
    std::lock_guard<std::mutex> guard(m_exec_map_mutex);
    auto it = m_exec_map.find(func);
    if (it != m_exec_map.end()) {
      rc = it->second;
      return rc;
    }
  }

  rc = make_shared<Executable>(func, m_device);
  {
    std::lock_guard<std::mutex> guard(m_exec_map_mutex);
    m_exec_map.insert({func, rc});
    return rc;
  }
}

void Backend::remove_compiled_function(shared_ptr<Executable> exec) {
  std::lock_guard<std::mutex> guard(m_exec_map_mutex);
  for (auto it = m_exec_map.begin(); it != m_exec_map.end(); ++it) {
    if (it->second == exec) {
      m_exec_map.erase(it);
      break;
    }
  }
}

bool Backend::is_supported(const Node& node) const {
  // TODO: check if the given backend/device supports the op. Right now we're
  // assuming
  // that the selected backend supports all opset3 ops
  const auto& opset = ngraph::get_opset4();
  return opset.contains_op_type(&node);
}

shared_ptr<runtime::Tensor> Backend::create_dynamic_tensor(
    const element::Type& type, const PartialShape& shape) {
  return make_shared<IETensor>(type, shape);
}

vector<string> Backend::get_registered_devices() {
  InferenceEngine::Core core;
  return core.GetAvailableDevices();
}

shared_ptr<runtime::Tensor> Backend::create_tensor(
    const element::Type& element_type, const Shape& shape) {
  return make_shared<IETensor>(element_type, shape);
}

shared_ptr<runtime::Tensor> Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* data) {
  return make_shared<IETensor>(element_type, shape, data);
}
}
}