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

shared_ptr<Executable> Backend::Compile(shared_ptr<ngraph::Function> func,
                                        bool) {
  return make_shared<Executable>(func, m_device);
}

bool Backend::IsSupported(const Node& node) const {
  // TODO: check if the given backend/device supports the op. Right now we're
  // assuming
  // that the selected backend supports all opset5 ops
  const auto& opset = ngraph::get_opset5();
  return opset.contains_op_type(&node);
}

}  // namespace ngraph_bridge
}  // namespace tensorflow