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

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ie_executable.h"
#include "ngraph_bridge/ie_tensor.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

IE_Executable::IE_Executable(shared_ptr<Function> func, string device)
    : m_device{device} {
  NGRAPH_VLOG(2) << "Checking for unsupported ops in IE backend";
  const auto& opset = ngraph::get_opset3();
  for (const auto& node : func->get_ops()) {
    if (!opset.contains_op_type(node.get())) {
      NGRAPH_VLOG(0) << "UNSUPPORTED OP DETECTED: "
                     << node->get_type_info().name;
      THROW_IE_EXCEPTION << "Detected op not belonging to opset3!";
    }
  }

  NGRAPH_VLOG(2) << "Creating IE CNN network using nGraph function";
  m_network = InferenceEngine::CNNNetwork(func);
  set_parameters_and_results(*func);

  if (std::getenv("NGRAPH_TF_DUMP_GRAPHS")) {
    auto& name = m_network.getName();
    m_network.serialize(name + ".xml", name + ".bin");
  }

  NGRAPH_VLOG(2) << "Loading IE CNN network to device " << m_device;

  InferenceEngine::Core ie;
  // Load model to the plugin (m_device)
  InferenceEngine::ExecutableNetwork exe_network =
      ie.LoadNetwork(m_network, m_device);

  // Create infer request
  m_infer_req = exe_network.CreateInferRequest();
}

bool IE_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                         const vector<shared_ptr<runtime::Tensor>>& inputs) {
  InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();
  if (input_info.size() != inputs.size()) {
    THROW_IE_EXCEPTION
        << "Function inputs number differ from number of given inputs";
  }

  size_t i = 0;
  for (const auto& it : input_info) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(inputs[i]);
    m_infer_req.SetBlob(it.first, tv->get_blob());
    i++;
  }

  //  Prepare output blobs
  InferenceEngine::OutputsDataMap output_info = m_network.getOutputsInfo();
  if (output_info.size() != outputs.size()) {
    THROW_IE_EXCEPTION
        << "Function outputs number differ from number of given outputs";
  }

  i = 0;
  for (const auto& it : output_info) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(outputs[i]);
    m_infer_req.SetBlob(it.first, tv->get_blob());
    i++;
  }

  m_infer_req.Infer();
  return true;
}
}
}