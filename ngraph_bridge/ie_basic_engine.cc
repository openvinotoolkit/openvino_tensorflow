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

#include <iostream>

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ie_basic_engine.h"
#include "ngraph_bridge/ie_utils.h"

namespace tensorflow {
namespace ngraph_bridge {

IE_Basic_Engine::IE_Basic_Engine(InferenceEngine::CNNNetwork ie_network,
                                 std::string device)
    : IE_Backend_Engine(ie_network, device) {}

IE_Basic_Engine::~IE_Basic_Engine() {}

void IE_Basic_Engine::infer(
    std::vector<std::shared_ptr<IETensor>>& inputs,
    std::vector<std::string>& input_names,
    std::vector<std::shared_ptr<IETensor>>& outputs,
    std::vector<std::string>& output_names,
    std::vector<std::shared_ptr<IETensor>>& hoisted_params,
    std::vector<std::string>& param_names) {
  load_network();
  if (m_infer_reqs.empty()) {
    m_infer_reqs.push_back(m_exe_network.CreateInferRequest());
  }

  //  Prepare input blobs
  auto func = m_network.getFunction();
  auto parameters = func->get_parameters();
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr)
      m_infer_reqs[0].SetBlob(input_names[i], inputs[i]->get_blob());
  }

  for (int i = 0; i < hoisted_params.size(); i++) {
    if (hoisted_params[i] != nullptr)
      m_infer_reqs[0].SetBlob(param_names[i], hoisted_params[i]->get_blob());
  }

  //  Prepare output blobs
  auto results = func->get_results();
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] != nullptr) {
      NGRAPH_VLOG(4) << "Executable::call() SetBlob()";
      m_infer_reqs[0].SetBlob(output_names[i], outputs[i]->get_blob());
    }
  }

  m_infer_reqs[0].Infer();

  // Set dynamic output blobs
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] == nullptr) {
      NGRAPH_VLOG(4) << "Executable::call() GetBlob()";
      auto blob = m_infer_reqs[0].GetBlob(output_names[i]);
      outputs[i] = std::make_shared<IETensor>(blob);
    }
  }

  // return true;
}
}
}
