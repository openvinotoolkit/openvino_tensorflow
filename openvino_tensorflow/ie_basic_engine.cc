/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>

#include "logging/ovtf_log.h"
#include "openvino_tensorflow/ie_basic_engine.h"
#include "openvino_tensorflow/ie_utils.h"

using namespace InferenceEngine;

namespace tensorflow {
namespace openvino_tensorflow {

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
    if (inputs[i] != nullptr){

      if(m_device != "MYRIAD" && m_device != "VAD-M")
        m_infer_reqs[0].SetBlob(input_names[i], inputs[i]->get_blob());
      else{
        auto input_blob = m_infer_reqs[0].GetBlob(input_names[i]);
        MemoryBlob::Ptr minput = as<MemoryBlob>(input_blob);
        auto minputHolder = minput->wmap();

        auto inputBlobData = minputHolder.as<uint8_t*>();
        size_t input_data_size = input_blob->byteSize();
        inputs[i]->read((void*)inputBlobData, input_data_size);
      }
    }
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
  NGRAPH_VLOG(4) << "Inference Successful";

  // return true;
}
}// namespace openvino_tensorflow
}// namespace tensorflow
