/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>
#include <memory>

#include "logging/ovtf_log.h"
#include "openvino_tensorflow/ie_basic_engine.h"
#include "openvino_tensorflow/ie_utils.h"

using namespace InferenceEngine;

namespace tensorflow {
namespace openvino_tensorflow {

//IE_Basic_Engine::IE_Basic_Engine(InferenceEngine::CNNNetwork ie_network,
//                                 std::string device)
//    : IE_Backend_Engine(ie_network, device) {}
IE_Basic_Engine::IE_Basic_Engine(std::shared_ptr<ov::Model> model,
                                 std::string device)
    : IE_Backend_Engine(model, device) {}

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
    //m_infer_reqs.push_back(m_exe_network.CreateInferRequest());
    m_infer_reqs.push_back(m_compiled_model.create_infer_request());
  }

  //  Prepare input blobs
  //auto model = m_network.getFunction();
  auto parameters = m_model->get_parameters();
  int j=0;
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr) {
        ov::Tensor tensor(inputs[i]->get_element_type(), inputs[i]->get_shape(), inputs[i]->get_data_ptr());
#if defined(OPENVINO_2021_2)
      if (m_device != "MYRIAD" && m_device != "HDDL")
        //m_infer_reqs[0].SetBlob(input_names[i], inputs[i]->get_blob());
        m_infer_reqs[0].set_input_tensor(j++, tensor);
      else {
        auto input_blob = m_infer_reqs[0].GetBlob(input_names[i]);
        MemoryBlob::Ptr minput = as<MemoryBlob>(input_blob);
        auto minputHolder = minput->wmap();

        auto inputBlobData = minputHolder.as<uint8_t*>();
        size_t input_data_size = input_blob->byteSize();
        inputs[i]->read((void*)inputBlobData, input_data_size);
      }
#else
      //m_infer_reqs[0].SetBlob(input_names[i], inputs[i]->get_blob());
      m_infer_reqs[0].set_input_tensor(j++, tensor);
#endif
    }
  }

  //for (int i = 0; i < hoisted_params.size(); i++) {
  //  if (hoisted_params[i] != nullptr)
  //    m_infer_reqs[0].SetBlob(param_names[i], hoisted_params[i]->get_blob());
  //}

  //  Prepare output blobs
  auto results = m_model->get_results();
  j = 0;
  for (int i = 0; i < results.size(); i++) {
    ov::Tensor tensor(outputs[i]->get_element_type(), outputs[i]->get_shape(), outputs[i]->get_data_ptr());
    if (outputs[i] != nullptr) {
      OVTF_VLOG(4) << "Executable::call() SetBlob()";
      //m_infer_reqs[0].SetBlob(output_names[i], outputs[i]->get_blob());
      m_infer_reqs[0].set_output_tensor(j++, tensor);
    }
  }

  m_infer_reqs[0].infer();

  // Set dynamic output blobs
  //for (int i = 0; i < results.size(); i++) {
  //  if (outputs[i] == nullptr) {
  //    OVTF_VLOG(4) << "Executable::call() GetBlob()";
  //    auto blob = m_infer_reqs[0].GetBlob(output_names[i]);
  //    outputs[i] = std::make_shared<IETensor>(blob);
  //  }
  //}
  OVTF_VLOG(4) << "Inference Successful";

  // return true;
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
