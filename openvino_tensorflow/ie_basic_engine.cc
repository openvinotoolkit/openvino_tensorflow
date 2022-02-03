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
    m_infer_reqs.push_back(m_compiled_model.create_infer_request());
  }

  std::cout << "OVTF_LOG - IE_Basic_Engine - A" << std::endl;
  //  Prepare input blobs
  auto parameters = m_model->get_parameters();
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr) {
        ov::Tensor tensor(inputs[i]->get_element_type(), inputs[i]->get_shape(), inputs[i]->get_data_ptr());
#if defined(OPENVINO_2021_2)
      if (m_device != "MYRIAD" && m_device != "HDDL") {
        OVTF_VLOG(4) << "IE_Basic_Engine::infer() set_input_tensor() ("
                     << input_names[i] << ")";
        const int in_idx = get_input_idx(input_names[i]);
        if (in_idx < 0) {
          throw std::runtime_error("Input parameter with friendly name " +
                                   input_names[i] + " not found in ov::Model");
        }
        m_infer_reqs[0].set_input_tensor(in_idx, *(inputs[i]));
      } else {
        OVTF_VLOG(4) << "IE_Basic_Engine::infer() get_input_tensor() ("
                     << input_names[i] << ")";
        const int in_idx = get_input_idx(input_names[i]);
        if (in_idx < 0) {
          throw std::runtime_error("Input parameter with friendly name " +
                                   input_names[i] + " not found in ov::Model");
        }
        auto tensor = m_infer_reqs[0].get_input_tensor(in_idx);
        size_t input_data_size = tensor.get_byte_size();
        std::copy((uint8_t*)(inputs[i]->data()),
                  ((uint8_t*)(inputs[i]->data())) + input_data_size,
                  (uint8_t*)(tensor.data()));
      }
#else
      OVTF_VLOG(4) << "IE_Basic_Engine::infer() set_input_tensor() ("
                   << input_names[i] << ")";
      const int in_idx = get_input_idx(input_names[i]);
      if (in_idx < 0) {
        throw std::runtime_error("Input with friendly name " + input_names[i] +
                                 " not found in ov::Model");
      }
      m_infer_reqs[0].set_input_tensor(in_idx, *(inputs[i]));
#endif
    }
  }

  for (int i = 0; i < hoisted_params.size(); i++) {
    if (hoisted_params[i] != nullptr) {
      OVTF_VLOG(4) << "IE_Basic_Engine::infer() set_input_tensor() ("
                   << param_names[i] << ")";
      const int param_idx = get_input_idx(param_names[i]);
      if (param_idx < 0) {
        throw std::runtime_error("Hoisted parameter with friendly name " +
                                 param_names[i] + " not found in ov::Model");
      }
      m_infer_reqs[0].set_input_tensor(param_idx, *(hoisted_params[i]));
    }
  }

  //  Prepare output blobs
  auto results = m_model->get_results();
  for (int i = 0; i < results.size(); i++) {
    ov::Tensor tensor(outputs[i]->get_element_type(), outputs[i]->get_shape(), outputs[i]->get_data_ptr());
    if (outputs[i] != nullptr) {
      OVTF_VLOG(4) << "IE_Basic_Engine::infer() set_output_tensor() ("
                   << output_names[i] << ")";
      const int out_idx = get_output_idx(output_names[i]);
      if (out_idx < 0) {
        throw std::runtime_error("Output with friendly name " +
                                 output_names[i] + " not found in ov::Model");
      }
      m_infer_reqs[0].set_output_tensor(out_idx, *(outputs[i]));
    }
  }

  m_infer_reqs[0].infer();

  // Set dynamic output blobs
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] == nullptr) {
      OVTF_VLOG(4) << "IE_Basic_Engine::infer() get_output_tensor() ("
                   << output_names[i] << ")";
      const int out_idx = get_output_idx(output_names[i]);
      if (out_idx < 0) {
        throw std::runtime_error("Output with friendly name " +
                                 output_names[i] + " not found in ov::Model");
      }
      auto tensor = m_infer_reqs[0].get_output_tensor(out_idx);
      outputs[i] = std::make_shared<IETensor>(
          tensor.get_element_type(), tensor.get_shape(), tensor.data());
    }
  }
  OVTF_VLOG(4) << "Inference Successful";
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
