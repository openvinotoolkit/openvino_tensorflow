/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>

#include "backend_manager.h"
#include "openvino_tensorflow/ie_backend_engine.h"
#include "openvino_tensorflow/ie_utils.h"

namespace tensorflow {
namespace openvino_tensorflow {

//IE_Backend_Engine::IE_Backend_Engine(InferenceEngine::CNNNetwork ie_network,
//                                     std::string device)
//    : m_network(ie_network),
//      m_model(ie_network.getFunction()),
IE_Backend_Engine::IE_Backend_Engine(std::shared_ptr<ov::Model> model,
                                     std::string device)
    : m_model(model),
      m_device(device),
      m_multi_req_execution(false),
      m_network_ready(false) {
  //if (std::getenv("OPENVINO_TF_DUMP_GRAPHS")) {
  //  auto& name = m_network.getName();
  //  m_network.serialize(name + ".xml", name + ".bin");
  //}
}

IE_Backend_Engine::~IE_Backend_Engine() {}

void IE_Backend_Engine::load_network() {
  if (m_network_ready) return;

  std::map<std::string, std::string> config;

  if (m_device == "MYRIAD") {
    // Set MYRIAD configurations
    if (IE_Utils::VPUConfigEnabled()) {
      config["MYRIAD_DETECT_NETWORK_BATCH"] = "NO";
    }

    if (IE_Utils::VPUFastCompileEnabled()) {
      config["MYRIAD_HW_INJECT_STAGES"] = "NO";
      config["MYRIAD_COPY_OPTIMIZATION"] = "NO";
    }
  }

  // Load network to the plugin (m_device)
  auto backend = BackendManager::GetBackend();
  auto dev_type = backend->GetDeviceType();
  if (dev_type.find("GPU") != string::npos) dev_type = "GPU";
  //m_exe_network = Backend::GetGlobalContext().ie_core.LoadNetwork(
  //    m_network, dev_type, config);
  //m_compiled_model = Backend::GetGlobalContext().ie_core.compile_model(
  //    m_model, dev_type, config);
  m_compiled_model = Backend::GetGlobalContext().ie_core.compile_model(
      m_model, dev_type);
  m_network_ready = true;
}

void IE_Backend_Engine::start_async_inference(const int req_id) {
  // Start Async inference
  try {
    m_infer_reqs[req_id].start_async();
  } catch (InferenceEngine::details::InferenceEngineException e) {
    THROW_IE_EXCEPTION << "Couldn't start Inference: ";
  } catch (...) {
    THROW_IE_EXCEPTION << "Couldn't start Inference: ";
  }
}

void IE_Backend_Engine::complete_async_inference(const int req_id) {
  // Wait for Async inference completion
  try {
    //m_infer_reqs[req_id].wait(
    //    InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    m_infer_reqs[req_id].wait();
  } catch (InferenceEngine::details::InferenceEngineException e) {
    THROW_IE_EXCEPTION << " Exception with completing Inference: ";
  } catch (...) {
    THROW_IE_EXCEPTION << " Exception with completing Inference: ";
  }
}

size_t IE_Backend_Engine::get_output_batch_size(size_t inputBatchSize) const {
  //return m_network.getBatchSize() *
  //       IE_Utils::GetNumRequests(inputBatchSize, m_device);
  return 1;
}

// Enables multi request execution if the execution engine supprts
void IE_Backend_Engine::enable_multi_req_execution() {
  m_multi_req_execution = true;
}
// Disables multi request execution
void IE_Backend_Engine::disable_multi_req_execution() {
  m_multi_req_execution = false;
}

std::shared_ptr<ov::Model> IE_Backend_Engine::get_model() {
  return m_model;
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
