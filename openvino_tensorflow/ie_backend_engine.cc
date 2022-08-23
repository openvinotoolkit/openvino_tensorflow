/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>

#include "backend_manager.h"
#include "openvino_tensorflow/ie_backend_engine.h"
#include "openvino_tensorflow/ie_utils.h"

namespace tensorflow {
namespace openvino_tensorflow {

IE_Backend_Engine::IE_Backend_Engine(std::shared_ptr<ov::Model> model,
                                     std::string device)
    : m_model(model),
      m_device(device),
      m_multi_req_execution(false),
      m_network_ready(false) {}

IE_Backend_Engine::~IE_Backend_Engine() {}

void IE_Backend_Engine::load_network() {
  if (m_network_ready) return;

  // TODO: Model caching needs to be validated with different use cases.
  const char* model_cache_dir =
      std::getenv("OPENVINO_TF_MODEL_CACHE_DIR");

  if (!(model_cache_dir == nullptr)) {
      Backend::GetGlobalContext().ie_core.set_property(ov::cache_dir(std::string(model_cache_dir)));
  }

  if (m_device == "MYRIAD") {
    // Set MYRIAD configurations
    ov::AnyMap config;

    if (IE_Utils::VPUConfigEnabled()) {
      config["MYRIAD_DETECT_NETWORK_BATCH"] = "NO";
    }

    if (IE_Utils::VPUFastCompileEnabled()) {
      config["MYRIAD_HW_INJECT_STAGES"] = "NO";
      config["MYRIAD_COPY_OPTIMIZATION"] = "NO";
    }
    Backend::GetGlobalContext().ie_core.set_property("MYRIAD", config);
  }

  // Load network to the plugin (m_device)
  auto backend = BackendManager::GetBackend();
  auto dev_type = backend->GetDeviceType();
  if (dev_type.find("GPU") != string::npos) dev_type = "GPU";
  m_compiled_model =
      Backend::GetGlobalContext().ie_core.compile_model(m_model, dev_type);
  m_network_ready = true;
}

void IE_Backend_Engine::start_async_inference(const int req_id) {
  // Start Async inference
  try {
    m_infer_reqs[req_id].start_async();
  } catch (ov::Exception const& e) {
    throw ov::Exception("Couldn't start Inference: ");
  } catch (...) {
    throw ov::Exception("Couldn't start Inference: ");
  }
}

void IE_Backend_Engine::complete_async_inference(const int req_id) {
  // Wait for Async inference completion
  try {
    m_infer_reqs[req_id].wait();
  } catch (ov::Exception const& e) {
    throw ov::Exception(" Exception with completing Inference: ");
  } catch (...) {
    throw ov::Exception(" Exception with completing Inference: ");
  }
}

size_t IE_Backend_Engine::get_output_batch_size(size_t inputBatchSize) const {
  ov::Dimension batch_dim = ov::get_batch(m_model);
  int64_t model_batch_size =
      (batch_dim.is_static() ? batch_dim.get_length() : 1);
  return model_batch_size * IE_Utils::GetNumRequests(inputBatchSize, m_device);
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

std::shared_ptr<ov::Model> IE_Backend_Engine::get_model() { return m_model; }

const int IE_Backend_Engine::get_input_idx(const std::string name) const {
  for (int i = 0; i < m_model->inputs().size(); i++) {
    if (m_model->inputs()[i].get_node()->get_friendly_name() == name) {
      return i;
    }
  }
  return -1;
}

const int IE_Backend_Engine::get_output_idx(const std::string name) const {
  for (int i = 0; i < m_model->outputs().size(); i++) {
    if (m_model->outputs()[i].get_node()->get_friendly_name() == name) {
      return i;
    }
  }
  return -1;
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
