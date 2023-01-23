/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>
#include <memory>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "logging/ovtf_log.h"
#include "openvino_tensorflow/backend_manager.h"
#include "openvino_tensorflow/ie_basic_engine.h"
#include "openvino_tensorflow/ie_utils.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

#ifdef _WIN32
#define GetCurrentTimeNanos() profiler::GetCurrentTimeNanos()
#else
#define GetCurrentTimeNanos() absl::GetCurrentTimeNanos()
#endif

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
  double start_ns = 0;
  if (BackendManager::OVTFProfilingEnabled()) start_ns = GetCurrentTimeNanos();
  load_network();
  if (BackendManager::OVTFProfilingEnabled()) {
    double duration_in_ms = (GetCurrentTimeNanos() - start_ns) / 1e6;
    OVTF_VLOG(1) << "OVTF_LOAD_NETWORK_TIME: " << duration_in_ms << " ms";
  }

  if (BackendManager::OVTFProfilingEnabled()) start_ns = GetCurrentTimeNanos();
  if (m_infer_reqs.empty()) {
    m_infer_reqs.push_back(m_compiled_model.create_infer_request());
  }
  if (BackendManager::OVTFProfilingEnabled()) {
    double duration_in_ms = (GetCurrentTimeNanos() - start_ns) / 1e6;
    OVTF_VLOG(1) << "OVTF_CREATE_REQUEST_TIME: " << duration_in_ms << " ms";
  }

  if (BackendManager::OVTFProfilingEnabled()) start_ns = GetCurrentTimeNanos();
  //  Prepare input blobs
  auto parameters = m_model->get_parameters();
  if (m_in_idx.size() == 0) {
    m_in_idx.resize(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
      if (inputs[i] != nullptr) {
        m_in_idx[i] = get_input_idx(input_names[i]);
      }
    }
  }
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr) {
      OVTF_VLOG(4) << "IE_Basic_Engine::infer() set_input_tensor() ("
                   << input_names[i] << ")";
      if (inputs[i]->get_shape().size() > 0 && inputs[i]->get_shape()[0] == 0)
        continue;
      const int in_idx = m_in_idx[i];
      if (in_idx < 0) {
        throw std::runtime_error("Input with friendly name " + input_names[i] +
                                 " not found in ov::Model");
      }
      m_infer_reqs[0].set_input_tensor(in_idx, *(inputs[i]));
    }
  }
  if (BackendManager::OVTFProfilingEnabled()) {
    double duration_in_ms = (GetCurrentTimeNanos() - start_ns) / 1e6;
    OVTF_VLOG(1) << "OVTF_INPUTS_SET_TIME: " << duration_in_ms << " ms";
  }

  if (BackendManager::OVTFProfilingEnabled()) start_ns = GetCurrentTimeNanos();
  if (m_param_idx.size() == 0) {
    m_param_idx.resize(hoisted_params.size());
    for (int i = 0; i < hoisted_params.size(); i++) {
      if (hoisted_params[i] != nullptr) {
        m_param_idx[i] = get_input_idx(param_names[i]);
      }
    }
  }
  for (int i = 0; i < hoisted_params.size(); i++) {
    if (hoisted_params[i] != nullptr) {
      OVTF_VLOG(4) << "IE_Basic_Engine::infer() set_input_tensor() ("
                   << param_names[i] << ")";
      const int param_idx = m_param_idx[i];
      if (param_idx < 0) {
        throw std::runtime_error("Hoisted parameter with friendly name " +
                                 param_names[i] + " not found in ov::Model");
      }
      m_infer_reqs[0].set_input_tensor(param_idx, *(hoisted_params[i]));
    }
  }
  if (BackendManager::OVTFProfilingEnabled()) {
    double duration_in_ms = (GetCurrentTimeNanos() - start_ns) / 1e6;
    OVTF_VLOG(1) << "OVTF_PARAMS_SET_TIME: " << duration_in_ms << " ms";
  }

  if (BackendManager::OVTFProfilingEnabled()) start_ns = GetCurrentTimeNanos();
  //  Prepare output blobs
  auto results = m_model->get_results();
  if (m_out_idx.size() == 0) {
    m_out_idx.resize(results.size());
    for (int i = 0; i < results.size(); i++) {
      m_out_idx[i] = get_output_idx(output_names[i]);
    }
  }
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] != nullptr) {
      OVTF_VLOG(4) << "IE_Basic_Engine::infer() set_output_tensor() ("
                   << output_names[i] << ")";
      const int out_idx = m_out_idx[i];
      if (out_idx < 0) {
        throw std::runtime_error("Output with friendly name " +
                                 output_names[i] + " not found in ov::Model");
      }
      m_infer_reqs[0].set_output_tensor(out_idx, *(outputs[i]));
    }
  }
  if (BackendManager::OVTFProfilingEnabled()) {
    double duration_in_ms = (GetCurrentTimeNanos() - start_ns) / 1e6;
    OVTF_VLOG(1) << "OVTF_OUTPUTS_SET_TIME: " << duration_in_ms << " ms";
  }
  if (BackendManager::OVTFProfilingEnabled()) start_ns = GetCurrentTimeNanos();
  m_infer_reqs[0].infer();
  if (BackendManager::OVTFProfilingEnabled()) {
    double duration_in_ms = (GetCurrentTimeNanos() - start_ns) / 1e6;
    OVTF_VLOG(1) << "OVTF_INFERENCE_TIME: " << duration_in_ms << " ms";
  }

  if (BackendManager::PerfCountersEnabled()) {
    std::cout << "Performance counts:" << std::endl;
    auto prof_infos = m_infer_reqs[0].get_profiling_info();
    std::sort(prof_infos.begin(), prof_infos.end(),
              [](auto prof_info_a, auto prof_info_b) {
                return prof_info_b.real_time < prof_info_a.real_time;
              });
    std::cout << "Type;Real Time (ms);Node Name" << std::endl;
    for (auto info : prof_infos) {
      std::cout << info.node_type << ";"
                << std::to_string(
                       std::chrono::duration<double, std::milli>(info.real_time)
                           .count())
                << ";" << info.node_name << std::endl;
    }
  }

  if (BackendManager::OVTFProfilingEnabled()) start_ns = GetCurrentTimeNanos();
  // Set dynamic output blobs
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] == nullptr) {
      OVTF_VLOG(4) << "IE_Basic_Engine::infer() get_output_tensor() ("
                   << output_names[i] << ")";
      const int out_idx = m_out_idx[i];
      if (out_idx < 0) {
        throw std::runtime_error("Output with friendly name " +
                                 output_names[i] + " not found in ov::Model");
      }
      auto tensor = m_infer_reqs[0].get_output_tensor(out_idx);
      outputs[i] = std::make_shared<IETensor>(
          tensor.get_element_type(), tensor.get_shape(), tensor.data());
    }
  }
  if (BackendManager::OVTFProfilingEnabled()) {
    double duration_in_ms = (GetCurrentTimeNanos() - start_ns) / 1e6;
    OVTF_VLOG(1) << "OVTF_DYNAMIC_OUTPUT_SET_TIME: " << duration_in_ms << " ms";
  }
  OVTF_VLOG(4) << "Inference Successful";
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
