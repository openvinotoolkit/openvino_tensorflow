/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "backend_manager.h"
#include "logging/ovtf_log.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

shared_ptr<Backend> BackendManager::m_backend;
string BackendManager::m_backend_name;
mutex BackendManager::m_backend_mutex;
bool BackendManager::m_perf_counters_enabled = false;
bool BackendManager::m_enable_ovtf_profiling = false;
char* BackendManager::m_model_cache_dir = nullptr;
bool BackendManager::m_tf_frontend_disabled = false;
bool BackendManager::m_dynamic_shapes_enabled = false;
bool BackendManager::m_output_zero_copy = false;
bool BackendManager::m_static_input_checks_disabled = false;

BackendManager::~BackendManager() {
  OVTF_VLOG(2) << "BackendManager::~BackendManager()";
}

Status BackendManager::SetBackend(const string& backend_name) {
  OVTF_VLOG(2) << "BackendManager::SetBackend(" << backend_name << ")";
  shared_ptr<Backend> backend;
  string bname(backend_name);

  auto status = CreateBackend(backend, bname);
  if (!status.ok() || backend == nullptr) {
    return errors::Internal("Failed to set backend: ", status.error_message());
  }

  lock_guard<mutex> lock(m_backend_mutex);
  m_backend = backend;
  if (bname.find("MYRIAD") != string::npos) {
    m_backend_name = "MYRIAD";
  } else if (bname.find("GPU") != string::npos) {
    // Since m_backend_name is assigned "GPU" whenever the string "GPU" is found
    // in bname,
    // for ex: in GPU.0, GPU.1, or GPU.1_FP16, we can ignore maintaining
    // fullnames,
    // as m_backend_name is used by OCM and only needs "GPU" to check for Op
    // support.
    // In OVTF, we will assume that all types of GPU devices have the same Op
    // support.
    m_backend_name = "GPU";
  } else {
    m_backend_name = bname;
  }
  // read value of OPENVINO_TF_ENABLE_PERF_COUNT
  const char* openvino_tf_enable_perf_count =
      std::getenv("OPENVINO_TF_ENABLE_PERF_COUNT");
  if (openvino_tf_enable_perf_count != nullptr) {
    if (1 == std::stoi(openvino_tf_enable_perf_count)) {
      m_perf_counters_enabled = true;
    }
  }

  // read value of OPENVINO_TF_ENABLE_OVTF_PROFILING
  const char* openvino_tf_enable_ovtf_profiling =
      std::getenv("OPENVINO_TF_ENABLE_OVTF_PROFILING");
  if (openvino_tf_enable_ovtf_profiling != nullptr) {
    if (1 == std::stoi(openvino_tf_enable_ovtf_profiling)) {
      m_enable_ovtf_profiling = true;
    }
  }

  const char* openvino_tf_disable_tffe =
      std::getenv("OPENVINO_TF_DISABLE_TFFE");
  if (openvino_tf_disable_tffe != nullptr) {
    if (1 == std::stoi(openvino_tf_disable_tffe)) {
      m_tf_frontend_disabled = true;
    }
  }

  const char* openvino_tf_enable_dynamic_shapes =
      std::getenv("OPENVINO_TF_ENABLE_DYNAMIC_SHAPES");
  if (openvino_tf_enable_dynamic_shapes != nullptr) {
    if (1 == std::stoi(openvino_tf_enable_dynamic_shapes)) {
      m_dynamic_shapes_enabled = true;
    }
  }

  const char* openvino_tf_output_zero_copy =
      std::getenv("OPENVINO_TF_OUTPUT_ZERO_COPY");
  if (openvino_tf_output_zero_copy != nullptr) {
    if (1 == std::stoi(openvino_tf_output_zero_copy)) {
      m_output_zero_copy = true;
    }
  }

  const char* openvino_tf_disable_static_input_checks =
      std::getenv("OPENVINO_TF_DISABLE_STATIC_INPUT_CHECKS");
  if (openvino_tf_disable_static_input_checks != nullptr) {
    if (1 == std::stoi(openvino_tf_disable_static_input_checks)) {
      m_static_input_checks_disabled = true;
    }
    if (m_tf_frontend_disabled == true) {
      m_static_input_checks_disabled = false;  // ignore envvar and always
                                               // enable static input checking
                                               // when TFFE is not enabled
    }
  }

  //  read value of OPENVINO_TF_MODEL_CACHE_DIR
  m_model_cache_dir = std::getenv("OPENVINO_TF_MODEL_CACHE_DIR");

  return Status::OK();
}

shared_ptr<Backend> BackendManager::GetBackend() {
  OVTF_VLOG(2) << "BackendManager::GetBackend()";
  if (m_backend == nullptr) {
    auto status = SetBackend();
    if (!status.ok()) {
      OVTF_VLOG(0) << "Failed to get backend: " << status.error_message();
      throw errors::Internal("Failed to get backend: ", status.error_message());
    }
  }
  lock_guard<mutex> lock(m_backend_mutex);
  return m_backend;
}

Status BackendManager::GetBackendName(string& backend_name) {
  OVTF_VLOG(2) << "BackendManager::GetBackendName()";
  if (m_backend == nullptr) {
    auto status = SetBackend();
    if (!status.ok()) {
      OVTF_VLOG(0) << "Failed to get backend name: " << status.error_message();
      return errors::Internal("Failed to get backend name: ",
                              status.error_message());
    }
  }
  lock_guard<mutex> lock(m_backend_mutex);
  backend_name = m_backend_name;
  return Status::OK();
}

Status BackendManager::CreateBackend(shared_ptr<Backend>& backend,
                                     string& backend_name) {
  const char* env = std::getenv("OPENVINO_TF_BACKEND");
  // Array should be of max length MYRIAD.
  char backendName[7];

  if (env != nullptr) {
    strncpy((char*)backendName, env, sizeof(backendName));
    backendName[6] = '\0';  // null terminate to remove warnings
    backend_name = std::string(backendName);
  }

  if (backend_name == "HDDL") {
    return errors::Internal("Failed to Create backend: ",
                            backend_name + " backend not available");
  }
  if (backend_name == "VAD-M") backend_name = "HDDL";

  try {
    backend = make_shared<Backend>(backend_name);
  } catch (const std::exception& e) {
    return errors::Internal("Could not create backend of type ", backend_name,
                            ". Got exception: ", e.what());
  }
  if (backend == nullptr) {
    return errors::Internal("Could not create backend of type ", backend_name,
                            " got nullptr");
  }

  OVTF_VLOG(2) << "BackendManager::CreateBackend(): " << backend_name;
  return Status::OK();
}

// Returns the supported backend names
vector<string> BackendManager::GetSupportedBackends() {
  ov::Core core;
  auto devices = core.get_available_devices();
  auto pos = find(devices.begin(), devices.end(), "HDDL");
  if (pos != devices.end()) {
    devices.erase(pos);
    devices.push_back("VAD-M");
  }
  return devices;
}

// Returns if Performance Counters are Enabled
bool BackendManager::PerfCountersEnabled() { return m_perf_counters_enabled; }

// Returns if Profiling is Enabled in OVTF
bool BackendManager::OVTFProfilingEnabled() { return m_enable_ovtf_profiling; }

// Returns the value of Model Cache Dir if set
char* BackendManager::GetModelCacheDir() { return m_model_cache_dir; }

// Returns if TF Frontend is disabled
bool BackendManager::TFFrontendDisabled() { return m_tf_frontend_disabled; }

// Returns true if dynamic input shape support is enabled
bool BackendManager::DynamicShapesEnabled() { return m_dynamic_shapes_enabled; }

// Returns true if zero-copy enabled for dynamic outputs
bool BackendManager::OutputZeroCopy() { return m_output_zero_copy; }

// Returns true if static input checking is disabled
bool BackendManager::StaticInputChecksDisabled() {
  return m_static_input_checks_disabled;
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
