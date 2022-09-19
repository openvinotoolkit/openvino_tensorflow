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
    m_tf_frontend_disabled = true;
  } else if (bname.find("GPU") != string::npos) {
    m_backend_name = "GPU";
  } else {
    m_backend_name = bname;
    if (bname.find("HDDL") != string::npos) m_tf_frontend_disabled = true;
  }
  // read value of OPENVINO_TF_ENABLE_PERF_COUNT
  if (std::getenv("OPENVINO_TF_ENABLE_PERF_COUNT") != nullptr) {
    if (1 == std::stoi(std::getenv("OPENVINO_TF_ENABLE_PERF_COUNT"))) {
      m_perf_counters_enabled = true;
    }
  }

  // read value of OPENVINO_TF_ENABLE_OVTF_PROFILING
  if (std::getenv("OPENVINO_TF_ENABLE_OVTF_PROFILING") != nullptr) {
    if (1 == std::stoi(std::getenv("OPENVINO_TF_ENABLE_OVTF_PROFILING"))) {
      m_enable_ovtf_profiling = true;
    }
  }

  if (std::getenv("OPENVINO_TF_DISABLE_TFFE") != nullptr) {
    if (1 == std::stoi(std::getenv("OPENVINO_TF_DISABLE_TFFE"))) {
      m_tf_frontend_disabled = true;
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

}  // namespace openvino_tensorflow
}  // namespace tensorflow
