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
  std::cout << "OVTF_DEBUG - SetBackend - 1" << std::endl;
  shared_ptr<Backend> backend;
  std::cout << "OVTF_DEBUG - SetBackend - 2" << std::endl;
  string bname(backend_name);
  std::cout << "OVTF_DEBUG - SetBackend - 3" << std::endl;

  auto status = CreateBackend(backend, bname);
  std::cout << "OVTF_DEBUG - SetBackend - 4" << std::endl;
  if (!status.ok() || backend == nullptr) {
  std::cout << "OVTF_DEBUG - SetBackend - 5" << std::endl;
    return errors::Internal("Failed to set backend: ", status.error_message());
  }
  std::cout << "OVTF_DEBUG - SetBackend - 6" << std::endl;

  lock_guard<mutex> lock(m_backend_mutex);
  std::cout << "OVTF_DEBUG - SetBackend - 7" << std::endl;
  m_backend = backend;
  std::cout << "OVTF_DEBUG - SetBackend - 8" << std::endl;
  if (bname.find("MYRIAD") != string::npos) {
  std::cout << "OVTF_DEBUG - SetBackend - 9" << std::endl;
    m_backend_name = "MYRIAD";
  std::cout << "OVTF_DEBUG - SetBackend - 10" << std::endl;
    m_tf_frontend_disabled = true;
  std::cout << "OVTF_DEBUG - SetBackend - 11" << std::endl;
  } else if (bname.find("GPU") != string::npos) {
    // Since m_backend_name is assigned "GPU" whenever the string "GPU" is found
    // in bname,
    // for ex: in GPU.0, GPU.1, or GPU.1_FP16, we can ignore maintaining
    // fullnames,
    // as m_backend_name is used by OCM and only needs "GPU" to check for Op
    // support.
    // In OVTF, we will assume that all types of GPU devices have the same Op
    // support.
  std::cout << "OVTF_DEBUG - SetBackend - 12" << std::endl;
    m_backend_name = "GPU";
  std::cout << "OVTF_DEBUG - SetBackend - 13" << std::endl;
  } else {
  std::cout << "OVTF_DEBUG - SetBackend - 14" << std::endl;
    m_backend_name = bname;
  std::cout << "OVTF_DEBUG - SetBackend - 15" << std::endl;
    if (bname.find("HDDL") != string::npos) m_tf_frontend_disabled = true;
  std::cout << "OVTF_DEBUG - SetBackend - 16" << std::endl;
  }
  std::cout << "OVTF_DEBUG - SetBackend - 17" << std::endl;
  // read value of OPENVINO_TF_ENABLE_PERF_COUNT
  const char* openvino_tf_enable_perf_count =
      std::getenv("OPENVINO_TF_ENABLE_PERF_COUNT");
  std::cout << "OVTF_DEBUG - SetBackend - 18" << std::endl;
  if (openvino_tf_enable_perf_count != nullptr) {
  std::cout << "OVTF_DEBUG - SetBackend - 19" << std::endl;
    if (1 == std::stoi(openvino_tf_enable_perf_count)) {
  std::cout << "OVTF_DEBUG - SetBackend - 20" << std::endl;
      m_perf_counters_enabled = true;
  std::cout << "OVTF_DEBUG - SetBackend - 21" << std::endl;
    }
  }
  std::cout << "OVTF_DEBUG - SetBackend - 22" << std::endl;

  // read value of OPENVINO_TF_ENABLE_OVTF_PROFILING
  const char* openvino_tf_enable_ovtf_profiling =
      std::getenv("OPENVINO_TF_ENABLE_OVTF_PROFILING");
  std::cout << "OVTF_DEBUG - SetBackend - 23" << std::endl;
  if (openvino_tf_enable_ovtf_profiling != nullptr) {
  std::cout << "OVTF_DEBUG - SetBackend - 24" << std::endl;
    if (1 == std::stoi(openvino_tf_enable_ovtf_profiling)) {
  std::cout << "OVTF_DEBUG - SetBackend - 25" << std::endl;
      m_enable_ovtf_profiling = true;
  std::cout << "OVTF_DEBUG - SetBackend - 26" << std::endl;
    }
  }
  std::cout << "OVTF_DEBUG - SetBackend - 27" << std::endl;

  const char* openvino_tf_disable_tffe =
      std::getenv("OPENVINO_TF_DISABLE_TFFE");
  std::cout << "OVTF_DEBUG - SetBackend - 28" << std::endl;
  if (openvino_tf_disable_tffe != nullptr) {
  std::cout << "OVTF_DEBUG - SetBackend - 29" << std::endl;
    if (1 == std::stoi(openvino_tf_disable_tffe)) {
  std::cout << "OVTF_DEBUG - SetBackend - 30" << std::endl;
      m_tf_frontend_disabled = true;
  std::cout << "OVTF_DEBUG - SetBackend - 31" << std::endl;
    }
  }

  std::cout << "OVTF_DEBUG - SetBackend - 32" << std::endl;
  //  read value of OPENVINO_TF_MODEL_CACHE_DIR
  m_model_cache_dir = std::getenv("OPENVINO_TF_MODEL_CACHE_DIR");
  std::cout << "OVTF_DEBUG - SetBackend - 33" << std::endl;

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
  std::cout << "OVTF_DEBUG - CreateBackend - 1" << std::endl;
  const char* env = std::getenv("OPENVINO_TF_BACKEND");
  // Array should be of max length MYRIAD.
  std::cout << "OVTF_DEBUG - CreateBackend - 2" << std::endl;
  char backendName[7];

  std::cout << "OVTF_DEBUG - CreateBackend - 3" << std::endl;
  if (env != nullptr) {
  std::cout << "OVTF_DEBUG - CreateBackend - 4" << std::endl;
    strncpy((char*)backendName, env, sizeof(backendName));
  std::cout << "OVTF_DEBUG - CreateBackend - 5" << std::endl;
    backendName[6] = '\0';  // null terminate to remove warnings
  std::cout << "OVTF_DEBUG - CreateBackend - 6" << std::endl;
    backend_name = std::string(backendName);
  std::cout << "OVTF_DEBUG - CreateBackend - 7" << std::endl;
  }

  std::cout << "OVTF_DEBUG - CreateBackend - 8" << std::endl;
  if (backend_name == "HDDL") {
  std::cout << "OVTF_DEBUG - CreateBackend - 9" << std::endl;
    return errors::Internal("Failed to Create backend: ",
                            backend_name + " backend not available");
  }
  std::cout << "OVTF_DEBUG - CreateBackend - 10" << std::endl;
  if (backend_name == "VAD-M") backend_name = "HDDL";
  std::cout << "OVTF_DEBUG - CreateBackend - 11" << std::endl;

  try {
  std::cout << "OVTF_DEBUG - CreateBackend - 12" << std::endl;
    backend = make_shared<Backend>(backend_name);
  std::cout << "OVTF_DEBUG - CreateBackend - 13" << std::endl;
  } catch (const std::exception& e) {
    return errors::Internal("Could not create backend of type ", backend_name,
                            ". Got exception: ", e.what());
  }
  std::cout << "OVTF_DEBUG - CreateBackend - 14" << std::endl;
  if (backend == nullptr) {
  std::cout << "OVTF_DEBUG - CreateBackend - 15" << std::endl;
    return errors::Internal("Could not create backend of type ", backend_name,
                            " got nullptr");
  }
  std::cout << "OVTF_DEBUG - CreateBackend - 16" << std::endl;

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
