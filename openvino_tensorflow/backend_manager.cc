/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
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

BackendManager::~BackendManager() {
  OVTF_VLOG(2) << "BackendManager::~BackendManager()";
}

Status BackendManager::SetBackend(const string& backend_name) {
  OVTF_VLOG(2) << "BackendManager::SetBackend(" << backend_name << ")";
  shared_ptr<Backend> backend;
  string bname(backend_name);
  if (bname == "HDDL") {
    return errors::Internal("Failed to set backend: ",
                            bname + " backend not available");
  }
  if (bname == "VAD-M") bname = "HDDL";

  auto status = CreateBackend(backend, bname);
  if (!status.ok() || backend == nullptr) {
    return errors::Internal("Failed to set backend: ", status.error_message());
  }

  lock_guard<mutex> lock(m_backend_mutex);
  m_backend = backend;
  if (bname.find("MYRIAD") != string::npos) {
    m_backend_name = "MYRIAD";
  } else if (bname.find("GPU") != string::npos) {
    m_backend_name = "GPU";
  } else {
    m_backend_name = bname;
  }
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
  // Checkmarx fix. Array of max length MYRIAD.
  char backendName[6];

  if (env != nullptr) {
    strncpy((char*)backendName, env, sizeof(backendName));
    backend_name = std::string(backendName);
  }

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
  InferenceEngine::Core core;
  auto devices = core.GetAvailableDevices();
  auto pos = find(devices.begin(), devices.end(), "HDDL");
  if (pos != devices.end()) {
    devices.erase(pos);
    devices.push_back("VAD-M");
  }
  return devices;
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
