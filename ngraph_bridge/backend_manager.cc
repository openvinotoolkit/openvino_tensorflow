/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "backend_manager.h"
#include "logging/ngraph_log.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

shared_ptr<Backend> BackendManager::m_backend;
string BackendManager::m_backend_name;
mutex BackendManager::m_backend_mutex;

BackendManager::~BackendManager() {
  NGRAPH_VLOG(2) << "BackendManager::~BackendManager()";
}

Status BackendManager::SetBackend(const string& backend_name) {
  NGRAPH_VLOG(2) << "BackendManager::SetBackend(" << backend_name << ")";
  shared_ptr<Backend> backend;
  string bname(backend_name);
  auto status = CreateBackend(backend, bname);
  if (!status.ok() || backend == nullptr) {
    return errors::Internal("Failed to set backend: ", status.error_message());
  }

  lock_guard<mutex> lock(m_backend_mutex);
  m_backend = backend;
  m_backend_name = bname;
  return Status::OK();
}

shared_ptr<Backend> BackendManager::GetBackend() {
  NGRAPH_VLOG(2) << "BackendManager::GetBackend()";
  if (m_backend == nullptr) {
    auto status = SetBackend();
    if (!status.ok()) {
      NGRAPH_VLOG(0) << "Failed to get backend: " << status.error_message();
      throw errors::Internal("Failed to get backend: ", status.error_message());
    }
  }
  lock_guard<mutex> lock(m_backend_mutex);
  return m_backend;
}

Status BackendManager::GetBackendName(string& backend_name) {
  NGRAPH_VLOG(2) << "BackendManager::GetBackendName()";
  if (m_backend == nullptr) {
    auto status = SetBackend();
    if (!status.ok()) {
      NGRAPH_VLOG(0) << "Failed to get backend name: "
                     << status.error_message();
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
  const char* env = std::getenv("NGRAPH_TF_BACKEND");
  if (env != nullptr && strlen(env) > 0) {
    backend_name = string(env);
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

  NGRAPH_VLOG(2) << "BackendManager::CreateBackend(): " << backend_name;
  return Status::OK();
}

// Returns the supported backend names
vector<string> BackendManager::GetSupportedBackends() {
  InferenceEngine::Core core;
  return core.GetAvailableDevices();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
