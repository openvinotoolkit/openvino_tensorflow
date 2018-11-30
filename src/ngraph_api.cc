/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

#include "ngraph_api.h"

namespace tensorflow {
namespace ngraph_bridge {
namespace config {

static bool _is_enabled = true;
static bool _is_logging_placement = false;

extern "C" {
void ngraph_enable() { Enable(); }
void ngraph_disable() { Disable(); }
bool ngraph_is_enabled() { return IsEnabled(); }

size_t ngraph_backends_len() { return BackendsLen(); }

bool ngraph_list_backends(char** backends, int backends_len) {
  const auto ngraph_backends = ListBackends();
  if (backends_len != ngraph_backends.size()) {
    return false;
  }

  for (size_t idx = 0; idx < backends_len; idx++) {
    backends[idx] = strdup(ngraph_backends[idx].c_str());
  }
  return true;
}

bool ngraph_set_backend(const char* backend) {
  if (SetBackend(string(backend)) != tensorflow::Status::OK()) {
    return false;
  }
  return true;
}

extern bool ngraph_is_supported_backend(const char* backend) {
  return IsSupportedBackend(string(backend));
}

extern bool ngraph_get_currently_set_backend_name(char** backend) {
  backend[0] = strdup(GetCurrentlySetBackendName().c_str());
  return true;
}

void ngraph_start_logging_placement() { StartLoggingPlacement(); }
void ngraph_stop_logging_placement() { StopLoggingPlacement(); }
bool ngraph_is_logging_placement() { return IsLoggingPlacement(); }
}

// note that TensorFlow always uses camel case for the C++ API, but not for
// Python
void Enable() { _is_enabled = true; }
void Disable() { _is_enabled = false; }
bool IsEnabled() { return _is_enabled; }

size_t BackendsLen() { return BackendManager::GetNumOfSupportedBackends(); }

vector<string> ListBackends() {
  auto supported_backends = BackendManager::GetSupportedBackendNames();
  vector<string> backend_list(supported_backends.begin(),
                              supported_backends.end());
  return backend_list;
}

tensorflow::Status SetBackend(const string& type) {
  return BackendManager::SetBackendName(type);
}

bool IsSupportedBackend(const string& type) {
  return BackendManager::IsSupportedBackend(type);
}

string GetCurrentlySetBackendName() {
  return BackendManager::GetCurrentlySetBackendName();
}

void StartLoggingPlacement() { _is_logging_placement = true; }
void StopLoggingPlacement() { _is_logging_placement = false; }
bool IsLoggingPlacement() {
  return _is_enabled && (_is_logging_placement ||
                         std::getenv("NGRAPH_TF_LOG_PLACEMENT") != nullptr);
}

}  // namespace config
}  // namespace ngraph_bridge
}  // namespace tensorflow
