/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include "ngraph_bridge/ngraph_api.h"

namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace config {

static bool _is_enabled = true;
static bool _is_logging_placement = false;
static std::set<std::string> disabled_op_types{};

extern "C" {
void ngraph_enable() { Enable(); }
void ngraph_disable() { Disable(); }
bool ngraph_is_enabled() { return IsEnabled(); }

size_t ngraph_backends_len() { return BackendsLen(); }

bool ngraph_list_backends(char** backends, size_t backends_len) {
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

extern bool ngraph_get_currently_set_backend_name(char** backend) {
  string bend;
  if (GetCurrentlySetBackendName(&bend) != tensorflow::Status::OK()) {
    return false;
  }
  backend[0] = strdup(bend.c_str());
  return true;
}

void ngraph_start_logging_placement() { StartLoggingPlacement(); }
void ngraph_stop_logging_placement() { StopLoggingPlacement(); }
bool ngraph_is_logging_placement() { return IsLoggingPlacement(); }

extern void ngraph_set_disabled_ops(const char* op_type_list) {
  SetDisabledOps(std::string(op_type_list));
}

extern const char* ngraph_get_disabled_ops() {
  return ng::join(GetDisabledOps(), ",").c_str();
}
}

// note that TensorFlow always uses camel case for the C++ API, but not for
// Python
void Enable() { _is_enabled = true; }
void Disable() { _is_enabled = false; }
bool IsEnabled() { return _is_enabled; }

size_t BackendsLen() { return BackendManager::GetNumOfSupportedBackends(); }

vector<string> ListBackends() {
  return BackendManager::GetSupportedBackendNames();
}

Status SetBackend(const string& type) {
  return BackendManager::SetBackendName(type);
}

Status GetCurrentlySetBackendName(string* backend_name) {
  return BackendManager::GetCurrentlySetBackendName(backend_name);
}

void StartLoggingPlacement() { _is_logging_placement = true; }
void StopLoggingPlacement() { _is_logging_placement = false; }
bool IsLoggingPlacement() {
  return _is_enabled && (_is_logging_placement ||
                         std::getenv("NGRAPH_TF_LOG_PLACEMENT") != nullptr);
}

std::set<string> GetDisabledOps() { return disabled_op_types; }

void SetDisabledOps(string disabled_ops_str) {
  auto disabled_ops_list = ng::split(disabled_ops_str, ',');
  // In case string is '', then splitting yields ['']. So taking care that ['']
  // corresponds to empty set {}
  if (disabled_ops_list.size() >= 1 && disabled_ops_list[0] != "") {
    SetDisabledOps(
        set<string>(disabled_ops_list.begin(), disabled_ops_list.end()));
  } else {
    SetDisabledOps(set<string>{});
  }
}

void SetDisabledOps(set<string> disabled_ops_set) {
  disabled_op_types = disabled_ops_set;
}

}  // namespace config
}  // namespace ngraph_bridge
}  // namespace tensorflow
