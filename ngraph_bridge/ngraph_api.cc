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

#include "ngraph_bridge/ngraph_api.h"

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

size_t ngraph_backends_len() {
  const auto backends = ListBackends();
  return backends.size();
}

bool ngraph_list_backends(char** backends) {
  const auto ngraph_backends = ListBackends();
  for (size_t idx = 0; idx < ngraph_backends.size(); idx++) {
    backends[idx] = strdup(ngraph_backends[idx].c_str());
  }
  return true;
}

bool ngraph_set_backend(const char* backend) {
  return (SetBackend(string(backend)) == tensorflow::Status::OK());
}

extern bool ngraph_get_backend(char** backend) {
  string b = GetBackend();
  if (b == "") {
    return false;
  }
  *backend = strdup(b.c_str());
  return true;
}

void ngraph_start_logging_placement() { StartLoggingPlacement(); }
void ngraph_stop_logging_placement() { StopLoggingPlacement(); }
bool ngraph_is_logging_placement() { return IsLoggingPlacement(); }

extern void ngraph_set_disabled_ops(const char* op_type_list) {
  SetDisabledOps(std::string(op_type_list));
}

extern const char* ngraph_get_disabled_ops() {
  return ngraph::join(GetDisabledOps(), ",").c_str();
}
}

// note that TensorFlow always uses camel case for the C++ API, but not for
// Python
void Enable() { _is_enabled = true; }
void Disable() { _is_enabled = false; }
bool IsEnabled() { return _is_enabled; }

vector<string> ListBackends() { return BackendManager::GetSupportedBackends(); }

Status SetBackend(const string& type) {
  return BackendManager::SetBackend(type);
}

string GetBackend() {
  string backend;
  if (BackendManager::GetBackendName(backend) != Status::OK()) {
    return "";
  }
  return backend;
}

void StartLoggingPlacement() { _is_logging_placement = true; }
void StopLoggingPlacement() { _is_logging_placement = false; }
bool IsLoggingPlacement() {
  return _is_enabled && (_is_logging_placement ||
                         std::getenv("NGRAPH_TF_LOG_PLACEMENT") != nullptr);
}

std::set<string> GetDisabledOps() { return disabled_op_types; }

void SetDisabledOps(string disabled_ops_str) {
  auto disabled_ops_list = ngraph::split(disabled_ops_str, ',');
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
