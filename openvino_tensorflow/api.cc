/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "tensorflow/core/lib/core/errors.h"

#include "api.h"
#include "backend_manager.h"

namespace tensorflow {
namespace openvino_tensorflow {
namespace api {

static bool _is_enabled = true;
static bool _is_logging_placement = false;
static std::set<std::string> disabled_op_types{};
static char * backendName;
static char * backendList[4];

extern "C" {
void enable() { Enable(); }
void disable() { Disable(); }
bool is_enabled() { return IsEnabled(); }

size_t backends_len() {
  const auto backends = ListBackends();
  return backends.size();
}

bool list_backends(char** backends) {
  const auto ovtf_backends = ListBackends();
  for (size_t idx = 0; idx < ovtf_backends.size(); idx++) {
    backendList[idx] = strdup(ovtf_backends[idx].c_str());
    backends[idx] = backendList[idx] ;
  }
  return true;
}

void freeBackendsList() {
  const auto ovtf_backends = ListBackends();
  for (size_t idx = 0; idx < ovtf_backends.size(); idx++) {
    free(backendList[idx]);
  }
}

bool set_backend(const char* backend) { return SetBackend(string(backend)); }

extern bool get_backend(char** backend) {
  string b = GetBackend();
  if (b == "") {
    return false;
  }
  backendName = strdup(b.c_str());
  *backend = backendName;
  return true;
}
void  freeBackend() {
  free(backendName);
}
void start_logging_placement() { StartLoggingPlacement(); }
void stop_logging_placement() { StopLoggingPlacement(); }
bool is_logging_placement() { return IsLoggingPlacement(); }

extern void set_disabled_ops(const char* op_type_list) {
  SetDisabledOps(std::string(op_type_list));
}

extern const char* get_disabled_ops() {
  return ngraph::join(GetDisabledOps(), ",").c_str();
}
}

// note that TensorFlow always uses camel case for the C++ API, but not for
// Python
void Enable() { _is_enabled = true; }
void Disable() { _is_enabled = false; }
bool IsEnabled() { return _is_enabled; }

vector<string> ListBackends() { return BackendManager::GetSupportedBackends(); }

bool SetBackend(const string& type) {
  return (BackendManager::SetBackend(type) == Status::OK());
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
                         std::getenv("OPENVINO_TF_LOG_PLACEMENT") != nullptr);
}

std::set<string> GetDisabledOps() {
  const char* disabled_ops_char_ptr = std::getenv("OPENVINO_TF_DISABLED_OPS"); 
  if (disabled_ops_char_ptr != nullptr) {
    string disabled_ops_str = disabled_ops_char_ptr;
    SetDisabledOps(disabled_ops_str);
  }
  return disabled_op_types;
}

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

}  // namespace api
}  // namespace openvino_tensorflow
}  // namespace tensorflow
