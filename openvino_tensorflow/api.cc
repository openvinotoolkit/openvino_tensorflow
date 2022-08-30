/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef _WIN32
#include <dirent.h>
#endif
#include <sys/stat.h>

#include "tensorflow/core/lib/core/errors.h"

#include "api.h"
#include "backend_manager.h"

namespace tensorflow {
namespace openvino_tensorflow {
namespace api {

static bool _is_enabled = true;
static bool _is_logging_placement = false;
static bool _is_rewrite_pass_enabled = true;
static std::set<std::string> disabled_op_types{};
static char* backendName = nullptr;
static char* backendList[4];
static char* clusterInfo = nullptr;
static char* errMsg = nullptr;

extern "C" {
void enable() { Enable(); }
void disable() { Disable(); }
bool is_enabled() { return IsEnabled(); }

size_t backends_len() { return ListBackends().size(); }

bool list_backends(char** backends) {
  const auto ovtf_backends = ListBackends();
  int i = 0;
  for (size_t idx = 0; idx < ovtf_backends.size(); idx++) {
    backendList[idx] = strdup(ovtf_backends[idx].c_str());
    backends[i++] = backendList[idx];
  }
  return true;
}

void EXPORT_SYMBOL freeBackendsList() {
  const auto ovtf_backends = ListBackends();
  for (size_t idx = 0; idx < ovtf_backends.size(); idx++) {
    free(backendList[idx]);
  }
}

bool set_backend(const char* backend) {
  auto status = BackendManager::SetBackend(string(backend));
  if (status != Status::OK()) {
    std::cerr << status.error_message() << std::endl;
    return false;
  }
  return true;
}

extern bool get_backend(char** backend) {
  string b = GetBackend();
  if (b == "") {
    return false;
  }
  backendName = strdup(b.c_str());
  *backend = backendName;
  return true;
}
void EXPORT_SYMBOL freeBackend() { free(backendName); }
void start_logging_placement() { StartLoggingPlacement(); }
void stop_logging_placement() { StopLoggingPlacement(); }
bool is_logging_placement() { return IsLoggingPlacement(); }
void EXPORT_SYMBOL freeClusterInfo() { free(clusterInfo); }
void EXPORT_SYMBOL freeErrMsg() { free(errMsg); }

extern void set_disabled_ops(const char* op_type_list) {
  SetDisabledOps(std::string(op_type_list));
}

extern const char* get_disabled_ops() {
  return ngraph::join(GetDisabledOps(), ",").c_str();
}

void enable_dynamic_fallback() { EnableDynamicFallback(); }
void disable_dynamic_fallback() { DisableDynamicFallback(); }

void disable_rewrite_pass() { DisableRewritePass(); }

bool export_ir(const char* output_dir, char** cluster_info, char** err_msg) {
  string str_cluster_info("");
  string str_err_msg("");
  if (!ExportIR(string(output_dir), str_cluster_info, str_err_msg)) {
    errMsg = strdup(str_err_msg.c_str());
    *err_msg = errMsg;
    return false;
  }
  clusterInfo = strdup(str_cluster_info.c_str());
  *cluster_info = clusterInfo;
  return true;
}
}

// note that TensorFlow always uses camel case for the C++ API, but not for
// Python
void Enable() { _is_enabled = true; }
void Disable() { _is_enabled = false; }
bool IsEnabled() { return _is_enabled; }

vector<string> ListBackends() { return BackendManager::GetSupportedBackends(); }

void SetBackend(const string& type) {
  Status exec_status = BackendManager::SetBackend(type);
  if (exec_status != Status::OK()) {
    throw runtime_error(exec_status.error_message());
  }
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

void EnableDynamicFallback() { NGraphClusterManager::EnableClusterFallback(); }

void DisableDynamicFallback() {
  NGraphClusterManager::DisableClusterFallback();
}

void DisableRewritePass() { _is_rewrite_pass_enabled = false; }

bool IsRewritePassEnabled() { return _is_rewrite_pass_enabled; }

bool ExportIR(const string& output_dir, string& cluster_info, string& err_msg) {
  struct stat st;
  if (stat(output_dir.c_str(), &st) != 0) {
    err_msg = "Directory \"" + output_dir + "\" does not exist.";
    return false;
  }

  // Export IR into the output directory
  NGraphClusterManager::ExportMRUIRs(output_dir);

  // Dump cluster info
  NGraphClusterManager::DumpClusterInfos(cluster_info);
  err_msg = "";
  return true;
}

}  // namespace api
}  // namespace openvino_tensorflow
}  // namespace tensorflow
