/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <dirent.h>
#include <sys/stat.h>

#include "tensorflow/core/lib/core/errors.h"

#include "api.h"
#include "backend_manager.h"

namespace tensorflow {
namespace openvino_tensorflow {
namespace api {

static bool _is_enabled = true;
static bool _is_logging_placement = false;
static std::set<std::string> disabled_op_types{};
static char* backendName;
static char* backendList[4];
static char* clusterInfo;

extern "C" {
void enable() { Enable(); }
void disable() { Disable(); }
bool is_enabled() { return IsEnabled(); }

bool check_backend(char* backend) {
  const char* devices[4] = {"CPU", "GPU", "MYRIAD", "VAD-M"};
  for (int i = 0; i < 4; i++) {
    if (strcmp(backend, devices[i]) == 0) return true;
  }
  return false;
}
size_t backends_len() {
  const auto ovtf_backends = ListBackends();
  int backends_count = 0;
  for (size_t idx = 0; idx < ovtf_backends.size(); idx++) {
    backendList[idx] = strdup(ovtf_backends[idx].c_str());
    if (check_backend(backendList[idx])) backends_count++;
  }
  return backends_count;
}

bool list_backends(char** backends) {
  const auto ovtf_backends = ListBackends();
  int i = 0;
  for (size_t idx = 0; idx < ovtf_backends.size(); idx++) {
    backendList[idx] = strdup(ovtf_backends[idx].c_str());
    if (check_backend(backendList[idx])) backends[i++] = backendList[idx];
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
void freeBackend() { free(backendName); }
void start_logging_placement() { StartLoggingPlacement(); }
void stop_logging_placement() { StopLoggingPlacement(); }
bool is_logging_placement() { return IsLoggingPlacement(); }
void freeClusterInfo() { free(clusterInfo); }

extern void set_disabled_ops(const char* op_type_list) {
  SetDisabledOps(std::string(op_type_list));
}

extern const char* get_disabled_ops() {
  return ngraph::join(GetDisabledOps(), ",").c_str();
}

void enable_dynamic_fallback() { EnableDynamicFallback(); }
void disable_dynamic_fallback() { DisableDynamicFallback(); }

bool export_ir(const char* output_dir, char** cluster_info,
               bool confirm_before_overwrite) {
  string str_cluster_info("");
  if (!ExportIR(string(output_dir), str_cluster_info,
                confirm_before_overwrite)) {
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

void EnableDynamicFallback() { NGraphClusterManager::EnableClusterFallback(); }

void DisableDynamicFallback() {
  NGraphClusterManager::DisableClusterFallback();
}

bool ExportIR(const string& output_dir, string& cluster_info,
              bool confirm_before_overwrite) {
  // Create the directory/directories
  char* tmp = new char[output_dir.size()];
  char* p = NULL;
  size_t len;
  int dir_err;
  struct stat st;

  snprintf(tmp, sizeof(tmp) * output_dir.size(), "%s", output_dir.c_str());
  len = strlen(tmp);
  if (tmp[len - 1] == '/') tmp[len - 1] = 0;
  for (p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      if (stat(tmp, &st) != 0) {
        dir_err = mkdir(tmp, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
          delete[] tmp;
          return false;
        }
      }
      *p = '/';
    }
  }
  if (stat(tmp, &st) != 0) {
    dir_err = mkdir(tmp, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == dir_err) {
      delete[] tmp;
      return false;
    }
  }
  delete[] tmp;

  // Check for existing files
  if (confirm_before_overwrite) {
    DIR* dir;
    struct dirent* diread;
    vector<char*> files;

    if ((dir = opendir(output_dir.c_str())) != nullptr) {
      while ((diread = readdir(dir)) != nullptr) {
        files.push_back(diread->d_name);
      }
      closedir(dir);
    } else {
      perror("opendir");
      return false;
    }
    for (auto file : files) {
      size_t len_file = strlen(file);
      if (len_file > 17 && strncmp(file, "ovtf_cluster_", 13) == 0 &&
          (strncmp((file + len_file - 4), ".xml", 4) ||
           strncmp((file + len_file - 4), ".bin", 4))) {
        std::cout << "There are existing IR files in the directory you "
                     "specified. New IR files may overwrite on the existing "
                     "files. Do you want to continue? (y|n)"
                  << endl;
        int decision;
        decision = getchar();
        if (decision == 'y') {
          break;
        } else {
          return true;
        }
      }
    }
  }

  // Export IR into the output directory
  NGraphClusterManager::ExportMRUIRs(output_dir);

  // Dump cluster info
  NGraphClusterManager::DumpClusterInfos(cluster_info);
  return true;
}

}  // namespace api
}  // namespace openvino_tensorflow
}  // namespace tensorflow
