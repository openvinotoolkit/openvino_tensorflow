/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include "openvino_tensorflow/cluster_manager.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

// Static initializers
std::vector<GraphDef*> OpenVINOClusterManager::s_cluster_graphs;
std::vector<bool> OpenVINOClusterManager::s_cluster_fallback;
std::vector<std::shared_ptr<Executable>>
    OpenVINOClusterManager::s_mru_executables;
std::mutex OpenVINOClusterManager::s_cluster_graphs_mutex;
bool OpenVINOClusterManager::s_cluster_fallback_enabled = true;
std::map<size_t, std::string> OpenVINOClusterManager::s_cluster_info;

size_t OpenVINOClusterManager::NewCluster() {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);

  size_t new_idx = s_cluster_graphs.size();
  s_cluster_graphs.push_back(new GraphDef());
  s_cluster_fallback.push_back(false);
  s_mru_executables.push_back(nullptr);
  return new_idx;
}

GraphDef* OpenVINOClusterManager::GetClusterGraph(size_t idx) {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);
  return idx < s_cluster_graphs.size() ? s_cluster_graphs[idx] : nullptr;
}

size_t OpenVINOClusterManager::NumberOfClusters() {
  return s_cluster_graphs.size();
}

void OpenVINOClusterManager::EvictAllClusters() {
  s_cluster_graphs.clear();
  s_cluster_fallback.clear();
}

void OpenVINOClusterManager::EvictMRUClusters() { s_mru_executables.clear(); }

bool OpenVINOClusterManager::CheckClusterFallback(const size_t idx) {
  return (s_cluster_fallback_enabled && idx < s_cluster_fallback.size())
             ? s_cluster_fallback[idx]
             : false;
}

void OpenVINOClusterManager::SetClusterFallback(const size_t idx,
                                              const bool fallback) {
  if (s_cluster_fallback_enabled && idx < s_cluster_fallback.size())
    s_cluster_fallback[idx] = fallback;
}

void OpenVINOClusterManager::EnableClusterFallback() {
  s_cluster_fallback_enabled = true;
}

void OpenVINOClusterManager::DisableClusterFallback() {
  s_cluster_fallback_enabled = false;
}

bool OpenVINOClusterManager::IsClusterFallbackEnabled() {
  return s_cluster_fallback_enabled;
}

void OpenVINOClusterManager::SetMRUExecutable(
    const size_t idx, std::shared_ptr<Executable> mru_executable_ptr) {
  if (idx < s_mru_executables.size())
    s_mru_executables[idx] = mru_executable_ptr;
}

void OpenVINOClusterManager::ExportMRUIRs(const string& output_dir) {
  for (int i = 0; i < s_mru_executables.size(); i++) {
    if (s_mru_executables[i]) s_mru_executables[i]->ExportIR(output_dir);
  }
}

void OpenVINOClusterManager::SetClusterInfo(const size_t idx,
                                          const string cluster_info) {
  s_cluster_info[idx] = cluster_info;
}

void OpenVINOClusterManager::DumpClusterInfos(string& cluster_infos) {
  cluster_infos = "";
  for (int i = 0; i < s_mru_executables.size(); i++) {
    if (s_mru_executables[i]) cluster_infos += s_cluster_info[i] + "\n";
  }
}

void OpenVINOClusterManager::ClearMRUClusters() {
  s_mru_executables.assign(s_mru_executables.size(), nullptr);
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
