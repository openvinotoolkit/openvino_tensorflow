/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include "openvino_tensorflow/cluster_manager.h"
#include <memory>

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

// Static initializers
std::vector<GraphDef*> NGraphClusterManager::s_cluster_graphs;
std::vector<bool> NGraphClusterManager::s_cluster_fallback;
std::vector<std::shared_ptr<Executable>> NGraphClusterManager::s_mru_executables;
std::mutex NGraphClusterManager::s_cluster_graphs_mutex;
bool NGraphClusterManager::s_cluster_fallback_enabled = true;
int NGraphClusterManager::cluster_size = 0;
std::map<size_t, std::string> NGraphClusterManager::s_cluster_info;

size_t NGraphClusterManager::NewCluster() {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);

  size_t new_idx = s_cluster_graphs.size();
  s_cluster_graphs.push_back(new GraphDef());
  s_cluster_fallback.push_back(false);
  s_mru_executables.push_back(nullptr);
  return new_idx;
}

GraphDef* NGraphClusterManager::GetClusterGraph(size_t idx) {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);
  return idx < s_cluster_graphs.size() ? s_cluster_graphs[idx] : nullptr;
}

size_t NGraphClusterManager::NumberOfClusters() {
  return s_cluster_graphs.size();
}

void NGraphClusterManager::EvictAllClusters() {
  s_cluster_graphs.clear();
  s_cluster_fallback.clear();
}

bool NGraphClusterManager::CheckClusterFallback(const size_t idx) {
  return (s_cluster_fallback_enabled && idx < s_cluster_fallback.size())
             ? s_cluster_fallback[idx]
             : false;
}

void NGraphClusterManager::SetClusterFallback(const size_t idx,
                                              const bool fallback) {
  if (s_cluster_fallback_enabled && idx < s_cluster_fallback.size())
    s_cluster_fallback[idx] = fallback;
}

void NGraphClusterManager::EnableClusterFallback() {
  s_cluster_fallback_enabled = true;
}

void NGraphClusterManager::DisableClusterFallback() {
  s_cluster_fallback_enabled = false;
}

bool NGraphClusterManager::IsClusterFallbackEnabled() {
  return s_cluster_fallback_enabled;
}

void NGraphClusterManager::SetMRUExecutable(const size_t idx, std::shared_ptr<Executable> mru_executable_ptr) {
  if (idx < s_mru_executables.size())
    s_mru_executables[idx] = mru_executable_ptr;
}

void NGraphClusterManager::ExportMRUIRs(const string& output_dir) {
  for (int i=0; i<s_mru_executables.size(); i++) {
    if (s_mru_executables[i]) s_mru_executables[i]->ExportIR(output_dir);
  }
}

void NGraphClusterManager::SetClusterInfo(const size_t idx, const string cluster_info) {
    s_cluster_info[idx] = cluster_info;
}

void NGraphClusterManager::DumpClusterInfos(string &cluster_infos) {
    cluster_infos = "";
    for (auto it = s_cluster_info.begin(); it != s_cluster_info.end(); it++) {
        cluster_infos += it->second + "\n";
    }
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
