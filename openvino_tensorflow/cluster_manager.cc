/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include "openvino_tensorflow/cluster_manager.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

// Static initializers
std::vector<GraphDef*> NGraphClusterManager::s_cluster_graphs;
std::vector<bool> NGraphClusterManager::s_cluster_fallback;
std::mutex NGraphClusterManager::s_cluster_graphs_mutex;
bool NGraphClusterManager::s_cluster_fallback_enabled = true;

size_t NGraphClusterManager::NewCluster() {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);

  size_t new_idx = s_cluster_graphs.size();
  s_cluster_graphs.push_back(new GraphDef());
  s_cluster_fallback.push_back(false);
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

}  // namespace openvino_tensorflow
}  // namespace tensorflow
