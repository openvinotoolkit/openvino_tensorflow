/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#ifndef OPENVINO_TF_CLUSTER_MANAGER_H_
#define OPENVINO_TF_CLUSTER_MANAGER_H_

#include <mutex>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"

#include "openvino_tensorflow/executable.h"

namespace tensorflow {
namespace openvino_tensorflow {

class NGraphClusterManager {
 public:
  static size_t NewCluster();
  static tensorflow::GraphDef* GetClusterGraph(size_t idx);
  static void EvictAllClusters();
  static size_t NumberOfClusters();
  static bool CheckClusterFallback(const size_t idx);
  static void SetClusterFallback(const size_t idx, const bool fallback);
  static void EnableClusterFallback();
  static void DisableClusterFallback();
  static bool IsClusterFallbackEnabled();
  static void SetMRUExecutable(const size_t idx,
                               std::shared_ptr<Executable> executable_ptr);
  static void ExportMRUIRs(const string& output_dir);
  static void ClearMRUClusters();
  static void SetClusterInfo(const size_t idx, const string cluster_info);
  static void DumpClusterInfos(string& cluster_infos);

 private:
  static std::vector<tensorflow::GraphDef*> s_cluster_graphs;
  static std::vector<std::shared_ptr<Executable>> s_mru_executables;
  static std::map<size_t, std::string> s_cluster_info;
  static std::vector<bool> s_cluster_fallback;
  static bool s_cluster_fallback_enabled;
  static std::mutex s_cluster_graphs_mutex;
};

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_CLUSTER_MANAGER_H_
