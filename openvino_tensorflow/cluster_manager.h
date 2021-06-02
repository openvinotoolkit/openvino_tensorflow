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

 private:
  static std::vector<tensorflow::GraphDef*> s_cluster_graphs;
  static std::vector<bool> s_cluster_fallback;
  static std::mutex s_cluster_graphs_mutex;
};

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_CLUSTER_MANAGER_H_
