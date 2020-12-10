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
#include "ngraph_bridge/ngraph_cluster_manager.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

// Static initializers
std::vector<GraphDef*> NGraphClusterManager::s_cluster_graphs;
std::mutex NGraphClusterManager::s_cluster_graphs_mutex;

size_t NGraphClusterManager::NewCluster() {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);

  size_t new_idx = s_cluster_graphs.size();
  s_cluster_graphs.push_back(new GraphDef());
  return new_idx;
}

GraphDef* NGraphClusterManager::GetClusterGraph(size_t idx) {
  std::lock_guard<std::mutex> guard(s_cluster_graphs_mutex);
  return idx < s_cluster_graphs.size() ? s_cluster_graphs[idx] : nullptr;
}

void NGraphClusterManager::EvictAllClusters() { s_cluster_graphs.clear(); }

}  // namespace ngraph_bridge
}  // namespace tensorflow
