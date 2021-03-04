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
#ifndef NGRAPH_LIBRARY_MANAGER_H_
#define NGRAPH_LIBRARY_MANAGER_H_

#include <mutex>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {
namespace ngraph_bridge {

class NGraphClusterManager {
 public:
  static size_t NewCluster();
  static tensorflow::GraphDef* GetClusterGraph(size_t idx);
  static void EvictAllClusters();
  static size_t NumberOfClusters();

 private:
  static std::vector<tensorflow::GraphDef*> s_cluster_graphs;
  static std::mutex s_cluster_graphs_mutex;
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif
