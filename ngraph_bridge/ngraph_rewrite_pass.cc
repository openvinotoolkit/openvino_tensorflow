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

#include <iomanip>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"

#include "api.h"
#include "assign_clusters.h"
#include "cluster_manager.h"
#include "deassign_clusters.h"
#include "encapsulate_clusters.h"
#include "log.h"
#include "mark_for_clustering.h"
#include "ngraph_rewrite_pass.h"
#include "utils.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

int NGraphRewritePass::s_serial_counter = 0;
mutex NGraphRewritePass::s_serial_counter_mutex;

//
// Pass that rewrites the graph for nGraph operation.
//
// The pass has several phases, each executed in the below sequence:
//
//   1. Marking [mark_for_clustering.cc]
//   2. Cluster Assignment [assign_clusters.cc]
//   3. Cluster Deassignment [deassign_clusters.cc]
//   4. Cluster Encapsulation [encapsulate_clusters.cc]

Status NGraphRewritePass::Rewrite(
    Graph* graph, std::set<string> skip_these_nodes,
    std::unordered_map<std::string, std::string> config_map) {
  // For filename generation purposes, grab a fresh index. This is just an
  // arbitrary integer to avoid filename collisions resulting from subsequent
  // runs of this pass.
  int idx = FreshIndex();
  // If requested, dump unmarked graphs.
  util::DumpTFGraph(graph, idx, "unmarked");

  // If ngraph is disabled via ngraph_bridge api or NGRAPH_TF_DISABLE is set
  // we will not do anything; all subsequent
  // passes become a no-op.
  bool ngraph_not_enabled =
      (!api::IsEnabled()) || (std::getenv("NGRAPH_TF_DISABLE") != nullptr);
  bool already_processed = util::IsAlreadyProcessed(graph);
  if (!already_processed && ngraph_not_enabled) {
    NGRAPH_VLOG(0) << "NGraph is available but disabled.";
  }
  if (ngraph_not_enabled || already_processed) {
    NGRAPH_VLOG(1) << std::string("Rewrite pass will not run because ") +
                          (already_processed ? "graph is already preprocessed"
                                             : "ngraph is disabled");
    ClusterManager::EvictAllClusters();
    return Status::OK();
  }

  // Now Process the Graph

  // 1. Mark for clustering then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(MarkForClustering(graph, skip_these_nodes));
  util::DumpTFGraph(graph, idx, "marked");

  // 2. Assign clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(AssignClusters(graph));
  util::DumpTFGraph(graph, idx, "clustered");

  // 3. Deassign trivial clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(DeassignClusters(graph));
  util::DumpTFGraph(graph, idx, "declustered");

  // 4. Encapsulate clusters then, if requested, dump the graphs.
  auto status = EncapsulateClusters(graph, idx, config_map);
  if (status != Status::OK()) {
    return status;
  }

  util::DumpTFGraph(graph, idx, "encapsulated");
  return Status::OK();
}

Status NGraphRewritePass::Run(const GraphOptimizationPassOptions& options) {
  // If we don't get a main graph, log that fact and bail.
  if (options.graph == nullptr) {
    NGRAPH_VLOG(0) << "NGraphRewritePass: options.graph == nullptr";
    return Status::OK();
  }
  return Rewrite(options.graph->get());
}

}  // namespace ngraph_bridge

#ifndef NGRAPH_TF_USE_GRAPPLER_OPTIMIZER
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      ngraph_bridge::NGraphRewritePass);
#endif
}  // namespace tensorflow