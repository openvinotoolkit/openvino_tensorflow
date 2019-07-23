/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"

#include "logging/ngraph_log.h"
#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_enter_in_catalog.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_remove_ngraphassigns.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_replace_variable_modifiers.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_cluster_manager.h"
#include "ngraph_bridge/ngraph_deassign_clusters.h"
#include "ngraph_bridge/ngraph_encapsulate_clusters.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_rewrite_for_tracking.h"
#include "ngraph_bridge/ngraph_utils.h"

#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

class NGraphRewritePass : public GraphOptimizationPass {
 public:
  virtual Status Run(const GraphOptimizationPassOptions& options) = 0;

 protected:
  void DumpGraphs(const GraphOptimizationPassOptions& options, int idx,
                  std::string filename_prefix, std::string title) {
    // If we have a "main" graph, dump that.
    if (options.graph != nullptr) {
      auto dot_filename = DotFilename(filename_prefix, idx);
      auto pbtxt_filename = PbtxtFilename(filename_prefix, idx);
      NGRAPH_VLOG(0) << "Dumping main graph to " << dot_filename;
      NGRAPH_VLOG(0) << "Dumping main graph to " << pbtxt_filename;

      GraphToDotFile(options.graph->get(), dot_filename, title);
      GraphToPbTextFile(options.graph->get(), pbtxt_filename);
    }

    // If we have partition graphs (we shouldn't), dump those.
    if (options.partition_graphs != nullptr) {
      int sub_idx = 0;

      for (auto& kv : *options.partition_graphs) {
        auto dot_filename = DotFilename(filename_prefix, idx, sub_idx);
        auto pbtxt_filename = PbtxtFilename(filename_prefix, idx, sub_idx);
        NGRAPH_VLOG(0) << "Dumping subgraph " << sub_idx << " to "
                       << dot_filename;
        NGRAPH_VLOG(0) << "Dumping subgraph " << sub_idx << " to "
                       << pbtxt_filename;

        Graph* pg = kv.second.get();

        GraphToDotFile(pg, dot_filename, title);
        GraphToPbTextFile(pg, pbtxt_filename);

        sub_idx++;
      }
    }
  }

  // Returns a fresh "serial number" to avoid filename collisions in the graph
  // dumps.
  static int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static bool DumpAllGraphs() {
    return std::getenv("NGRAPH_TF_DUMP_GRAPHS") != nullptr;
  }

 private:
  static std::string DotFilename(std::string kind, int idx) {
    return GraphFilenamePrefix(kind, idx) + ".dot";
  }
  static std::string PbtxtFilename(std::string kind, int idx) {
    return GraphFilenamePrefix(kind, idx) + ".pbtxt";
  }
  static std::string DotFilename(std::string kind, int idx, int sub_idx) {
    return GraphFilenamePrefix(kind, idx, sub_idx) + ".dot";
  }
  static std::string PbtxtFilename(std::string kind, int idx, int sub_idx) {
    return GraphFilenamePrefix(kind, idx, sub_idx) + ".pbtxt";
  }
  static std::string GraphFilenamePrefix(std::string kind, int idx) {
    std::stringstream ss;
    ss << kind << "_" << std::setfill('0') << std::setw(4) << idx;
#if defined NGRAPH_DISTRIBUTED
    int rank_id = ngraph::get_distributed_interface()->get_rank();
    ss << "_" << std::setfill('0') << std::setw(4) << rank_id;
#endif
    return ss.str();
  }
  static std::string GraphFilenamePrefix(std::string kind, int idx,
                                         int sub_idx) {
    std::stringstream ss;
    ss << GraphFilenamePrefix(kind, idx) << "_" << std::setfill('0')
       << std::setw(4) << sub_idx;
#if defined NGRAPH_DISTRIBUTED
    int rank_id = ngraph::get_distributed_interface()->get_rank();
    ss << "_" << std::setfill('0') << std::setw(4) << rank_id;
#endif
    return ss.str();
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

int NGraphRewritePass::s_serial_counter = 0;
mutex NGraphRewritePass::s_serial_counter_mutex;

//
// The variable capture pass replaces all instances of VariableV2 with the
// NGraphVariable op. Making this replacement allows us to substitute in a
// kernel that tracks the freshness of variables (invalidating freshness when
// the reference is handed off to an "untrusted" op).
//
class NGraphVariableCapturePass : public NGraphRewritePass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    // If we don't get a main graph, log that fact and bail.
    if (options.graph == nullptr) {
      NGRAPH_VLOG(0) << "NGraphVariableCapturePass: options.graph == nullptr";
      return Status::OK();
    }

    // For filename generation purposes, grab a fresh index. This is just an
    // arbitrary integer to avoid filename collisions resulting from subsequent
    // runs of this pass.
    int idx = FreshIndex();

    // If requested, dump pre-capture graphs.
    if (DumpPrecaptureGraphs()) {
      DumpGraphs(options, idx, "precapture", "Pre-Capture Graph");
    }

    // If ngraph is disabled via ngraph_bridge api or NGRAPH_TF_DISABLE is set
    // we will not do anything; all subsequent
    // passes become a no-op.
    bool ngraph_not_enabled =
        (!config::IsEnabled()) || (std::getenv("NGRAPH_TF_DISABLE") != nullptr);
    bool already_processed = IsProcessedByNgraphPass(options.graph->get());
    if (ngraph_not_enabled || already_processed) {
      NGRAPH_VLOG(1) << "Not running through nGraph. nGraph not enabled: "
                     << ngraph_not_enabled
                     << " Already processed: " << already_processed;
      NGraphClusterManager::EvictAllClusters();
      return Status::OK();
    }

    // Do variable capture then, if requested, dump the graphs.
    std::set<string> skip_these_nodes = {};
    TF_RETURN_IF_ERROR(
        CaptureVariables(options.graph->get(), skip_these_nodes));
    if (DumpCapturedGraphs()) {
      DumpGraphs(options, idx, "captured", "Graph With Variables Captured");
    }

    return Status::OK();
  }

 private:
  static bool DumpPrecaptureGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_PRE_CAPTURED_GRAPHS") != nullptr;
  }
  static bool DumpCapturedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_CAPTURED_GRAPHS") != nullptr;
  }
};

//
// Pass that rewrites the graph for nGraph operation.
//
// The pass has several phases, each executed in sequence:
//
//   0. Replace Modifiers [ngraph_replace_variable_modifiers.cc]
//   1. Marking [ngraph_mark_for_clustering.cc]
//   2. Cluster Assignment [ngraph_assign_clusters.cc]
//   3. Cluster Deassignment [ngraph_deassign_clusters.cc]
//   4. Cluster Encapsulation [ngraph_encapsulate_clusters.cc]
//   5. Rewrite Variable Type Ops for Tracking [ngraph_rewrite_for_tracking.cc]
//   6. Enter In Catalog  [ngraph_enter_in_catalog.cc]
//   7. Remove NGraphAssigns [ngraph_remove_ngraphassigns.cc]
// Between phases, graph dumps (in both .dot and .pbtxt format) may be
// requested by setting the following environment variables:
//
//   NGRAPH_TF_DUMP_UNMARKED_GRAPHS=1            dumps graphs before phase 0
//   NGRAPH_TF_DUMP_REPLACEDMODIFIERS_GRAPHS=1   dumps graphs after phase 0
//   NGRAPH_TF_DUMP_MARKED_GRAPHS=1              dumps graphs after phase 1
//   NGRAPH_TF_DUMP_CLUSTERED_GRAPHS=1           dumps graphs after phase 2
//   NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS=1         dumps graphs after phase 3
//   NGRAPH_TF_DUMP_ENCAPSULATED_GRAPHS=1        dumps graphs after phase 4
//   NGRAPH_TF_DUMP_TRACKED_GRAPHS=1             dumps graphs after phase 5
//   NGRAPH_TF_DUMP_CATALOGED_GRAPHS=1           dumps graphs after phase 6
//   NGRAPH_TF_DUMP_REMOVENGASSIGNS_GRAPHS=1     dumps graphs after phase 7
//   NGRAPH_TF_DUMP_GRAPHS=1                     all of the above
//
class NGraphEncapsulationPass : public NGraphRewritePass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    // If we don't get a main graph, log that fact and bail.
    if (options.graph == nullptr) {
      NGRAPH_VLOG(0) << "NGraphEncapsulationPass: options.graph == nullptr";
      return Status::OK();
    }

    // For filename generation purposes, grab a fresh index. This is just an
    // arbitrary integer to avoid filename collisions resulting from subsequent
    // runs of this pass.
    int idx = FreshIndex();

    // If requested, dump unmarked graphs.
    if (DumpUnmarkedGraphs()) {
      DumpGraphs(options, idx, "unmarked", "Unmarked Graph");
    }

    // If ngraph is disabled via ngraph_bridge api or NGRAPH_TF_DISABLE is set
    // we will not do anything; all subsequent
    // passes become a no-op.
    bool ngraph_not_enabled =
        (!config::IsEnabled()) || (std::getenv("NGRAPH_TF_DISABLE") != nullptr);
    bool already_processed = IsProcessedByNgraphPass(options.graph->get());
    if (!already_processed && ngraph_not_enabled) {
      NGRAPH_VLOG(0) << "NGraph is available but disabled.";
    }
    if (ngraph_not_enabled || already_processed) {
      NGRAPH_VLOG(1) << std::string("Rewrite pass will not run because ") +
                            (already_processed ? "graph is already preprocessed"
                                               : "ngraph is disabled");
      NGraphClusterManager::EvictAllClusters();
      return Status::OK();
    }

    // Get backend + its configurations
    // to be attached to the nodes
    // Precedence Order: Env Variable > BackendManager
    std::unordered_map<std::string, std::string> config_map;
    string backend_name;
    TF_RETURN_IF_ERROR(
        BackendManager::GetCurrentlySetBackendName(&backend_name));

    // splits into {"ngraph_backend", "_ngraph_device_config"}
    config_map = BackendManager::GetBackendAttributeValues(
        backend_name);  // SplitBackendConfig
    backend_name = config_map.at("ngraph_backend");
    config_map.erase("ngraph_backend");
    NGRAPH_VLOG(0) << "NGraph using backend: " << backend_name;

    // Now Process the Graph

    // 0. Replace optimizers then, if requested, dump the graphs.
    TF_RETURN_IF_ERROR(ReplaceModifiers(options.graph->get(), idx));
    if (DumpReplacedModifiersGraphs()) {
      DumpGraphs(options, idx, "replaced_modifier",
                 "Graph with Modifiers replaced");
    }

    // 1. Mark for clustering then, if requested, dump the graphs.
    std::set<string> skip_these_nodes = {};
    TF_RETURN_IF_ERROR(MarkForClustering(options.graph->get(), skip_these_nodes,
                                         backend_name));
    if (DumpMarkedGraphs()) {
      DumpGraphs(options, idx, "marked", "Graph Marked for Clustering");
    }

    // 2. Assign clusters then, if requested, dump the graphs.
    TF_RETURN_IF_ERROR(AssignClusters(options.graph->get()));
    if (DumpClusteredGraphs()) {
      DumpGraphs(options, idx, "clustered", "Graph with Clusters Assigned");
    }

    // 3. Deassign trivial clusters then, if requested, dump the graphs.
    TF_RETURN_IF_ERROR(DeassignClusters(options.graph->get()));
    if (DumpDeclusteredGraphs()) {
      DumpGraphs(options, idx, "declustered",
                 "Graph with Trivial Clusters De-Assigned");
    }

    // 4. Encapsulate clusters then, if requested, dump the graphs.
    FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
    TF_RETURN_IF_ERROR(EncapsulateClusters(options.graph->get(), idx,
                                           fdeflib_new, config_map));
    // TODO: not using fdeflib_new in this path. Only grappler path uses it
    free(fdeflib_new);
    if (DumpEncapsulatedGraphs()) {
      DumpGraphs(options, idx, "encapsulated",
                 "Graph with Clusters Encapsulated");
    }

    // Rewrite for tracking then, if requested, dump the graphs.
    TF_RETURN_IF_ERROR(RewriteForTracking(options.graph->get(), idx));
    if (DumpTrackedGraphs()) {
      DumpGraphs(options, idx, "tracked",
                 "Graph with Variables Rewritten for Tracking");
    }

    // Enter in catalog then.
    TF_RETURN_IF_ERROR(EnterInCatalog(options.graph->get(), idx));
    if (DumpCatalogedGraphs()) {
      DumpGraphs(options, idx, "cataloged",
                 "Graph with Variables Inputs Entered in Catalog");
    }

    // Remove Certain NGraphAssigns then.
    TF_RETURN_IF_ERROR(RemoveNGraphAssigns(options.graph->get()));
    if (DumpRemoveNGraphAssignsGraphs()) {
      DumpGraphs(options, idx, "ngraphssigns_optimized",
                 "Graph with NGraphAssigns Optimized/Removed");
    }

    return Status::OK();
  }

 private:
  static bool DumpUnmarkedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_UNMARKED_GRAPHS") != nullptr;
  }
  static bool DumpReplacedModifiersGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_REPLACEDMODIFIERS_GRAPHS") != nullptr;
  }
  static bool DumpMarkedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_MARKED_GRAPHS") != nullptr;
  }
  static bool DumpClusteredGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_CLUSTERED_GRAPHS") != nullptr;
  }
  static bool DumpDeclusteredGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS") != nullptr;
  }
  static bool DumpEncapsulatedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_ENCAPSULATED_GRAPHS") != nullptr;
  }
  static bool DumpTrackedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_TRACKED_GRAPHS") != nullptr;
  }

  static bool DumpCatalogedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_CATALOGED_GRAPHS") != nullptr;
  }

  static bool DumpRemoveNGraphAssignsGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_REMOVENGASSIGNS_GRAPHS") != nullptr;
  }
};

}  // namespace ngraph_bridge

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      ngraph_bridge::NGraphVariableCapturePass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      ngraph_bridge::NGraphEncapsulationPass);
}  // namespace tensorflow
