/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include <fstream>
#include <iomanip>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"

// #include "logging/ngraph_log.h"
// #include "logging/tf_graph_writer.h"
// #include "ngraph_bridge/ngraph_api.h"
// #include "ngraph_bridge/ngraph_assign_clusters.h"
// #include "ngraph_bridge/ngraph_backend_manager.h"
// #include "ngraph_bridge/ngraph_capture_variables.h"
// #include "ngraph_bridge/ngraph_cluster_manager.h"
// #include "ngraph_bridge/ngraph_deassign_clusters.h"
// #include "ngraph_bridge/ngraph_encapsulate_clusters.h"
// #include "ngraph_bridge/ngraph_mark_for_clustering.h"
// #include "ngraph_bridge/ngraph_rewrite_for_tracking.h"
// #include "ngraph_bridge/ngraph_utils.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

void GraphToPbFile(Graph* graph, const string& filename) {
  GraphDef g_def;
  graph->ToGraphDef(&g_def);

  // string graph_pb_str;
  // protobuf::TextFormat::PrintToString(g_def, &graph_pb_str);
  std::ofstream ostrm_out(filename, std::ios_base::trunc);
  // ostrm_out << graph_pb_str;
  g_def.SerializeToOstream(&ostrm_out);
}

class NGraphOptimizationPass : public GraphOptimizationPass {
 public:
  virtual Status Run(const GraphOptimizationPassOptions& options) = 0;

 protected:
  std::string GraphFilenamePrefix(std::string kind, int idx) {
    std::stringstream ss;
    ss << kind << "_" << std::setfill('0') << std::setw(4) << idx;
    return ss.str();
  }

  std::string GraphFilenamePrefix(std::string kind, int idx, int sub_idx) {
    std::stringstream ss;
    ss << GraphFilenamePrefix(kind, idx) << "_" << std::setfill('0')
       << std::setw(4) << sub_idx;
    return ss.str();
  }

  void DumpGraphs(const GraphOptimizationPassOptions& options, int idx,
                  std::string filename_prefix) {
    // If we have a "main" graph, dump that.
    if (options.graph != nullptr) {
      auto pbtxt_filename = GraphFilenamePrefix(filename_prefix, idx) + ".pb";
      // VLOG(0) << "Dumping main graph to " << pbtxt_filename;
      GraphToPbFile(options.graph->get(), pbtxt_filename);
    }

    // If we have partition graphs (we shouldn't), dump those.
    if (options.partition_graphs != nullptr) {
      int sub_idx = 0;

      for (auto& kv : *options.partition_graphs) {
        auto pbtxt_filename =
            GraphFilenamePrefix(filename_prefix, idx, sub_idx) + ".pb";
        VLOG(0) << "Dumping subgraph " << sub_idx << " to " << pbtxt_filename;

        Graph* pg = kv.second.get();
        GraphToPbFile(pg, pbtxt_filename);

        sub_idx++;
      }
    }
  }
};

class NGraphPrePlacementPass : public NGraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    DumpGraphs(options, FreshIndex(), "pre_placement");
    return Status::OK();
  }

 private:
  // Returns a fresh "serial number" to avoid filename collisions in the graph
  // dumps.
  static int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

class NGraphPostPlacementPass : public NGraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    DumpGraphs(options, FreshIndex(), "post_placement");
    return Status::OK();
  }

 private:
  // Returns a fresh "serial number" to avoid filename collisions in the graph
  // dumps.
  static int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

class NGraphPostRewritePass : public NGraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    DumpGraphs(options, FreshIndex(), "post_rewrite");
    return Status::OK();
  }

 private:
  // Returns a fresh "serial number" to avoid filename collisions in the graph
  // dumps.
  static int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

class NGraphPostPartitionPass : public NGraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    DumpGraphs(options, FreshIndex(), "post_partition");
    return Status::OK();
  }

 private:
  // Returns a fresh "serial number" to avoid filename collisions in the graph
  // dumps.
  static int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

int NGraphPrePlacementPass::s_serial_counter = 0;
mutex NGraphPrePlacementPass::s_serial_counter_mutex;

int NGraphPostPlacementPass::s_serial_counter = 0;
mutex NGraphPostPlacementPass::s_serial_counter_mutex;

int NGraphPostRewritePass::s_serial_counter = 0;
mutex NGraphPostRewritePass::s_serial_counter_mutex;

int NGraphPostPartitionPass::s_serial_counter = 0;
mutex NGraphPostPartitionPass::s_serial_counter_mutex;

}  // namespace ngraph_bridge

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      ngraph_bridge::NGraphPrePlacementPass);

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      ngraph_bridge::NGraphPostPlacementPass);

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      ngraph_bridge::NGraphPostRewritePass);

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      ngraph_bridge::NGraphPostPartitionPass);
}  // namespace tensorflow