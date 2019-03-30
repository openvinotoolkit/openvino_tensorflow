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

#include "ngraph_optimizer.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

#include <iomanip>
#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

#include <iostream>

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

Status NgraphOptimizer::Optimize(tensorflow::grappler::Cluster* cluster,
                                 const tensorflow::grappler::GrapplerItem& item,
                                 GraphDef* output) {
  NGRAPH_VLOG(5) << "Here at NgraphOptimizer ";
  NGRAPH_VLOG(5) << "NgraphOptimizer : grappler item id " << item.id;

  // Convert the GraphDef to Graph
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = true;
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, item.graph, &graph));

  // For filename generation purposes, grab a fresh index. This is just an
  // arbitrary integer to avoid filename collisions resulting from subsequent
  // runs of this pass.
  int idx = FreshIndex();

  // If requested, dump pre-capture graphs.
  if (DumpPrecaptureGraphs()) {
    DumpGraphs(graph, idx, "precapture", "Pre-Capture Graph");
  }

  // If ngraph is disabled via ngraph_bridge api or NGRAPH_TF_DISABLE is set
  // we will not do anything; all subsequent
  // passes become a no-op.
  if (config::IsEnabled() == false ||
      std::getenv("NGRAPH_TF_DISABLE") != nullptr) {
    NGRAPH_VLOG(0) << "Ngraph is disabled ";
    return Status::OK();
  }

  // Get the nodes to be skipped
  std::vector<string> fetch_nodes;
  for (const string& f : item.fetch) {
    NGRAPH_VLOG(5) << "Skip fetch node: " << f;
    int pos = f.find(":");
    fetch_nodes.push_back(f.substr(0, pos));
  }
  std::vector<string>& skip_these_nodes = fetch_nodes;

  //
  // Variable capture: Part that replaces all instances of VariableV2 with the
  // NGraphVariable op. Making this replacement allows us to substitute in a
  // kernel that tracks the freshness of variables (invalidating freshness when
  // the reference is handed off to an "untrusted" op).
  //

  // Do variable capture then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(CaptureVariables(&graph, skip_these_nodes));
  if (DumpCapturedGraphs()) {
    DumpGraphs(graph, idx, "captured", "Graph With Variables Captured");
  }

  //
  // Encapsulation: Part that rewrites the graph for nGraph operation.
  //
  // The part has several phases, each executed in sequence:
  //
  //   1. Marking [ngraph_mark_for_clustering.cc]
  //   2. Cluster Assignment [ngraph_assign_clusters.cc]
  //   3. Cluster Deassignment [ngraph_deassign_clusters.cc]
  //   4. Cluster Encapsulation [ngraph_encapsulate_clusters.cc] - currently
  //      part of the ngraph_rewrite_pass.cc to be executed after POST_REWRITE
  //
  // Between phases, graph dumps (in both .dot and .pbtxt format) may be
  // requested by setting the following environment variables:
  //
  //   NGRAPH_TF_DUMP_UNMARKED_GRAPHS=1      dumps graphs before phase 1
  //   NGRAPH_TF_DUMP_MARKED_GRAPHS=1        dumps graphs after phase 1
  //   NGRAPH_TF_DUMP_CLUSTERED_GRAPHS=1     dumps graphs after phase 2
  //   NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS=1   dumps graphs after phase 3
  //   NGRAPH_TF_DUMP_ENCAPSULATED_GRAPHS=1  dumps graphs after phase 4
  //   NGRAPH_TF_DUMP_GRAPHS=1               all of the above
  //

  // If requested, dump unmarked graphs.
  if (DumpUnmarkedGraphs()) {
    DumpGraphs(graph, idx, "unmarked", "Unmarked Graph");
  }

  // 1. Mark for clustering then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(MarkForClustering(&graph, skip_these_nodes));
  if (DumpMarkedGraphs()) {
    DumpGraphs(graph, idx, "marked", "Graph Marked for Clustering");
  }

  // 2. Assign clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(AssignClusters(&graph));
  if (DumpClusteredGraphs()) {
    DumpGraphs(graph, idx, "clustered", "Graph with Clusters Assigned");
  }

  // 3. Deassign trivial clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(DeassignClusters(&graph));
  if (DumpDeclusteredGraphs()) {
    DumpGraphs(graph, idx, "declustered",
               "Graph with Trivial Clusters De-Assigned");
  }

  // 4. Encapsulate clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(EncapsulateClusters(&graph));
  if (DumpEncapsulatedGraphs()) {
    DumpGraphs(graph, idx, "encapsulated", "Graph with Clusters Encapsulated");
  }

  // Rewrite for tracking then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(RewriteForTracking(&graph));
  if (DumpTrackedGraphs()) {
    DumpGraphs(graph, idx, "tracked",
               "Graph with Variables Rewritten for Tracking");
  }

  // Convert the graph back to Graphdef
  graph.ToGraphDef(output);
  return Status::OK();
}

void NgraphOptimizer::Feedback(tensorflow::grappler::Cluster* cluster,
                               const tensorflow::grappler::GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // no-op
}

void NgraphOptimizer::DumpGraphs(Graph& graph, int idx,
                                 std::string filename_prefix,
                                 std::string title) {
  // If we have a "main" graph, dump that.
  auto dot_filename = DotFilename(filename_prefix, idx);
  auto pbtxt_filename = PbtxtFilename(filename_prefix, idx);
  NGRAPH_VLOG(0) << "NGRAPH-TF OPTIMIZER Dumping main graph to "
                 << dot_filename;
  NGRAPH_VLOG(0) << "NGRAPH-TF OPTIMIZER Dumping main graph to "
                 << pbtxt_filename;

  GraphToDotFile(&graph, dot_filename, title);
  GraphToPbTextFile(&graph, pbtxt_filename);
}

int NgraphOptimizer::FreshIndex() {
  mutex_lock l(s_serial_counter_mutex);
  return s_serial_counter++;
}

REGISTER_GRAPH_OPTIMIZER_AS(NgraphOptimizer, "ngraph-optimizer");

}  // end namespace ngraph_bridge

}  // end namespace tensorflow
