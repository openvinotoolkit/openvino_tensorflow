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

#include <iomanip>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

#include "ngraph_bridge/grappler/ngraph_optimizer.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_cluster_manager.h"

#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

#include <iostream>

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

Status NgraphOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  const auto params = config->parameter_map();
  for (size_t i = 0; i < compulsory_attrs.size(); i++) {
    if (params.count(compulsory_attrs[i]) == 0) {
      NGRAPH_VLOG(0) << "NGTF_OPTIMIZER: Compulsory attribute "
                     << compulsory_attrs[i] << " not found.";
      return errors::Internal("NGTF_OPTIMIZER: Missing compulsory attributes.");
    }
  }
  config_backend_name = params.at("ngraph_backend").s();
  config_device_id = params.at("device_id").s();
  NGRAPH_VLOG(3) << "Backend name from config: " << config_backend_name;
  std::set<ShapeHintMap> shape_hints;
  // typedef std::map<std::string, std::vector<int>> ShapeHintMap;
  for (auto i : params) {
    if (i.first != "ngraph_backend") {
      // TODO: slightly hacky. The bridge reserves the right to use optional
      // attributes whose names start with shape_hint
      if (i.first.rfind("shape_hint", 0) != 0) {
        config_map[(i.first == "device_id" ? "" : "_") +
                   std::string("ngraph_") + i.first] = i.second.s();
        NGRAPH_VLOG(3) << "Attribute: " << i.first
                       << " Value: " << config_map["_ngraph_" + i.first];
      } else {
        ShapeHintMap hint;
        for (auto k : i.second.func().attr().at("hint_body").func().attr()) {
          vector<int> full_or_partial_shape;
          for (auto dim : k.second.tensor().int_val()) {
            full_or_partial_shape.push_back(dim);
          }
          hint[k.first] = full_or_partial_shape;
        }
        shape_hints.insert(hint);
      }
    }
  }
  auto itr = params.find("aot_requested");
  bool do_aot = false;
  if (itr != params.end()) {
    do_aot = itr->second.s() == "1";
  }
  if (!do_aot && shape_hints.size() > 0) {
    return errors::Internal(
        "Did not requested AOT, but passed shape hints. Please request to use "
        "shape hints (by using --precompile in tf2ngraph.py), or if AOT is not "
        "desired then do not pass shape hints");
  }
  aot_info = make_pair(do_aot, shape_hints);
  return Status::OK();
}

Status NgraphOptimizer::Optimize(tensorflow::grappler::Cluster* cluster,
                                 const tensorflow::grappler::GrapplerItem& item,
                                 GraphDef* output) {
  NGRAPH_VLOG(3) << "NGTF_OPTIMIZER: Here at NgraphOptimizer ";
  NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: grappler item id " << item.id;

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
  // we will not do anything; all subsequent passes become a no-op.
  bool ngraph_not_enabled =
      (!config::IsEnabled()) || (std::getenv("NGRAPH_TF_DISABLE") != nullptr);
  bool already_processed = IsProcessedByNgraphPass(&graph);
  if (!already_processed && ngraph_not_enabled) {
    NGRAPH_VLOG(0) << "NGraph is available but disabled.";
  }
  if (ngraph_not_enabled || already_processed) {
    NGRAPH_VLOG(1) << std::string("Rewrite pass will not run because ") +
                          (already_processed ? "graph is already preprocessed"
                                             : "ngraph is disabled");
    NGraphClusterManager::EvictAllClusters();
    graph.ToGraphDef(output);
    return Status::OK();
  }

  // TODO: Find out a better way to preserve feed nodes, init_ops and
  // keep_ops instead of just skipping those from clustering.
  // Get nodes to be preserved/skipped
  std::set<string> nodes_to_preserve;

  // Feed Nodes
  for (size_t i = 0; i < item.feed.size(); i++) {
    nodes_to_preserve.insert(item.feed[i].first);
  }

  // Keep Ops
  nodes_to_preserve.insert(item.keep_ops.begin(), item.keep_ops.end());

  // Init Ops
  nodes_to_preserve.insert(item.init_ops.begin(), item.init_ops.end());

  // Find a list of nodes that are of the types that are disabled
  std::set<string> disabled_nodes;
  std::set<string> disabled_ops_set = config::GetDisabledOps();
  for (auto itr : graph.nodes()) {
    if (disabled_ops_set.find(itr->type_string()) != disabled_ops_set.end()) {
      disabled_nodes.insert(itr->name());
    }
  }

  // Fetch Nodes
  std::set<string> fetch_nodes;
  for (const string& f : item.fetch) {
    int pos = f.find(":");
    fetch_nodes.insert(f.substr(0, pos));
  }

  // nodes_to_add_identity_to = fetch_nodes - disabled_nodes
  std::set<string> nodes_to_add_identity_to;
  std::set_difference(fetch_nodes.begin(), fetch_nodes.end(),
                      disabled_nodes.begin(), disabled_nodes.end(),
                      std::inserter(nodes_to_add_identity_to,
                                    nodes_to_add_identity_to.begin()));

  // Rewrite graph to add IdentityN node so the fetch node can be encapsulated
  // as well
  // If the fetch node in question has 0 outputs or any of the outputs
  // has ref type as a data type then don't add IdentityN node, but the fetch
  // node will be skipped from capturing and marking for clustering.
  TF_RETURN_IF_ERROR(AddIdentityN(&graph, nodes_to_add_identity_to));

  nodes_to_preserve.insert(nodes_to_add_identity_to.begin(),
                           nodes_to_add_identity_to.end());
  std::set<string>& skip_these_nodes = nodes_to_preserve;

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

  // Get backend + its configurations, to be attached to the nodes
  // using RewriteConfig
  string backend_creation_string = BackendManager::GetBackendCreationString(
      config_backend_name, config_device_id);

  // Override from the env. for debugging purposes
  if (std::getenv("NGRAPH_TF_BACKEND") != nullptr) {
    backend_creation_string = std::getenv("NGRAPH_TF_BACKEND");
  }

  TF_RETURN_IF_ERROR(BackendManager::CanCreateBackend(backend_creation_string));
  NGRAPH_VLOG(1) << "Setting backend from the RewriteConfig "
                 << backend_creation_string;

  NGRAPH_VLOG(0) << "NGraph using backend: " << backend_creation_string;

  // 1. Mark for clustering then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(
      MarkForClustering(&graph, skip_these_nodes, backend_creation_string));
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
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  TF_RETURN_IF_ERROR(
      // TODO: right now _ngraph_aot_requested is passed along in config_map.
      EncapsulateClusters(&graph, idx, fdeflib_new, config_map, aot_info));
  if (DumpEncapsulatedGraphs()) {
    DumpGraphs(graph, idx, "encapsulated", "Graph with Clusters Encapsulated");
  }

  // Rewrite for tracking then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(RewriteForTracking(&graph, idx));
  if (DumpTrackedGraphs()) {
    DumpGraphs(graph, idx, "tracked",
               "Graph with Variables Rewritten for Tracking");
  }

  // Convert the graph back to Graphdef
  graph.ToGraphDef(output);
  // According to the doc, the message takes ownership of the allocated object
  // https://developers.google.com/protocol-buffers/docs/reference/cpp-generated#proto3_string
  // Hence no need to free fdeflib_new
  output->set_allocated_library(fdeflib_new);
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
  NGRAPH_VLOG(0) << "NGTF_OPTIMIZER: Dumping main graph to " << dot_filename;
  NGRAPH_VLOG(0) << "NGTF_OPTIMIZER: Dumping main graph to " << pbtxt_filename;

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