/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

#include "openvino_tensorflow/api.h"
#include "openvino_tensorflow/grappler/ovtf_optimizer.h"
#include "openvino_tensorflow/cluster_manager.h"

#include <iostream>

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

Status NgraphOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  const auto params = config->parameter_map();
  for (auto i : params) {
    m_config_map["_ngraph_" + i.first] = i.second.s();
    NGRAPH_VLOG(3) << "Attribute: " << i.first
                   << " Value: " << m_config_map["_ngraph_" + i.first];
  }
  return Status::OK();
}

Status NgraphOptimizer::Optimize(tensorflow::grappler::Cluster* cluster,
                                 const tensorflow::grappler::GrapplerItem& item,
                                 GraphDef* output) {
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

  // If ngraph is disabled via ngraph_bridge api or NGRAPH_TF_DISABLE is set
  // we will not do anything; all subsequent passes become a no-op.
  bool ngraph_not_enabled =
      (!api::IsEnabled()) || (std::getenv("NGRAPH_TF_DISABLE") != nullptr);
  bool already_processed = util::IsAlreadyProcessed(&graph);
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
  std::set<string> disabled_ops_set = api::GetDisabledOps();
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

  // If requested, dump unmarked graphs.
  util::DumpTFGraph(&graph, idx, "unmarked");

  // 1. Mark for clustering then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(MarkForClustering(&graph, skip_these_nodes));
  util::DumpTFGraph(&graph, idx, "marked");

  // 2. Assign clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(AssignClusters(&graph));
  util::DumpTFGraph(&graph, idx, "clustered");

  // 3. Deassign trivial clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(DeassignClusters(&graph));
  util::DumpTFGraph(&graph, idx, "declustered");

  // 4. Encapsulate clusters then, if requested, dump the graphs.
  auto status = EncapsulateClusters(&graph, idx, m_config_map);
  if (status != Status::OK()) {
    return status;
  }
  util::DumpTFGraph(&graph, idx, "encapsulated");

  // Convert the graph back to Graphdef
  graph.ToGraphDef(output);
  return Status::OK();
}

void NgraphOptimizer::Feedback(tensorflow::grappler::Cluster* cluster,
                               const tensorflow::grappler::GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // no-op
}

int NgraphOptimizer::FreshIndex() {
  mutex_lock l(s_serial_counter_mutex);
  return s_serial_counter++;
}

REGISTER_GRAPH_OPTIMIZER_AS(NgraphOptimizer, "ngraph-optimizer");

}  // end namespace ngraph_bridge
}  // end namespace tensorflow
