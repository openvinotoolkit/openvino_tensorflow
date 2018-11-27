/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/device_name_utils.h"

#include "ngraph_api.h"
#include "ngraph_assign_clusters.h"
#include "ngraph_deassign_clusters.h"
#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// The clustering pass of ngraph_assign_clusters.cc sometimes generates many
// small, trivial clusters. In this pass, we simply deassign (i.e., remove the
// _ngraph_cluster and _ngraph_marked_for_clustering attributes) any such
// trivial clusters. For now, "trivial" just means that there are not at least
// two non-trivial ops in the graph, where a "trivial op" means "Const" or
// "Identity".
//
// For unit testing purposes, this pass can be bypassed by setting
// NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS=1.
//

static const int MIN_NONTRIVIAL_NODES = 2;

static void MaybeLogPlacement(const Graph* graph) {
  if (!config::IsLoggingPlacement()) return;

  std::map<int, std::set<const Node*>> final_cluster_map;
  int number_of_nodes = 0, nodes_marked_for_clustering = 0,
      nodes_assigned_a_cluster = 0;
  for (auto node : graph->nodes()) {
    number_of_nodes++;
    // Check marked for clustering
    if (NodeIsMarkedForClustering(node)) {
      nodes_marked_for_clustering++;
    }

    // Check Cluster Assignment
    int cluster_idx;
    if (!GetNodeCluster(node, &cluster_idx).ok()) {
      cluster_idx = -1;
    } else {
      nodes_assigned_a_cluster++;
    }
    final_cluster_map[cluster_idx].insert(node);
  }
  if (number_of_nodes == 0) return;

  int perc_marked_for_clustering_of_total =
      (int)((nodes_marked_for_clustering * 100.0) / number_of_nodes);
  int perc_assigned_clusters_of_total =
      (int)((nodes_assigned_a_cluster * 100.0) / number_of_nodes);
  int perc_assigned_clusters_of_marked =
      nodes_marked_for_clustering > 0
          ? (int)((nodes_assigned_a_cluster * 100.0) /
                  nodes_marked_for_clustering)
          : 0;

  std::cout << "Number of nodes in the graph: " << number_of_nodes << std::endl;
  std::cout << "Number of nodes marked for clustering: "
            << nodes_marked_for_clustering << " ("
            << perc_marked_for_clustering_of_total << "% of total nodes)"
            << std::endl;
  std::cout << "Number of nodes assigned a cluster: "
            << nodes_assigned_a_cluster << " ("
            << perc_assigned_clusters_of_total << "% of total nodes) \t"
            << " (" << perc_assigned_clusters_of_marked
            << "% of nodes marked for clustering) \t" << std::endl;
  std::cout << "Number of ngraph clusters :" << final_cluster_map.size() - 1
            << std::endl;

  for (auto kv : final_cluster_map) {
    int cluster_idx = kv.first;
    if (cluster_idx != -1) {
      std::cout << "Size of nGraph Cluster[" << cluster_idx << "]\t"
                << kv.second.size() << std::endl;
    }
  }

  for (auto kv : final_cluster_map) {
    int cluster_idx = kv.first;
    std::set<const Node*>& nodes = kv.second;
    for (auto node : nodes) {
      std::stringstream placement_dev;
      placement_dev << "OP_placement:\t";
      if (cluster_idx == -1) {
        placement_dev << "Host\t";
      } else {
        placement_dev << "nGraph[" << cluster_idx << "]\t";
      }
      placement_dev << node->name() << " (" << node->type_string() << ")";
      std::cout << placement_dev.str() << std::endl;
    }
  }
}

Status DeassignClusters(Graph* graph) {
  //
  // When running unit tests, we do not want to see trivial clusters
  // deassigned. This flag (used by the Python tests) makes this possible.
  //
  if (std::getenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS") != nullptr) {
    MaybeLogPlacement(graph);
    return Status::OK();
  }

  std::map<int, std::set<Node*>> cluster_map;

  for (auto node : graph->nodes()) {
    int cluster_idx;

    if (GetNodeCluster(node, &cluster_idx) != Status::OK()) {
      continue;
    }

    cluster_map[cluster_idx].insert(node);
  }

  for (auto& kv : cluster_map) {
    int cluster_idx = kv.first;
    std::set<Node*>& nodes = kv.second;

    int non_trivial_count = 0;

    for (auto node : nodes) {
      // TODO(amprocte): less hard-coding here
      if (node->type_string() != "Const" && node->type_string() != "Identity") {
        non_trivial_count++;
      }
    }

    if (non_trivial_count < MIN_NONTRIVIAL_NODES) {
      NGRAPH_VLOG(2) << "Busting cluster " << cluster_idx;
      for (auto node : nodes) {
        NGRAPH_VLOG(2) << "Busting node: " << node->name() << " ["
                       << node->type_string() << "]";

        // TODO(amprocte): move attr name to a constant
        node->ClearAttr("_ngraph_cluster");
        // TODO(amprocte): move attr name to a constant
        node->ClearAttr("_ngraph_marked_for_clustering");
      }
    }
  }

  //
  // At this point we have made our final decision about cluster assignment, so
  // we will log the cluster assignment now.
  //
  MaybeLogPlacement(graph);

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
