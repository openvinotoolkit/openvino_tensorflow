/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/device_name_utils.h"

#include "api.h"
#include "logging/ovtf_log.h"
#include "openvino_tensorflow/assign_clusters.h"
#include "openvino_tensorflow/deassign_clusters.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_utils.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

//
// The clustering pass of ngraph_assign_clusters.cc sometimes generates many
// small, trivial clusters. In this pass, we simply deassign (i.e., remove the
// _ngraph_cluster and _ngraph_marked_for_clustering attributes) any such
// trivial clusters. For now, "trivial" just means that there are not at least
// two non-trivial ops in the graph, where a "trivial op" means "Const" or
// "Identity".
//
// For unit testing purposes, this pass can be bypassed by setting
// OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS=1.
//

unordered_map<string, int> deassigned_histogram;
int num_nodes_marked_before_deassign = 0;

static void MaybeLogPlacement(const Graph* graph) {
  if (!api::IsLoggingPlacement()) return;

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
      (int)((num_nodes_marked_before_deassign * 100.0) / number_of_nodes);
  int perc_assigned_clusters_of_total =
      (int)((nodes_assigned_a_cluster * 100.0) / number_of_nodes);
  int perc_assigned_clusters_of_marked =
      num_nodes_marked_before_deassign > 0
          ? (int)((nodes_assigned_a_cluster * 100.0) /
                  num_nodes_marked_before_deassign)
          : 0;

  std::cout << "\n";  // insert a new line at the start of NGTF_SUMMARY
  std::cout << "NGTF_SUMMARY: Number of nodes in the graph: " << number_of_nodes
            << std::endl;
  // print out the number of nodes marked before deassign
  std::cout << "NGTF_SUMMARY: Number of nodes marked for clustering: "
            << num_nodes_marked_before_deassign << " ("
            << perc_marked_for_clustering_of_total << "% of total nodes)"
            << std::endl;
  // print out the number of nodes that are running on NGraph after deassign
  std::cout << "NGTF_SUMMARY: Number of nodes assigned a cluster: "
            << nodes_assigned_a_cluster << " ("
            << perc_assigned_clusters_of_total << "% of total nodes) \t"
            << " (" << perc_assigned_clusters_of_marked
            << "% of nodes marked for clustering) \t" << std::endl;
  int num_encapsulates = final_cluster_map.size() - 1;
  std::cout << "NGTF_SUMMARY: Number of ngraph clusters :" << num_encapsulates
            << std::endl;
  std::cout << "NGTF_SUMMARY: Nodes per cluster: "
            << ((num_encapsulates > 0) ? (float(nodes_assigned_a_cluster) /
                                          float(num_encapsulates))
                                       : 0)
            << endl;

  for (auto kv : final_cluster_map) {
    int cluster_idx = kv.first;
    if (cluster_idx != -1) {
      std::cout << "NGTF_SUMMARY: Size of nGraph Cluster[" << cluster_idx
                << "]:\t" << kv.second.size() << std::endl;
    }
  }

  // log the ops gets deassigned
  std::cout << "NGTF_SUMMARY: Op_deassigned: ";
  util::PrintNodeHistogram(deassigned_histogram);

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
  std::cout << endl;
}

Status DeassignClusters(Graph* graph) {
  //
  // When running unit tests, we do not want to see trivial clusters
  // deassigned. This flag (used by the Python tests) makes this possible.
  //
  num_nodes_marked_before_deassign = 0;  // reset for every TF graph
  deassigned_histogram.clear();          // reset the histogram

  if (std::getenv("OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS") != nullptr) {
    // still need to calculate num_nodes_marked_before_deassign
    for (auto node : graph->nodes()) {
      int cluster_idx;

      if (GetNodeCluster(node, &cluster_idx) == Status::OK()) {
        num_nodes_marked_before_deassign++;
      }
    }
    MaybeLogPlacement(graph);
    return Status::OK();
  }

  std::map<int, std::set<Node*>> cluster_map;

  for (auto node : graph->nodes()) {
    int cluster_idx;
    if (GetNodeCluster(node, &cluster_idx) != Status::OK()) {
      continue;
    }

    num_nodes_marked_before_deassign++;
    cluster_map[cluster_idx].insert(node);
  }

  for (auto& kv : cluster_map) {
    int cluster_idx = kv.first;
    std::set<Node*>& nodes = kv.second;

    int non_trivial_count = 0;

    std::unordered_set<std::string> trivial_ops = {"Const", "Identitiy"};
    for (auto node : nodes) {
      if (trivial_ops.find(node->type_string()) == trivial_ops.end()) {
        non_trivial_count++;
      }
    }

    int min_non_trivial_nodes = 6;
    if (std::getenv("OPENVINO_TF_MIN_NONTRIVIAL_NODES") != nullptr) {
      min_non_trivial_nodes =
          std::stoi(std::getenv("OPENVINO_TF_MIN_NONTRIVIAL_NODES"));
    }
    NGRAPH_VLOG(1) << "MIN_NONTRIVIAL_NODES set to " << min_non_trivial_nodes;

    if (non_trivial_count < min_non_trivial_nodes) {
      NGRAPH_VLOG(2) << "Busting cluster " << cluster_idx;
      for (auto node : nodes) {
        NGRAPH_VLOG(2) << "Busting node: " << node->name() << " ["
                       << node->type_string() << "]";

        // TODO(amprocte): move attr name to a constant
        node->ClearAttr("_ngraph_cluster");
        // TODO(amprocte): move attr name to a constant
        node->ClearAttr("_ngraph_marked_for_clustering");

        deassigned_histogram[node->type_string()]++;
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

}  // namespace openvino_tensorflow
}  // namespace tensorflow
