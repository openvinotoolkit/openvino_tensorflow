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
#include "openvino_tensorflow/backend_manager.h"
#include "openvino_tensorflow/deassign_clusters.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_utils.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

//
// The clustering pass of assign_clusters.cc sometimes generates many
// small, trivial clusters. In this pass, we simply deassign (i.e., remove the
// _ovtf_cluster and _ovtf_marked_for_clustering attributes) any such
// trivial clusters. For now, "trivial" just means that there are not at least
// two non-trivial ops in the graph, where a "trivial op" means "Const" or
// "Identity".
//
// For unit testing purposes, this pass can be bypassed by setting
// OPENVINO_TF_DISABLE_DEASSIGN_CLUSTERS=1.
//

// For sorting the clusters for MYRIAD
static bool cmp(pair<int, std::set<Node*>>& a, pair<int, std::set<Node*>>& b) {
  return a.second.size() > b.second.size();
}

unordered_map<string, int> deassigned_histogram;
int num_nodes_marked_before_deassign = 0;

static void MaybeLogPlacement(const Graph* graph) {
  std::map<int, std::set<const Node*>> final_cluster_map;
  int number_of_nodes = 0, nodes_marked_for_clustering = 0,
      nodes_assigned_a_cluster = 0, functional_nodes = 0;
  for (auto node : graph->nodes()) {
    number_of_nodes++;
    // Check marked for clustering
    if (NodeIsMarkedForClustering(node)) {
      nodes_marked_for_clustering++;
      functional_nodes++;
    } else if (node->type_string() != "NoOp" && node->type_string() != "_Arg" &&
               node->type_string() != "_Retval") {
      functional_nodes++;
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

  std::cout << NGraphLogMessage::GetTimeStampForLogging()
            << ": OVTF Summary -> " << nodes_assigned_a_cluster << " out of "
            << number_of_nodes << " nodes in the graph ("
            << perc_assigned_clusters_of_total
            << "%) are now running with OpenVINO™ backend" << std::endl;

  if (api::IsLoggingPlacement()) {
    std::cout << "\n";  // insert a new line at the start of OVTF_SUMMARY
    std::cout << "OVTF_SUMMARY: Number of nodes in the graph: "
              << number_of_nodes << std::endl;
    // print out the number of nodes marked before deassign
    std::cout << "OVTF_SUMMARY: Number of nodes marked for clustering: "
              << num_nodes_marked_before_deassign << " ("
              << perc_marked_for_clustering_of_total << "% of total nodes)"
              << std::endl;
    // print out the number of nodes that are running on NGraph after deassign
    std::cout << "OVTF_SUMMARY: Number of nodes assigned a cluster: "
              << nodes_assigned_a_cluster << " ("
              << perc_assigned_clusters_of_total << "% of total nodes) \t"
              << " (" << perc_assigned_clusters_of_marked
              << "% of nodes marked for clustering) \t" << std::endl;
    int num_encapsulates = final_cluster_map.size() - 1;
    std::cout << "OVTF_SUMMARY: Number of ngraph clusters :" << num_encapsulates
              << std::endl;
    std::cout << "OVTF_SUMMARY: Average Nodes per cluster: "
              << ((num_encapsulates > 0) ? (float(nodes_assigned_a_cluster) /
                                            float(num_encapsulates))
                                         : 0)
              << endl;
  }

  for (auto kv : final_cluster_map) {
    int cluster_idx = kv.first;
    if (cluster_idx != -1) {
      int perc_nodes_assigned =
          functional_nodes > 0
              ? (int)((kv.second.size() * 100.0) / functional_nodes)
              : 0;
      std::string cluster_info = "ovtf_cluster_" + std::to_string(cluster_idx) +
                                 ": " + std::to_string(perc_nodes_assigned) +
                                 "%";
      NGraphClusterManager::SetClusterInfo(cluster_idx, cluster_info);
      if (api::IsLoggingPlacement()) {
        std::cout << "OVTF_SUMMARY: Size of nGraph Cluster[" << cluster_idx
                  << "]:\t" << kv.second.size() << std::endl;
      }
    }
  }

  if (!api::IsLoggingPlacement()) return;

  // log the ops gets deassigned
  std::cout << "OVTF_SUMMARY: Op_deassigned: ";
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

  string device;
  BackendManager::GetBackendName(device);

  std::vector<int> alive_clusters;
  int max_cluster_size = 0;
  int max_cluster_idx = -1;

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

    int min_non_trivial_nodes = num_nodes_marked_before_deassign >> 5;
    int avg_nodes_marked_before_deassign =
        num_nodes_marked_before_deassign / cluster_map.size();
    if (min_non_trivial_nodes < avg_nodes_marked_before_deassign * 2) {
      min_non_trivial_nodes >>= 2;
    }
    if (min_non_trivial_nodes < 6) {
      min_non_trivial_nodes = 6;
    }
    if (std::getenv("OPENVINO_TF_MIN_NONTRIVIAL_NODES") != nullptr) {
      min_non_trivial_nodes =
          std::stoi(std::getenv("OPENVINO_TF_MIN_NONTRIVIAL_NODES"));
    }
    OVTF_VLOG(1) << "MIN_NONTRIVIAL_NODES set to " << min_non_trivial_nodes;

    if (non_trivial_count < min_non_trivial_nodes) {
      OVTF_VLOG(2) << "Busting cluster " << cluster_idx;
      for (auto node : nodes) {
        OVTF_VLOG(2) << "Busting node: " << node->name() << " ["
                     << node->type_string() << "]";

        // TODO(amprocte): move attr name to a constant
        node->ClearAttr("_ovtf_cluster");
        // TODO(amprocte): move attr name to a constant
        node->ClearAttr("_ovtf_marked_for_clustering");

        deassigned_histogram[node->type_string()]++;
      }
      continue;
    }
    // Disable dynamic to static
    std::vector<Node*> dyn_node_check;
    for (auto node : nodes) {
      if (node->type_string() == "NonMaxSuppressionV2" ||
          node->type_string() == "Reshape") {
        dyn_node_check.push_back(node);
      }
    }
    bool invalid_dyn_op = false;
    while (dyn_node_check.size() > 0) {
      Node* node = dyn_node_check.back();
      dyn_node_check.pop_back();

      for (auto it : node->out_nodes()) {
        int out_cluster;
        Status s = GetNodeAttr(it->attrs(), "_ovtf_cluster", &out_cluster);
        if (s == Status::OK()) {
          if (out_cluster == cluster_idx &&
              (it->type_string() != "NonMaxSuppressionV2" &&
               it->type_string() != "Reshape")) {
            if (it->type_string() == "ZerosLike" ||
                it->type_string() == "Size" || it->type_string() == "Conv2D" ||
                it->type_string() == "Unpack") {
              invalid_dyn_op = true;
              break;
            } else {
              dyn_node_check.push_back(it);
            }
          }
        }
      }
    }
    if (invalid_dyn_op) {
      OVTF_VLOG(2) << "Busting cluster " << cluster_idx;
      for (auto node : nodes) {
        OVTF_VLOG(2) << "Busting node: " << node->name() << " ["
                     << node->type_string() << "]";

        // TODO(amprocte): move attr name to a constant
        node->ClearAttr("_ovtf_cluster");
        // TODO(amprocte): move attr name to a constant
        node->ClearAttr("_ovtf_marked_for_clustering");

        deassigned_histogram[node->type_string()]++;
      }
      continue;
    }

    unordered_set<std::string> input_args;
    vector<string> cluster_inputs;
    bool omit_cluster = false;

    for (auto node : nodes) {
      for (auto it : node->in_nodes()) {
        if (!input_args.count(it->name())) {
          cluster_inputs.push_back(it->name());
        }
        input_args.insert(it->name());
      }
    }
    for (auto node : nodes) {
      if (node->type_string() == "Prod") {
        for (auto it : node->in_nodes()) {
          auto inp_name = it->name();
          auto iter =
              find(cluster_inputs.begin(), cluster_inputs.end(), inp_name);
          if (iter != cluster_inputs.end()) {
            omit_cluster = true;
            break;
          }
        }
      }
      if (omit_cluster) break;
    }
    if (omit_cluster) {
      for (auto node : nodes) {
        node->ClearAttr("_ovtf_cluster");
        node->ClearAttr("_ovtf_marked_for_clustering");
        deassigned_histogram[node->type_string()]++;
      }
      continue;
    }

    if (device == "HDDL") {
      std::vector<std::string> illegal_input_nodes = {"Unpack"};
      std::vector<std::string> illegal_output_nodes = {"Greater"};
      bool omit_cluster = false;
      for (auto node : nodes) {
        if (std::find(illegal_output_nodes.begin(), illegal_output_nodes.end(),
                      node->type_string()) != illegal_output_nodes.end()) {
          for (auto it : node->out_nodes()) {
            int out_cluster;
            Status s = GetNodeAttr(it->attrs(), "_ovtf_cluster", &out_cluster);
            if ((s == Status::OK() && out_cluster != cluster_idx) ||
                (s != Status::OK())) {
              omit_cluster = true;
              break;
            }
          }
        }
        if (std::find(illegal_input_nodes.begin(), illegal_input_nodes.end(),
                      node->type_string()) != illegal_input_nodes.end()) {
          for (auto it : node->in_nodes()) {
            int in_cluster;
            Status s = GetNodeAttr(it->attrs(), "_ovtf_cluster", &in_cluster);
            if ((s == Status::OK() && in_cluster != cluster_idx) ||
                s != Status::OK()) {
              omit_cluster = true;
              break;
            }
          }
        }
      }
      if (omit_cluster) {
        for (auto node : nodes) {
          node->ClearAttr("_ovtf_cluster");
          node->ClearAttr("_ovtf_marked_for_clustering");
          deassigned_histogram[node->type_string()]++;
        }
        continue;
      }
    }

    if (max_cluster_idx == -1) {
      max_cluster_idx = cluster_idx;
      max_cluster_size = nodes.size();
    } else if (max_cluster_size < nodes.size()) {
      max_cluster_idx = cluster_idx;
      max_cluster_size = nodes.size();
    }
    alive_clusters.push_back(cluster_idx);
  }

  if (device == "HDDL" || device == "MYRIAD") {
    for (int i = 0; i < alive_clusters.size(); i++) {
      int alive_cluster_idx = alive_clusters[i];
      if (alive_cluster_idx != max_cluster_idx) {
        set<Node*>& nodes = cluster_map[alive_cluster_idx];

        for (auto node : nodes) {
          node->ClearAttr("_ovtf_cluster");
          node->ClearAttr("_ovtf_marked_for_clustering");
          deassigned_histogram[node->type_string()]++;
        }
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
