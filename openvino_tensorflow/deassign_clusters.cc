/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
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
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#if (TF_MAJOR_VERSION >= 2) && (TF_MINOR_VERSION > 2)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#else
#include "tensorflow/core/graph/graph_constructor.h"
#endif

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

void AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {
  if (src_slot == Graph::kControlSlot) {
    dst->add_input(strings::StrCat("^", src_name));
  } else if (src_slot == 0) {
    dst->add_input(src_name.data(), src_name.size());
  } else {
    dst->add_input(strings::StrCat(src_name, ":", src_slot));
  }
}

Status PopulateClusterGraphDef(
    int target_cluster_idx, GraphDef& gdef, Graph* graph,
    std::map<std::tuple<int, std::string, int>, string>& input_rename_map) {
  // Borrowing this from encapsulate_clusters, as by the time the cluster
  // graph(GraphDef) is populated
  // in encapsulation it's too late.
  for (auto node : graph->op_nodes()) {
    int cluster_idx;

    if (GetNodeAttr(node->attrs(), "_ovtf_cluster", &cluster_idx) !=
        Status::OK()) {
      continue;
    }

    if (cluster_idx != target_cluster_idx) continue;

    // Because the input names may have changed from the original node def,
    // we will need to borrow some code from Graph::ToGraphDefSubRange in
    // tensorflow/core/graph/graph.cc that rewrites the node's input list.

    // begin code copied and pasted (and modified) from graph.cc...
    NodeDef original_def = node->def();

    // Get the inputs for this Node.  We make sure control inputs are
    // after data inputs, as required by GraphDef.
    std::vector<const Edge*> inputs;
    inputs.resize(node->num_inputs(), nullptr);
    for (const Edge* edge : node->in_edges()) {
      if (edge->IsControlEdge()) {
        int src_cluster_idx;
        auto ctrl_src = edge->src();
        auto st = GetNodeCluster(ctrl_src, &src_cluster_idx);
        if (st.ok()) {
          if (src_cluster_idx == cluster_idx) {
            inputs.push_back(edge);
          }
        }
      } else {
        CHECK(inputs[edge->dst_input()] == nullptr)
            << "Edge " << edge->src()->DebugString() << ":"
            << edge->dst()->DebugString() << " with dst_input "
            << edge->dst_input() << " and had pre-existing input edge "
            << inputs[edge->dst_input()]->src()->DebugString() << ":"
            << inputs[edge->dst_input()]->dst()->DebugString();

        inputs[edge->dst_input()] = edge;
      }
    }
    original_def.clear_input();
    original_def.mutable_input()->Reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      const Edge* edge = inputs[i];
      if (edge == nullptr) {
        if (i < node->requested_inputs().size()) {
          original_def.add_input(node->requested_inputs()[i]);
        } else {
          original_def.add_input("");
        }
      } else {
        const Node* src = edge->src();
        if (!src->IsOp()) continue;
        AddInput(&original_def, src->name(), edge->src_output());
      }
    }
    // ...end code copied and pasted (and modified) from graph.cc

    auto node_def = gdef.add_node();
    *node_def = original_def;
    for (auto& input : *(node_def->mutable_input())) {
      TensorId tensor_id = ParseTensorName(input);

      string tensor_name(tensor_id.first);
      auto it = input_rename_map.find(
          std::make_tuple(cluster_idx, tensor_name, tensor_id.second));

      if (it != input_rename_map.end()) {
        input = it->second;
      }
    }
  }
  return Status::OK();
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
  Status exec_status = BackendManager::GetBackendName(device);
  if (exec_status != Status::OK()) {
    throw runtime_error(exec_status.error_message());
  }

  // Variable to store the min threshold for non trivial nodes count in a
  // cluster
  int min_non_trivial_nodes = 6;
  // A map from cluster indices to TF Profiled cumulative costs of nodes in the
  // cluster
  std::vector<std::pair<int, int64_t>> cluster_cost_map_in_ms;

  if (!api::IsRewritePassEnabled()) {
    std::map<int, int> arg_index_count;
    std::map<int, GraphDef> cluster_graph_map;
    std::map<std::tuple<int, std::string, int>, string> input_rename_map;
    // A map from cluster indices to a vector of input data types.
    std::map<int, std::vector<std::tuple<int, int, DataType>>>
        cluster_input_map;

    // edge traversal is required to find out whether src or dst is clustered
    // once the cluster is known, that fetch nodes of the cluster can be
    // identified properly for
    // grappler_item creation
    for (auto edge : graph->edges()) {
      // TODO(amprocte): should actually keep of these. During clustering we
      // will already have identified any intra-cluster control deps. Should
      // maintain inter-cluster control deps.
      if (edge->IsControlEdge()) continue;

      Node* src = edge->src();
      Node* dst = edge->dst();

      // TODO(amprocte): the following rejects edges involving source/sink. Is
      // that what we want to do?
      if (!src->IsOp() || !dst->IsOp()) {
        continue;
      }

      int dst_cluster_idx;
      bool dst_clustered =
          (GetNodeCluster(dst, &dst_cluster_idx) == Status::OK());

      int src_cluster_idx;
      bool src_clustered =
          (GetNodeCluster(src, &src_cluster_idx) == Status::OK());

      // Ignore edges within a cluster. (Note that this test also works when
      // both nodes are unclustered; GetNodeCluster gives us -1 in that case.
      if (dst_cluster_idx == src_cluster_idx) {
        continue;
      }

      DataType dt = dst->input_type(edge->dst_input());
      // If the source node lies within a cluster
      // src is now one of the output nodes of the cluster as it connects to one
      // of the fanout edges
      if (src_clustered) {
        std::stringstream ss_input_to_retval;
        ss_input_to_retval << src->name() << ":" << edge->src_output();
      }

      // If the destination node lies within a cluster
      // src here is one of the input nodes of the cluster
      if (dst_clustered) {
        std::stringstream ss;
        ss << "ngraph_input_" << cluster_input_map[dst_cluster_idx].size();
        std::string new_input_name = ss.str();

        input_rename_map[std::make_tuple(dst_cluster_idx, src->name(),
                                         edge->src_output())] = new_input_name;
        string input_prov_tag = src->name();

        if (cluster_graph_map.find(dst_cluster_idx) == cluster_graph_map.end())
          cluster_graph_map[dst_cluster_idx] =
              *NGraphClusterManager::GetClusterGraph(dst_cluster_idx);

        auto new_input_node_def = cluster_graph_map[dst_cluster_idx].add_node();

        new_input_node_def->set_name(new_input_name);
        new_input_node_def->set_op("_Arg");

        SetAttrValue(dt, &((*(new_input_node_def->mutable_attr()))["T"]));
        SetAttrValue(arg_index_count[dst_cluster_idx],
                     &((*(new_input_node_def->mutable_attr()))["index"]));
        SetAttrValue(input_prov_tag,
                     &((*(new_input_node_def->mutable_attr()))["_prov_tag"]));

        arg_index_count[dst_cluster_idx]++;
        cluster_input_map[dst_cluster_idx].push_back(
            std::make_tuple(src->id(), edge->src_output(), dt));
      }
    }

    for (auto& c : cluster_graph_map) {
      int cluster_idx = c.first;
      GraphDef graph_def = c.second;

      PopulateClusterGraphDef(cluster_idx, graph_def, graph, input_rename_map);

      GraphConstructorOptions opts;
      opts.allow_internal_ops = true;
      opts.expect_device_spec = false;
      FunctionLibraryDefinition flib(OpRegistry::Global(), graph_def.library());
      Graph g(flib);
      TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, graph_def, &g));

      int num_nodes = 0;
      std::pair<int, int64_t> pair_idx_cost;
      pair_idx_cost.first = cluster_idx;
      pair_idx_cost.second = 0;
      for (auto n : g.nodes()) {
        tensorflow::int64 node_cost = 0;
        if (GetNodeAttr(n->attrs(), "cost", &node_cost) != Status::OK())
          continue;
        pair_idx_cost.second += node_cost;
        num_nodes++;
      }
      pair_idx_cost.second /= 1e6;  // ns to ms
      cluster_cost_map_in_ms.push_back(pair_idx_cost);

      OVTF_VLOG(1) << "Cluster with ID " << cluster_idx
                   << ", num_nodes: " << num_nodes << ", costs "
                   << pair_idx_cost.second << " ms";
    }

    // sort clusters based on cost
    std::sort(
        cluster_cost_map_in_ms.begin(), cluster_cost_map_in_ms.end(),
        [](auto& left, auto& right) { return left.second > right.second; });

    // Keep only Top K clusters, based on the value of K set by the following
    // environment variable
    int top_k_clusters = 3;
    if (std::getenv("OPENVINO_TF_MAX_CLUSTERS") != nullptr) {
      top_k_clusters = std::stoi(std::getenv("OPENVINO_TF_MAX_CLUSTERS"));
    }
    // take only top-3 clusters
    cluster_cost_map_in_ms.erase(
        cluster_cost_map_in_ms.begin() + top_k_clusters,
        cluster_cost_map_in_ms.end());

  } else {
    int max_nodes = 0;
    for (auto& kv : cluster_map) {
      int cluster_nodes = kv.second.size();
      if (cluster_nodes > max_nodes) max_nodes = cluster_nodes;
    }
    min_non_trivial_nodes = num_nodes_marked_before_deassign >> 5;
    int avg_nodes_marked_before_deassign =
        num_nodes_marked_before_deassign / cluster_map.size();
    if (avg_nodes_marked_before_deassign < (max_nodes >> 5)) {
      min_non_trivial_nodes <<= 1;
    } else if (min_non_trivial_nodes < avg_nodes_marked_before_deassign * 2) {
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
  }

  // Common section
  std::vector<int> alive_clusters;
  std::vector<std::pair<int, int>> alive_clusters_pairs;
  int max_cluster_size = 0;
  int max_cluster_idx = -1;
  for (auto& kv : cluster_map) {
    int cluster_idx = kv.first;
    std::set<Node*>& nodes = kv.second;

    if (!api::IsRewritePassEnabled()) {
      // Cluster selection is based on compute cost
      if (std::find_if(
              cluster_cost_map_in_ms.begin(), cluster_cost_map_in_ms.end(),
              [&cluster_idx](const std::pair<int, int64_t>& pair_idx_cost) {
                return pair_idx_cost.first == cluster_idx;
              }) != cluster_cost_map_in_ms.end()) {
        OVTF_VLOG(1) << "Scheduling cluster " << cluster_idx
                     << " on the OpenVINO backend";
      } else {
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
    } else {  // Cluster Selection based on Min Number of Non Trivial Nodes
      int non_trivial_count = 0;
      std::unordered_set<std::string> trivial_ops = {"Const", "Identitiy"};
      bool illegal_pattern = false;
      for (auto node : nodes) {
        if (trivial_ops.find(node->type_string()) == trivial_ops.end()) {
          non_trivial_count++;
        }

        // Check for illegal patterns to deassign.
        // TODO: This should be removed once these
        // patterns are passing OV optimizations.
        if (node->type_string() == "Select") {
            Node *in_node_first = nullptr;
            node->input_node(0, &in_node_first);
            if (in_node_first && (in_node_first->type_string() == "Less" || in_node_first->type_string() == "Greater")) {
                Node *in_node_second = nullptr;
                in_node_first->input_node(0, &in_node_second);
                if (in_node_second && in_node_second->type_string() == "Const") {
                    illegal_pattern = true;
                }
            }
        }

        if (node->type_string() == "Pack") {
            Node *in_node_first = nullptr;
            node->input_node(1, &in_node_first);
            if (in_node_first && in_node_first->type_string() == "Reshape") {
                Node *in_node_second = nullptr;
                in_node_first->input_node(0, &in_node_second);
                if (in_node_second && in_node_second->type_string() == "Tile") {
                    Node *in_node_third = nullptr;
                    in_node_second->input_node(1, &in_node_third);
                    if (in_node_third && in_node_third->type_string() == "Const") {
                        illegal_pattern = true;
                    }
                }
            }
        }
      }

      if (non_trivial_count < min_non_trivial_nodes || illegal_pattern) {
        OVTF_VLOG(2) << "Busting cluster " << cluster_idx
                     << " as number of non trivial nodes in it are less than "
                        "MIN_NONTRIVIAL_NODES threshold";
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
    }

    // Enable dynamic to static check only for GPU
    if (device == "GPU") {
      OVTF_VLOG(3) << "Checking Dynamic to Static flow for GPU";

      std::vector<Node*> dyn_node_check;
      std::set<Node*> visited_node_check;
      for (auto node : nodes) {
        if (node->type_string() == "NonMaxSuppressionV2") {
          dyn_node_check.push_back(node);
          visited_node_check.insert(node);
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
                (it->type_string() != "NonMaxSuppressionV2")) {
              if (it->type_string() == "ZerosLike") {
                invalid_dyn_op = true;
                break;
              } else if (visited_node_check.find(it) ==
                         visited_node_check.end()) {
                dyn_node_check.push_back(it);
                visited_node_check.insert(it);
              }
            }
          }
        }
      }
      if (invalid_dyn_op) {
        OVTF_VLOG(2) << "Busting cluster " << cluster_idx
                     << " due to Dynamic to Static Flow";
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
    alive_clusters_pairs.push_back(std::make_pair(cluster_idx, nodes.size()));
  }

  // For HDDL and Myriad, schedule only top 1 cluster
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

  if (api::IsRewritePassEnabled()) {
    // Keep only Top K clusters, based on the value of K set by the following
    // environment variable
    int top_k_clusters = -1;
    if (std::getenv("OPENVINO_TF_MAX_CLUSTERS") != nullptr) {
      top_k_clusters = std::stoi(std::getenv("OPENVINO_TF_MAX_CLUSTERS"));
    }
    if (top_k_clusters != -1) {
      // sort the clusters
      std::sort(
          alive_clusters_pairs.begin(), alive_clusters_pairs.end(),
          [](const auto& x, const auto& y) { return x.second > y.second; });
      int total_clusters = alive_clusters_pairs.size();
      for (int count = total_clusters - 1; count >= top_k_clusters; count--) {
        // clear the cluster
        auto cluster_id_to_clear = alive_clusters_pairs[count].first;
        set<Node*>& nodes = cluster_map[cluster_id_to_clear];

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
