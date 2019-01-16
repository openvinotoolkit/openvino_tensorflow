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
#include "ngraph_cluster_manager.h"
#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_utils.h"
#include "tf_deadness_analysis.h"
#include "tf_graphcycles.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// The clustering pass performs a greedy search for groups of nGraph-marked ops
// that can be coalesced into a single nGraph graph, and assigns each such
// group a unique identifier called a "cluster ID".
//
// For example, consider the following graph:
//
//       N1
//      /  \
//    N2    N5
//    / \    |
//   N3  N4 N6
//           |
//          N7
//
// If nodes N1, N2, N3, N4, N6, and N7 are all marked, but N5 is unmarked, the
// clustering pass will assign nodes N1, N2, N3, and N4 to one cluster, and
// nodes N6 and N7 to another.
//
// After clustering, it must be the case that:
//
//   (1) every marked node is assigned to exactly one cluster;
//   (2) no unmarked node is assigned to any cluster;
//   (3) for every pair (N1,N2) of nodes where N1 and N2 are in the same
//       cluster, there is no path from N1 to N2 traversing any node N3 that
//       is _not_ in the same cluster as N1 and N2 (in other words,
//       data/control flow cannot "re-enter" the cluster).
//
// Other Constraints (Non Data Flow Constraints)
//
//   (1) If N1 is a static input to N2, N1 and N2 are not placed in the same
//       cluster (More on static inputs in ngraph_mark_for_clustering)
//   (2) If N1 and N2 have mismatching deadness predicates, they are not
//       placed in the same cluster (More on deadness in tf_deadness_analysis)
//
// Given the above constraints, we try to find the "biggest" clusters we can.
//
// The assigned cluster index is represented by the "_ngraph_cluster"
// attribute, which has integer type.
//
// Assumption: the "MarkForClustering" pass (ngraph_mark_for_clustering.cc) has
// already been run. This attaches the "_ngraph_marked_for_clustering"
// attribute to ops which we will cluster.
//
// TODO(amprocte): Say more about the algorithm.
//

namespace {
struct Cluster {
  int index;
  std::set<tensorflow::Node*> nodes;
  std::string backend;
#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
  std::string predicate_string;
  std::set<const Edge*> outgoing_edges;
#endif
};

Status InitialiseNodeBackend(Node* node, string* backend) {
  NGRAPH_VLOG(5) << "Initialize Node Backend " << node->name();
  if (!HasNodeAttr(node->def(), "_ngraph_backend")) {
    *backend = "HOST";
    return Status::OK();
  }
  NGRAPH_VLOG(5) << "Should have been assigned Node Backend " << node->name();
  TF_RETURN_IF_ERROR(GetNodeBackend(node, backend));
  // TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "_ngraph_backend", backend));
  return Status::OK();
}

Status CanContractEdgeBackendCheck(
    Edge* edge, const std::map<Node*, std::shared_ptr<Cluster>>& cluster_map,
    bool& is_backend_ok) {
  Node* src = edge->src();
  Node* dst = edge->dst();

  string src_backend = cluster_map.at(src)->backend;
  string dst_backend = cluster_map.at(dst)->backend;

  if (src_backend == dst_backend) {
    is_backend_ok = true;
  } else {
    is_backend_ok = false;
  }
  return Status::OK();
}

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
// Returns the predicate of the merged cluster
// If Src Predicate is TRUE then merged cluster gets the dst predicate
// WARNING : This function does not do any checks
// Use this function when ready to merge
inline string GetMergedClusterPred(const string& src_predicate,
                                   const string& dst_predicate) {
  return DeadnessAnalysis::IsTruePredString(src_predicate) ? dst_predicate
                                                           : src_predicate;
}

// Checks whether it's ok to contract the edge as far as deadness is concerned
// Source and Dst Predicates of the edge should match
Status CanContractEdgeDeadnessCheck(
    Edge* edge, const std::map<Node*, std::shared_ptr<Cluster>>& cluster_map,
    bool& is_deadness_ok) {
  Node* src = edge->src();
  Node* dst = edge->dst();

  string src_predicate = cluster_map.at(src)->predicate_string;
  string dst_predicate = cluster_map.at(dst)->predicate_string;

  // If the node marked for clustering has CONTROL_FLOW_PRED_STRING, it
  // breaks our assumption that all supported ops are data flow ops
  if (DeadnessAnalysis::IsControlFlowPredString(src_predicate) ||
      DeadnessAnalysis::IsControlFlowPredString(dst_predicate)) {
    return errors::Internal(
        "Attempting to contract edge with control flow ops : ",
        edge->DebugString());
  }

  // Case src X , dst Y , X!=Y // cannot be contracted
  if (!DeadnessAnalysis::IsTruePredString(src_predicate) &&
      !DeadnessAnalysis::IsTruePredString(dst_predicate) &&
      src_predicate != dst_predicate) {
    is_deadness_ok = false;
    return Status::OK();
  }

  // Case src X , dst True // invalid scenario
  // If src has Non-True Predicate and dst has True Predicate, it implies that
  // the dst node is control flow
  if (!DeadnessAnalysis::IsTruePredString(src_predicate) &&
      DeadnessAnalysis::IsTruePredString(dst_predicate)) {
    return errors::Internal("Attempting to cluster control-flow node ",
                            dst->name(), "[", dst->type_string(), "]");
  }

  // Case src True, dst Y
  // If this edge is contracted, all the outputs of the merged cluster will
  // have the predicate Y (True & Y = Y). Hence contraction is possible only
  // when, all outputs of the src cluster (other than the current edge) have the
  // predicate Y
  if (DeadnessAnalysis::IsTruePredString(src_predicate)) {
    auto src_cluster_out_edges = cluster_map.at(src)->outgoing_edges;
    bool found_same_out_preds = true;
    std::string pred_check = dst_predicate;

    for (const Edge* src_cluster_edge : src_cluster_out_edges) {
      if (src_cluster_edge == edge) {
        continue;
      }
      Node* src_cluster_dst = src_cluster_edge->dst();
      std::string src_cluster_dst_pred =
          cluster_map.at(src_cluster_dst)->predicate_string;
      // Note that if dst predicate is True, then it does not matter what the
      // src_cluster_dst_pred is; After merge the merged cluster will always
      // have a less strict predicate, True (since True is the least strict
      // predicate)
      if (!DeadnessAnalysis::IsTruePredString(pred_check) &&
          pred_check != src_cluster_dst_pred) {
        found_same_out_preds = false;
        break;
      }
    }
    // Cannot contract this edge
    if (!found_same_out_preds) {
      is_deadness_ok = false;
      return Status::OK();
    }
  }

  // Case src X, dst Y, X==Y
  // Case src True, dst Y(Y can be True), all other output edges from src
  // cluster have pred Y
  // Ok to contract
  is_deadness_ok = true;
  return Status::OK();
}

// Some sanity checks for Node's cluster assignment wrt Deadness
Status CheckNodeClusterAssignmentWRTDeadness(
    Node* node, const std::map<Node*, string>& nodes_predicate_map,
    const std::map<Node*, std::shared_ptr<Cluster>>& cluster_map) {
  auto itr = nodes_predicate_map.find(node);
  if (itr == nodes_predicate_map.end()) {
    return errors::Internal("Node ", node->name(), " [", node->type_string(),
                            "]", " not found in predicate map");
  }
  std::string node_pred_string = itr->second;

  if (DeadnessAnalysis::IsControlFlowPredString(node_pred_string)) {
    return errors::Internal(
        "Node ", node->name(), " [", node->type_string(), "]",
        " should not be clustered as it is a control flow op");
  }

  std::string cluster_pred_string = cluster_map.at(node)->predicate_string;
  int node_cluster_index = cluster_map.at(node)->index;

  // If the node has Non-True Pred (P1) it can only be placed in a cluster with
  // the same pred
  if (!DeadnessAnalysis::IsTruePredString(node_pred_string) &&
      node_pred_string != cluster_pred_string) {
    return errors::Internal(
        "Node ", node->name(), " [", node->type_string(), "]", " Predicate : ",
        node_pred_string, "should not be clustered in cluster with predicate ",
        cluster_pred_string);
  }

  // If the node has True Pred (T1) and its cluster pred is non-true (P1)
  // Then all outgoing edges from node which are not in the same cluster should
  // be connected to clusters with pred P1
  if (DeadnessAnalysis::IsTruePredString(node_pred_string) &&
      !DeadnessAnalysis::IsTruePredString(cluster_pred_string)) {
    for (auto e : node->out_edges()) {
      Node* e_dst = e->dst();
      if (cluster_map.at(e_dst)->index != node_cluster_index) {
        string e_dst_cluster_pred = cluster_map.at(e_dst)->predicate_string;
        if (e_dst_cluster_pred != cluster_pred_string) {
          return errors::Internal(
              "Node ", node->name(), " [", node->type_string(), "]",
              " Predicate : ", node_pred_string,
              " cannot not be clustered in cluster with predicate ",
              cluster_pred_string,
              " as it has outgoing edge to a cluster with predicate ",
              e_dst_cluster_pred);
        }
      }
    }
  }

  return Status::OK();
}
#endif

// Merges src and dst clusters of the edge
// This function does not do any checks for merging, but rather implements the
// merge, i.e. updates the properties of the merged cluster
// WARNING : Use this function when ready to merge
void MergeClusters(Edge* edge,
                   std::map<Node*, std::shared_ptr<Cluster>>& cluster_map) {
  Node* src = edge->src();
  Node* dst = edge->dst();
  int src_index = cluster_map[src]->index;
  int dst_index = cluster_map[dst]->index;

  // Merge dst cluster into src cluster
  NGRAPH_VLOG(5) << "Contracting: " << src->name() << "[" << src->type_string()
                 << " , " << edge->src_output() << "]@" << src_index << " -> "
                 << dst->name() << "[" << dst->type_string() << " , "
                 << edge->dst_input() << "]@" << dst_index;

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
  string src_predicate = cluster_map[src]->predicate_string;
  string dst_predicate = cluster_map[dst]->predicate_string;
  NGRAPH_VLOG(5) << "Src pred: " << src_predicate
                 << ", Dst pred: " << dst_predicate;

  std::string cluster_pred = GetMergedClusterPred(src_predicate, dst_predicate);

  cluster_map[src]->predicate_string = cluster_pred;
  // Update outgoing edges of the merged cluster
  cluster_map[src]->outgoing_edges.insert(
      cluster_map[dst]->outgoing_edges.begin(),
      cluster_map[dst]->outgoing_edges.end());
  cluster_map[src]->outgoing_edges.erase(edge);
#endif

  auto cluster_dst = cluster_map[dst];
  // using cluster_map[dst]->nodes in the loop directly appears to
  // invalidate the iterator when `node` == `dst`
  // this happens with clang but not gcc
  for (auto node : cluster_dst->nodes) {
    cluster_map[src]->nodes.insert(node);
    cluster_map[node] = cluster_map[src];
  }
}

}  // namespace

// Main Entry point for Cluster Assignment to the Node
// Adds an attribute "_ngraph_cluster" (cluster_id) to each Node that can be
// encapsulated
Status AssignClusters(Graph* graph) {
  std::map<Node*, std::shared_ptr<Cluster>> cluster_map;

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
  std::unique_ptr<DeadnessAnalysis> deadness_analyzer;
  TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(*graph, &deadness_analyzer));
  // This map is used only for error checking
  std::map<Node*, std::string> nodes_predicate_map;
#endif

  GraphCycles gc;

  // Initial Step: Each node is a cluster of its own
  for (auto node : graph->nodes()) {
    int new_index = gc.NewNode();
    cluster_map[node] = std::make_shared<Cluster>();
    cluster_map[node]->index = new_index;
    string backend;
    TF_RETURN_IF_ERROR(InitialiseNodeBackend(node, &backend));

    cluster_map[node]->backend = backend;
    cluster_map[node]->nodes.insert(node);
    NGRAPH_VLOG(5) << "Creating graphcycle Node: " << new_index << " for "
                   << node->name() << "[" << node->type_string() << "]"
                   << " backend " << backend;

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
    // get predicate string for the node
    string pred_string;
    TF_RETURN_IF_ERROR(deadness_analyzer->GetNodePredicate(*node, pred_string));
    nodes_predicate_map[node] = pred_string;
    cluster_map[node]->predicate_string = pred_string;

    cluster_map[node]->outgoing_edges = std::set<const Edge*>(
        node->out_edges().begin(), node->out_edges().end());
    NGRAPH_VLOG(5) << node->name() << "[" << node->type_string() << "]"
                   << "  : Predicate " << pred_string;
#endif
  }

  // Check for existing cyclicity in the graph
  for (auto edge : graph->edges()) {
    Node* src = edge->src();
    Node* dst = edge->dst();

    // Skip source/sink
    if (!src->IsOp() || !dst->IsOp()) {
      continue;
    }

    // Skip NextIteration
    if (src->IsNextIteration() || dst->IsNextIteration()) {
      continue;
    }

    if (!gc.InsertEdge(cluster_map[src]->index, cluster_map[dst]->index)) {
      NGRAPH_VLOG(5) << "Failing due to cycle";
      return errors::Unimplemented(
          "Input graph has a cycle (inserting an edge from ",
          src->DebugString(), " to ", dst->DebugString(),
          " would create a cycle)");
    }
  }

  // If we wish to add a constraint that 2 particular nodes not lie in the same
  // cluster, then all we have to do is add 2 'shadow' edges and 1 'shadow' node
  // in the gc data structure between the 2 nodes. The shadow edges go from the
  // node closer to toposort source to the node closer to sink, through a shadow
  // node. src--->S--->dst. (not the other way round, else it would introduce a
  // cycle).
  // TF world node (o), gc world node (+), static input *
  // Normal edge traslation:
  // (o)---->(o)   ==>  (+)---->(+)
  // Static input edge translation:
  // (o)---->*(o)  ==>  (+)---->(+)
  //                     |       ^
  //                     |       |
  //                      --(+)--

  // The contraction only happens on 'real' edges (edges that are
  // present in the TF graph itself). Therefore the shadow edges in the gc
  // data structure will never suffer contraction. Anytime the shadow path's src
  // and dst attempt a merge (by contracting some real edge between them),
  // the shadow path will introduce a cycle and not allow it

  // Warning: this relies on the fact that we attempt to contract 'real' edges
  // from the TF graph. For optimization, one might attempt to contract the gc
  // edges, which keep decreasing unlike the TF edges. But this fix would break
  // then, since we have broken the contract that an edge in gc implies an edge
  // in TF in this fix
  for (auto node : graph->op_nodes()) {
    std::vector<int32> static_inputs;
    GetStaticInputs(node, &static_inputs);
    if (static_inputs.size() > 0) {
      std::vector<const Edge*> edges_to_node;
      TF_RETURN_IF_ERROR(node->input_edges(&edges_to_node));
      for (auto static_inp_idx : static_inputs) {
        auto static_edge = edges_to_node[static_inp_idx];
        if (static_edge->src()->type_string() != "Const") {
          int shadow_node_index = gc.NewNode();
          bool gc_success = gc.InsertEdge(
              cluster_map[static_edge->src()]->index, shadow_node_index);
          gc_success &= gc.InsertEdge(shadow_node_index,
                                      cluster_map[static_edge->dst()]->index);
          if (!gc_success)
            return errors::Internal(
                "Unable to create shadow edges in GraphCycles");
        }
      }
    }
  }

  NGRAPH_VLOG(2) << "Starting contraction";
  bool changed;
  bool collect_non_contracting_edge_info = false;  // Must init with false

  // 7 exhaustive reasons why edges might non contract
  // The reasons are not mutually exclusive, but there is an order of priority
  // that makes them mutually exclusive
  enum EdgeNonContractionReasons {
    NOTANOP,      // edge connects to non-ops
    UNSUPPORTED,  // either the src or dst is an unsupported op
    DEADNESS,     // deadness criteria not met
    BACKEND,      // different backends
    SAMECLUSTER,  // both ends lie in the same cluster
    STATICINPUT,  // static input in dst (not fed by const)
    PATHEXISTS    // base case reason. contraction causes cycles
  };
  // a cluster pair is the string "cluster1_id, cluster2_id"
  // Using string, because a pair won't hash unless implemented
  // Note that we store a vector of "reasons", because there could be multiple
  // reasons
  using ClusterPairToReason =
      std::unordered_map<std::string, std::vector<EdgeNonContractionReasons>>;
  ClusterPairToReason cluster_separation_reason;
  auto get_string_key = [](int x, int y) {
    return to_string(x) + "," + to_string(y);
  };
  // (src id, dst id) -> (src predicate, dst predicate, other neighbours
  // predicates)
  std::unordered_map<std::string, tuple<string, string, vector<string>>>
      deadness_info;

  do {
    changed = false;

    for (auto edge : graph->edges()) {
      Node* src = edge->src();
      Node* dst = edge->dst();

      int src_index = cluster_map[src]->index;
      int dst_index = cluster_map[dst]->index;

      if (!src->IsOp() || !dst->IsOp()) {
        if (collect_non_contracting_edge_info) {
          cluster_separation_reason[get_string_key(src_index, dst_index)]
              .push_back(EdgeNonContractionReasons::NOTANOP);
        }
        continue;
      }

      if (!NodeIsMarkedForClustering(src) || !NodeIsMarkedForClustering(dst)) {
        NGRAPH_VLOG(5) << "Skipping (not marked): " << src->name() << "["
                       << edge->src_output() << "]@" << src_index << " -> "
                       << dst->name() << "[" << edge->dst_input() << "]@"
                       << dst_index;
        if (collect_non_contracting_edge_info) {
          cluster_separation_reason[get_string_key(src_index, dst_index)]
              .push_back(EdgeNonContractionReasons::UNSUPPORTED);
        }
        continue;
      }

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
      // check if the edge can be contracted with respect to deadness
      bool is_deadness_ok = false;
      TF_RETURN_IF_ERROR(
          CanContractEdgeDeadnessCheck(edge, cluster_map, is_deadness_ok));
      if (!is_deadness_ok) {
        // do not contract, src and dst node cannot be in the same cluster
        NGRAPH_VLOG(5) << "Skipping (deadness not ok): " << src->name() << "["
                       << edge->src_output() << "]@" << src_index << " -> "
                       << dst->name() << "[" << edge->dst_input() << "]@"
                       << dst_index;
        if (collect_non_contracting_edge_info) {
          cluster_separation_reason[get_string_key(src_index, dst_index)]
              .push_back(EdgeNonContractionReasons::DEADNESS);

          auto src_cluster = cluster_map[src];
          auto dst_cluster = cluster_map[dst];
          vector<string> neighbours_predicate;
          // Collect predicates of src's neighbours (except dst)
          for (const Edge* src_cluster_edge : src_cluster->outgoing_edges) {
            if (src_cluster_edge != edge) {
              neighbours_predicate.push_back(
                  cluster_map[src_cluster_edge->dst()]->predicate_string);
            }
          }
          deadness_info[get_string_key(src_index, dst_index)] = make_tuple(
              cluster_map.at(src)->predicate_string,
              cluster_map.at(dst)->predicate_string, neighbours_predicate);
        }
        continue;
      }
#endif

      // check if the edge can be constracted with respect to backend
      bool is_backend_ok = false;
      TF_RETURN_IF_ERROR(
          CanContractEdgeBackendCheck(edge, cluster_map, is_backend_ok));
      if (!is_backend_ok) {
        NGRAPH_VLOG(5) << "Skipping (backend not ok): " << src->name() << "["
                       << edge->src_output() << "]@" << src_index << " -> "
                       << dst->name() << "[" << edge->dst_input() << "]@"
                       << dst_index;
        if (collect_non_contracting_edge_info) {
          cluster_separation_reason[get_string_key(src_index, dst_index)]
              .push_back(EdgeNonContractionReasons::BACKEND);
        }
        // do not contract, src and dst node cannot be in the same cluster
        continue;
      }

      // Check if contracting the edge will lead to cycles
      // if not, MergeClusters
      if (gc.HasEdge(src_index, dst_index) &&
          gc.ContractEdge(src_index, dst_index)) {
        MergeClusters(edge, cluster_map);
        // something changed
        changed = true;
      } else {
        if (collect_non_contracting_edge_info) {
          // either static input
          // or there exists a longer path, so contracting this edge causes
          // cycles
          std::vector<int32> static_inputs;
          GetStaticInputs(dst, &static_inputs);
          bool is_static = std::find(static_inputs.begin(), static_inputs.end(),
                                     edge->dst_input()) != static_inputs.end();
          // 3 possible reasons here:
          // src dst lies in same cluster, so nothing to do (trivial cycle
          // induced in graphcycles)
          // dst has static input
          // a longer irreducible path exists
          cluster_separation_reason[get_string_key(src_index, dst_index)]
              .push_back(src_index == dst_index
                             ? EdgeNonContractionReasons::SAMECLUSTER
                             : (is_static
                                    ? EdgeNonContractionReasons::STATICINPUT
                                    : EdgeNonContractionReasons::PATHEXISTS));
        }
      }
    }

    if (!changed && config::IsLoggingPlacement()) {
      // This will be entered only once if logging is enabled
      // When entered, it will force the do-while to run one last time,
      // collecting information
      if (!collect_non_contracting_edge_info) {
        changed = true;
        collect_non_contracting_edge_info = true;
      }
    }
  } while (changed);

  NGRAPH_VLOG(2) << "Contraction done";

  NGRAPH_VLOG(2) << "Starting tagging";
  std::set<Cluster*> seen;
  unordered_map<int, int> cluster_to_encapsulate;
  for (auto kv : cluster_map) {
    auto cluster = kv.second.get();
    if (seen.count(cluster) != 0) {
      continue;
    }

    bool has_ngraph_ops = false;
    bool has_non_ngraph_ops = false;

    for (auto node : cluster->nodes) {
      if (NodeIsMarkedForClustering(node)) {
        has_ngraph_ops = true;

// Some sanity checks for deadness
#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
        TF_RETURN_IF_ERROR(CheckNodeClusterAssignmentWRTDeadness(
            node, nodes_predicate_map, cluster_map));
#endif
      } else {
        has_non_ngraph_ops = true;
      }
    }

    if (has_ngraph_ops && has_non_ngraph_ops) {
      NGRAPH_VLOG(2) << "Cluster " << cluster->index
                     << " has both nGraph and non-nGraph nodes";
      for (auto node : cluster->nodes) {
        NGRAPH_VLOG(2) << (NodeIsMarkedForClustering(node)
                               ? "nGraph node: "
                               : "non-nGraph node: ")
                       << node->name() << " [" << node->type_string() << "]";
      }
      return errors::Internal("Cluster ", cluster->index,
                              " has both nGraph and non-nGraph nodes");
    }

    if (!has_ngraph_ops) {
      seen.insert(cluster);
      continue;
    }

    int cluster_idx = NGraphClusterManager::NewCluster();

    for (auto node : cluster->nodes) {
      if (NGRAPH_VLOG_IS_ON(5)) {
        NGRAPH_VLOG(5) << ">> cluster " << cluster_idx << ": " << node->id()
                       << " " << node << " :: " << node->name() << " ["
                       << node->type_string() << "]";
      }

      if (!NodeIsMarkedForClustering(node)) {
        return errors::Internal("Node ", node->DebugString(),
                                " was not marked for clustering but was "
                                "placed in an nGraph cluster.");
      }

      // TODO(amprocte): move attr name to a constant
      node->AddAttr("_ngraph_cluster", cluster_idx);

      if (config::IsLoggingPlacement()) {
        // map from cluster id to ngraph_cluster id
        cluster_to_encapsulate[cluster->index] = cluster_idx;
      }
    }

    seen.insert(cluster);
  }
  NGRAPH_VLOG(2) << "Tagging done";

  if (config::IsLoggingPlacement()) {
    int num_reasons = 7;  // the number of elements in the reasons enum
    // histogram of reasons of non-contraction of clusters
    vector<int> reason_count_clusters(num_reasons, 0);
    vector<int> reason_count_encapsulates(num_reasons, 0);
    int num_non_contracted = 0;
    std::vector<string> reason_string(  // to convert the enum to string
        {"NOTANOP", "UNSUPPORTED", "DEADNESS", "BACKEND", "SAMECLUSTER",
         "STATICINPUT", "PATHEXISTS"});
    auto forbidden_reasons_for_not_merging_clusters =
        std::set<EdgeNonContractionReasons>{
            EdgeNonContractionReasons::NOTANOP,
            EdgeNonContractionReasons::UNSUPPORTED,
            EdgeNonContractionReasons::SAMECLUSTER};
    std::function<bool(EdgeNonContractionReasons)> is_forbidden_reason =
        [&forbidden_reasons_for_not_merging_clusters](
            EdgeNonContractionReasons r) -> bool {
      return forbidden_reasons_for_not_merging_clusters.find(r) !=
             forbidden_reasons_for_not_merging_clusters.end();
    };
    std::cout
        << "Encapsulate i->j: non contraction reason histogram (Cannot be "
           "UNSUPPORTED, NOTANOP or SAMECLUSTER because unsupported ops will "
           "not be "
           "assigned an encapsulate)\n";
    for (auto it : cluster_separation_reason) {
      num_non_contracted += it.second.size();
      auto cluster_id_vector = ng::split(it.first, ',');
      // function to find if this cluster became an ngraph_cluster
      // returns ngraph_cluster id if yes, else returns -1
      auto find_in_map = [&cluster_to_encapsulate, &cluster_id_vector](int x) {
        auto itr = cluster_to_encapsulate.find(stoi(cluster_id_vector[x]));
        return itr == cluster_to_encapsulate.end() ? -1 : itr->second;
      };
      int src_encapsulate = find_in_map(0);
      int dst_encapsulate = find_in_map(1);
      bool both_src_dst_are_encapsulates =
          src_encapsulate >= 0 && dst_encapsulate >= 0;
      bool src_dst_are_distinct = src_encapsulate != dst_encapsulate;
      vector<int> reason_count_encapsulates_for_pair(num_reasons, 0);
      bool pair_has_reason = false;
      string deadness_string = "";
      for (auto& inner_itr : it.second) {
        // This if checks if the pair are 2 distinct encapsulates
        // In which case it asserts certain non-merging reasons are not possible
        // And also prints the reasons of non-merging
        if (both_src_dst_are_encapsulates && src_dst_are_distinct) {
          if (is_forbidden_reason(inner_itr)) {
            return errors::Internal(
                inner_itr,
                " should not be a reason why 2 encapsulates did not "
                "merge, because unsupported ops would not end up in "
                "encapsulates");
          }
          pair_has_reason = true;
          reason_count_encapsulates_for_pair[inner_itr]++;
          auto deadness_itr = deadness_info.find(it.first);
          if (deadness_itr != deadness_info.end()) {
            auto deadness_predicates_tpl = deadness_itr->second;
            deadness_string +=
                ("Source[" + to_string(src_encapsulate) + "] predicate: " +
                 std::get<0>(deadness_predicates_tpl) + " Destination[" +
                 to_string(dst_encapsulate) + "] predicate: " +
                 std::get<1>(deadness_predicates_tpl) +
                 " Neighbours predicates: " +
                 ng::join(std::get<2>(deadness_predicates_tpl)) + "\n");
          }
        }
        reason_count_clusters[inner_itr]++;
      }  // end of the for over each cluster pair's reason vector

      if (deadness_string != "") {
        std::cout << "Deadness predicates information\n";
        std::cout << deadness_string;
      }

      if (pair_has_reason) {
        std::cout << src_encapsulate << "->" << dst_encapsulate << ": ";
        for (int reason_id = 0; reason_id < num_reasons; reason_id++) {
          if (!is_forbidden_reason(
                  static_cast<EdgeNonContractionReasons>(reason_id))) {
            // Update global histogram with current pair's counts
            reason_count_encapsulates[reason_id] +=
                reason_count_encapsulates_for_pair[reason_id];
            cout << reason_string[reason_id] << ":"
                 << reason_count_encapsulates_for_pair[reason_id]
                 << (reason_id < (num_reasons - 1) ? ", " : "");
          }
        }
        std::cout << endl;
      }
    }  // end of the for over cluster_separation_reason
    std::cout << endl;
    if (num_non_contracted != graph->num_edges()) {
      return errors::Internal(
          "Number of non contracted edges ", num_non_contracted,
          " should match number of edges ", graph->num_edges());
    }

    auto print_reason_summary = [&reason_string, &num_reasons](
        vector<int> reasons_count,
        std::function<bool(EdgeNonContractionReasons)>
            forbidden_reasons_filter) {
      bool first = true;
      for (int i = 0; i < num_reasons; i++) {
        if (!forbidden_reasons_filter(
                static_cast<EdgeNonContractionReasons>(i))) {
          std::cout << (first ? "NGTF_SUMMARY: " : "") << reason_string[i]
                    << ": " << reasons_count[i]
                    << (i < (num_reasons - 1) ? ", " : "\n");
          first = false;
        }
      }
    };
    std::cout << "NGTF_SUMMARY: Summary of reasons why a pair of edge "
                 "connected encapsulates did not merge\n";
    print_reason_summary(reason_count_encapsulates, is_forbidden_reason);
    std::cout << "NGTF_SUMMARY: Summary of reasons why a pair of edge "
                 "connected clusters did not merge\n";
    print_reason_summary(reason_count_clusters,
                         [](EdgeNonContractionReasons x) { return false; });
  }

  return Status::OK();
}

// Updates cluster with the assigned cluster-id of the node
Status GetNodeCluster(const Node* node, int* cluster) {
  // TODO(amprocte): move attr name to a constant
  Status s = GetNodeAttr(node->attrs(), "_ngraph_cluster", cluster);
  if (s != Status::OK()) {
    *cluster = -1;
  }
  return s;
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
