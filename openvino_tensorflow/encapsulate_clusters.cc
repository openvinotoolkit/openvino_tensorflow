/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/version.h"
#if (TF_MAJOR_VERSION >= 2) && (TF_MINOR_VERSION > 2)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#else
#include "tensorflow/core/graph/graph_constructor.h"
#endif
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/device_name_utils.h"

#include "api.h"
#include "logging/ovtf_log.h"
#include "logging/tf_graph_writer.h"
#include "openvino_tensorflow/assign_clusters.h"
#include "openvino_tensorflow/cluster_manager.h"
#include "openvino_tensorflow/encapsulate_clusters.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_builder.h"
#include "openvino_tensorflow/ovtf_utils.h"
#include "openvino_tensorflow/version.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

//
// For each cluster K in the input graph, the encapsulation pass takes the set
// of all nodes in K and replaces them with a single NGraphEncapsulate op that
// stands in for the internal subgraph represented by the cluster K.
//
// TODO(amprocte): Point to some more documentation on what we're doing here...
//

// begin code copied and pasted (and modified) from graph.cc...
void Encapsulator::AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {
  if (src_slot == Graph::kControlSlot) {
    dst->add_input(strings::StrCat("^", src_name));
  } else if (src_slot == 0) {
    dst->add_input(src_name.data(), src_name.size());
  } else {
    dst->add_input(strings::StrCat(src_name, ":", src_slot));
  }
}
// ...end code copied and pasted (and modified) from graph.cc

Status EncapsulateClusters(
    Graph* graph, int graph_id,
    const std::unordered_map<std::string, std::string>& device_config) {
  Encapsulator enc(graph);
  OVTF_VLOG(3) << "Running AnalysisPass in EncapsulateClusters";
  TF_RETURN_IF_ERROR(enc.AnalysisPass());
  OVTF_VLOG(3) << "Running RewritePass in EncapsulateClusters";
  TF_RETURN_IF_ERROR(enc.RewritePass(graph_id, device_config));

  set<int> newly_created_cluster_ids;
  TF_RETURN_IF_ERROR(enc.GetNewClusterIDs(newly_created_cluster_ids));

  // Pass 9 (optional, only run if environment variable
  // OPENVINO_TF_DUMP_CLUSTERS is set): validate the graph def, and
  // make sure we can construct a graph from it.
  if (std::getenv("OPENVINO_TF_DUMP_CLUSTERS")) {
    for (auto& cluster_idx : newly_created_cluster_ids) {
      TF_RETURN_IF_ERROR(graph::ValidateGraphDef(
          *NGraphClusterManager::GetClusterGraph(cluster_idx),
          *OpRegistry::Global()));

      Graph g(OpRegistry::Global());
      GraphConstructorOptions opts;
      opts.allow_internal_ops = true;
      TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
          opts, *NGraphClusterManager::GetClusterGraph(cluster_idx), &g));

      std::stringstream ss;
      ss << "ovtf_cluster_" << cluster_idx;
      std::string filename_prefix = ss.str();

      GraphToPbTextFile(&g, filename_prefix + ".pbtxt");
      GraphToDotFile(&g, filename_prefix + ".dot",
                     "nGraph Cluster Dump: " + filename_prefix);
    }
  }

  return Status::OK();
}

Encapsulator::Encapsulator(Graph* g)
    : graph(g), analysis_done(false), rewrite_done(false) {}

Status Encapsulator::AnalysisPass() {
  if (rewrite_done) {
    return errors::Internal(
        "In Encapsulator, AnalysisPass called after RewritePass was already "
        "done");
  }

  if (analysis_done) {
    return errors::Internal(
        "In Encapsulator, AnalysisPass called more than once");
  }
  // Pass 1: Populate the cluster-index-to-device name map for each existing
  // cluster. PIGGYBACKING BACKEND TEST HERE, THEY WILL GET COMBINED INTO ONE
  for (auto node : graph->op_nodes()) {
    int cluster_idx;

    if (GetNodeCluster(node, &cluster_idx) != Status::OK()) {
      continue;
    }

    auto it = device_name_map.find(cluster_idx);
    if (it != device_name_map.end()) {
      if (it->second != node->assigned_device_name()) {
        std::stringstream ss_err;
        ss_err << "Node " << node->name() << " in cluster " << cluster_idx
               << " has assigned device " << node->assigned_device_name()
               << " but another node with assigned device " << it->second
               << " has already been seen in the same cluster";

        return errors::Internal(ss_err.str());
      }
    } else {
      OVTF_VLOG(3) << "setting cluster " << cluster_idx
                   << " requested device to '" << node->assigned_device_name()
                   << "'";
      device_name_map[cluster_idx] = node->assigned_device_name();
    }
  }

  // Pass 2: Find all nodes that are feeding into/out of each cluster, and
  // add inputs for them to the corresponding FunctionDef(s).
  std::map<int, int> retval_index_count;
  std::map<int, int> arg_index_count;
  int count_arg = 0, count_retval = 0, count_both_arg_retval = 0,
      count_free = 0, count_encapsulated = 0, count_tot = 0;
  for (auto edge : graph->edges()) {
    count_tot++;
    // TODO(amprocte): should actually keep of these. During clustering we
    // will already have identified any intra-cluster control deps. Should
    // maintain inter-cluster control deps.
    if (edge->IsControlEdge()) {
      count_free++;
      continue;
    }

    Node* src = edge->src();
    Node* dst = edge->dst();

    // TODO(amprocte): the following rejects edges involving source/sink. Is
    // that what we want to do?
    if (!src->IsOp() || !dst->IsOp()) {
      count_free++;
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
      count_encapsulated++;
      continue;
    }

    // Some debug logging...
    DataType dt = dst->input_type(edge->dst_input());
    std::string flow_kind = dst_clustered && src_clustered
                                ? "cross-flow"
                                : dst_clustered ? "in-flow" : "out-flow";

    OVTF_VLOG(4) << "found " << flow_kind << ": " << src->name() << "["
                 << edge->src_output() << "] in " << src_cluster_idx << " to "
                 << dst->name() << "[" << edge->dst_input() << "] in "
                 << dst_cluster_idx << ", datatype: " << dt;

    bool edge_is_retval = false, edge_is_arg = false;

    // If the source node lies within a cluster, we must create an output for
    // it from the source cluster. For the moment we will just store this
    // fact in the output_remap_map.
    if (src_clustered &&
        output_remap_map.find(std::make_tuple(src->id(), edge->src_output())) ==
            output_remap_map.end()) {
      output_remap_map[std::make_tuple(src->id(), edge->src_output())] =
          std::make_tuple(src_cluster_idx,
                          cluster_output_dt_map[src_cluster_idx].size());

      std::stringstream ss;
      ss << "ngraph_output_" << cluster_output_dt_map[src_cluster_idx].size();
      string output_name = ss.str();
      auto new_output_node_def =
          NGraphClusterManager::GetClusterGraph(src_cluster_idx)->add_node();

#ifdef _WIN32
      auto src_node_def = src->def();
      *new_output_node_def = src_node_def;
      new_output_node_def->Clear();
#endif
      new_output_node_def->set_name(output_name);
      new_output_node_def->set_op("_Retval");
      edge_is_retval = true;

      std::stringstream ss_input_to_retval;
      ss_input_to_retval << src->name() << ":" << edge->src_output();

      new_output_node_def->add_input(ss_input_to_retval.str());

      SetAttrValue(dt, &((*(new_output_node_def->mutable_attr()))["T"]));
      SetAttrValue(retval_index_count[src_cluster_idx],
                   &((*(new_output_node_def->mutable_attr()))["index"]));

      retval_index_count[src_cluster_idx]++;

      cluster_output_dt_map[src_cluster_idx].push_back(dt);
    }

    // If the destination node lies within a cluster, we must create an input
    // for the source node to the destination cluster. For the moment we will
    // just store this fact in the input_remap_map.
    if (dst_clustered &&
        input_remap_map.find(
            std::make_tuple(dst_cluster_idx, src->id(), edge->src_output())) ==
            input_remap_map.end()) {
      input_remap_map[std::make_tuple(dst_cluster_idx, src->id(),
                                      edge->src_output())] =
          cluster_input_map[dst_cluster_idx].size();

      std::stringstream ss;
      ss << "ngraph_input_" << cluster_input_map[dst_cluster_idx].size();
      std::string new_input_name = ss.str();

      input_rename_map[std::make_tuple(dst_cluster_idx, src->name(),
                                       edge->src_output())] = new_input_name;
      string input_prov_tag = src->name();

      auto new_input_node_def =
          NGraphClusterManager::GetClusterGraph(dst_cluster_idx)->add_node();

#ifdef _WIN32
      auto src_node_def = src->def();
      *new_input_node_def = src_node_def;
      new_input_node_def->Clear();
#endif
      new_input_node_def->set_name(new_input_name);
      new_input_node_def->set_op("_Arg");
      edge_is_arg = true;

      SetAttrValue(dt, &((*(new_input_node_def->mutable_attr()))["T"]));
      SetAttrValue(arg_index_count[dst_cluster_idx],
                   &((*(new_input_node_def->mutable_attr()))["index"]));
      SetAttrValue(input_prov_tag,
                   &((*(new_input_node_def->mutable_attr()))["_prov_tag"]));

      if (src->type_string() == "ReadVariableOp") {
        SetAttrValue(
            true, &((*(new_input_node_def->mutable_attr()))["_is_variable"]));
      } else {
        SetAttrValue(
            false, &((*(new_input_node_def->mutable_attr()))["_is_variable"]));
      }

      vector<int> static_input_indexes;
      try {
        GetNodeAttr(dst->attrs(), "_ovtf_static_inputs", &static_input_indexes);
      } catch (const std::exception&) {
        OVTF_VLOG(1) << "Node " << dst->name()
                     << " does not have static inputs";
      }
      if (std::find(static_input_indexes.begin(), static_input_indexes.end(),
                    edge->dst_input()) != static_input_indexes.end()) {
        SetAttrValue(
            true, &((*(new_input_node_def->mutable_attr()))["_static_input"]));
      } else if (src->type_string() == "Const") {
        // TODO: This check might be redundant
        SetAttrValue(
            true, &((*(new_input_node_def->mutable_attr()))["_static_input"]));
      } else {
        SetAttrValue(
            false, &((*(new_input_node_def->mutable_attr()))["_static_input"]));
      }

      arg_index_count[dst_cluster_idx]++;

      cluster_input_map[dst_cluster_idx].push_back(
          std::make_tuple(src->id(), edge->src_output(), dt));
    }

    if (api::IsLoggingPlacement()) {
      if (edge_is_arg && edge_is_retval) {
        count_both_arg_retval++;
      } else {
        if (edge_is_arg) {
          count_arg++;
        } else {
          count_retval++;
        }
      }
    }
  }

  if (api::IsLoggingPlacement()) {
    int computed_edge_number = count_arg + count_retval +
                               count_both_arg_retval + count_free +
                               count_encapsulated;
    std::cout << "OVTF_SUMMARY: Types of edges:: args: " << count_arg
              << ", retvals: " << count_retval
              << ", both arg and retval: " << count_both_arg_retval
              << ", free: " << count_free
              << ", encapsulated: " << count_encapsulated
              << ", total: " << count_tot
              << ", computed total: " << computed_edge_number << endl;
    std::cout << "\n=============Ending sub-graph logs=============\n";
    if (!(computed_edge_number == count_tot &&
          count_tot == graph->num_edges())) {
      return errors::Internal("Computed number of edges ", computed_edge_number,
                              " and counted number of edges ", count_tot,
                              " and number of edges from querying TF api ",
                              graph->num_edges(), " do not match up\n");
    }
  }

  // Pass 5: Make copies of all clustered nodes inside the cluster graphs,
  // rewiring the inputs in their NodeDefs as we go.

  // Originally Pass 5 ran after Pass 4 ofcourse. But now calling it right after
  // Pass 2 in the Analysis Phase.
  // Pass 4 took care of removing some inter-cluster control edges, so by the
  // time Pass 5 was run, those control inputs would have been removed
  // But now since Pass 5 is running before Pass 4, we must take special care to
  // not add inter-cluster (or TF to cluster) control edges in the graphdef we
  // copy into the ClusterManager
  // This is taken care of in the "if (edge->IsControlEdge())" line in the for
  // loop over all edges
  for (auto node : graph->op_nodes()) {
    int cluster_idx;

    if (GetNodeAttr(node->attrs(), "_ovtf_cluster", &cluster_idx) !=
        Status::OK()) {
      continue;
    }

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

    auto node_def =
        NGraphClusterManager::GetClusterGraph(cluster_idx)->add_node();
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

  analysis_done = true;

  return Status::OK();
}

Status Encapsulator::RewritePass(
    int graph_id,
    const std::unordered_map<std::string, std::string>& device_config) {
  if (!analysis_done) {
    return errors::Internal(
        "In Encapsulator, called RewritePass without calling AnalysisPass");
  }
  if (rewrite_done) {
    return errors::Internal(
        "In Encapsulator, called RewritePass more than once");
  }
  // Pass 3: Create encapsulation nodes for all clusters.
  for (auto& kv : device_name_map) {
    int cluster_idx = kv.first;
    std::stringstream ss;
    ss << "ovtf_cluster_" << cluster_idx;

    string encap_node_name = ss.str();
    std::vector<DataType> input_types;
    std::vector<NodeBuilder::NodeOut> inputs;

    for (auto& tup : cluster_input_map[cluster_idx]) {
      int src_node_id = -1;
      int src_output_idx = -1;
      DataType dt;
      std::tie(src_node_id, src_output_idx, dt) = tup;

      input_types.push_back(dt);

      inputs.push_back(
          NodeBuilder::NodeOut(graph->FindNodeId(src_node_id), src_output_idx));
    }

    Node* n;
    NodeBuilder nb = NodeBuilder(encap_node_name, "_nGraphEncapsulate")
                         .Attr("ovtf_cluster", cluster_idx)
                         .Attr("Targuments", input_types)
                         .Attr("Tresults", cluster_output_dt_map[cluster_idx])
                         .Attr("ngraph_graph_id", graph_id)
                         .Device(device_name_map[cluster_idx])
                         .Input(inputs);
    if (!device_config.empty()) {
      OVTF_VLOG(3) << "Device config is not empty";
      for (auto const& i : device_config) {
        // Adding the optional attributes
        OVTF_VLOG(3) << "Attaching Attribute " << i.first << " Val "
                     << i.second;
        nb.Attr(i.first, i.second);
      }
    }

    // Find Static Inputs And Add as an attribute
    vector<int> static_input_indexes;
    GraphDef* gdef_for_current_encapsulate;
    gdef_for_current_encapsulate =
        NGraphClusterManager::GetClusterGraph(cluster_idx);
    if (gdef_for_current_encapsulate == nullptr) {
      return errors::Internal(
          "Did not find encapsulated graph in cluster manager for node ",
          encap_node_name);
    }
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    Graph graph_for_current_encapsulate(OpRegistry::Global());
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        opts, *gdef_for_current_encapsulate, &graph_for_current_encapsulate));

    TF_RETURN_IF_ERROR(
        GetStaticInputs(&graph_for_current_encapsulate, &static_input_indexes));
#ifdef _WIN32
    if (!static_input_indexes.empty()) {
      nb.Attr("_ovtf_static_inputs", static_input_indexes);
    }
#else
    nb.Attr("_ovtf_static_inputs", static_input_indexes);
#endif

    Status status = nb.Finalize(graph, &n);
    TF_RETURN_IF_ERROR(status);
    n->set_assigned_device_name(device_name_map[cluster_idx]);

    cluster_node_map[cluster_idx] = n;
  }

  // Pass 4: Remap all non-clustered inputs that are reading from
  // encapsulated edges, and all control edges that cross cluster
  // boundaries.

  // Copy the edge pointers, so as not to invalidate the iterator.
  std::vector<Edge*> edges;
  for (auto edge : graph->edges()) {
    edges.push_back(edge);
  }

  for (auto edge : edges) {
    int src_cluster_idx;
    bool src_clustered =
        (GetNodeCluster(edge->src(), &src_cluster_idx) == Status::OK());
    int dst_cluster_idx;
    bool dst_clustered =
        (GetNodeCluster(edge->dst(), &dst_cluster_idx) == Status::OK());

    if (src_cluster_idx == dst_cluster_idx) {
      continue;
    }

    if (edge->IsControlEdge()) {
      if (src_clustered && dst_clustered) {
        graph->RemoveControlEdge(edge);
        graph->AddControlEdge(cluster_node_map[src_cluster_idx],
                              cluster_node_map[dst_cluster_idx]);
      } else if (src_clustered) {
        Node* dst = edge->dst();
        graph->RemoveControlEdge(edge);
        graph->AddControlEdge(cluster_node_map[src_cluster_idx], dst);
      } else if (dst_clustered) {
        Node* src = edge->src();
        graph->RemoveControlEdge(edge);
        graph->AddControlEdge(src, cluster_node_map[dst_cluster_idx]);
      }
    } else {
      // This is handled at a later stage (TODO(amprocte): explain)
      if (dst_clustered) {
        continue;
      }

      auto it = output_remap_map.find(
          std::make_tuple(edge->src()->id(), edge->src_output()));

      if (it == output_remap_map.end()) {
        continue;
      }

      int cluster_idx = -1;
      int cluster_output = -1;
      std::tie(cluster_idx, cluster_output) = it->second;

      Status status =
          graph->UpdateEdge(cluster_node_map[cluster_idx], cluster_output,
                            edge->dst(), edge->dst_input());
      TF_RETURN_IF_ERROR(status);
    }
  }

  // Pass 6: Remove clustered nodes from the graph.
  std::vector<Node*> nodes_to_remove;
  for (auto node : graph->op_nodes()) {
    int cluster_idx;

    if (GetNodeAttr(node->attrs(), "_ovtf_cluster", &cluster_idx) !=
        Status::OK()) {
      continue;
    }
    nodes_to_remove.push_back(node);
  }

  for (auto node : nodes_to_remove) {
    OVTF_VLOG(4) << "Removing: " << node->name();
    graph->RemoveNode(node);
  }

  rewrite_done = true;
  return Status::OK();
}

Status Encapsulator::GetNewClusterIDs(set<int>& result) {
  if (!analysis_done) {
    return errors::Internal(
        "In Encapsulator, called GetNewClusterIDs without calling "
        "AnalysisPass");
  }
  result.clear();
  for (auto it = device_name_map.begin(); it != device_name_map.end(); ++it) {
    result.insert(it->first);
  }
  return Status::OK();
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
