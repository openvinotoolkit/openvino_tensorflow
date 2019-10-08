/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/device_name_utils.h"

#include "logging/ngraph_log.h"
#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_cluster_manager.h"
#include "ngraph_bridge/ngraph_encapsulate_clusters.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_partial_shapes.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// For each cluster K in the input graph, the encapsulation pass takes the set
// of all nodes in K and replaces them with a single NGraphEncapsulate op that
// stands in for the internal subgraph represented by the cluster K.
//
// TODO(amprocte): Point to some more documentation on what we're doing here...
//

// begin code copied and pasted (and modified) from graph.cc...
static void AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {
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
    Graph* graph, int graph_id, FunctionDefLibrary* fdeflib,
    std::unordered_map<std::string, std::string> device_config,
    AOTInfo aot_info) {
  // A map from cluster indices to the expected device name for nodes
  // in that cluster.
  std::map<int, std::string> device_name_map;

  // We *should* eventually have a way of monitoring the device and the backend
  // together
  std::map<int, std::string> backend_name_map;

  // As we build the graph we will be tracking the.. TODO(amprocte): finish
  // this comment.
  std::map<std::tuple<int, int>, std::tuple<int, int>> output_remap_map;
  std::map<std::tuple<int, int, int>, int> input_remap_map;
  std::map<std::tuple<int, std::string, int>, string> input_rename_map;

  // A map from cluster indices to a vector of input data types.
  std::map<int, std::vector<std::tuple<int, int, DataType>>> cluster_input_map;
  // A map from cluster indices to a vector of output data types.
  std::map<int, std::vector<DataType>> cluster_output_dt_map;

  // A map from cluster indices to corresponding NGraphEncapsulate nodes.
  std::map<int, Node*> cluster_node_map;

  // Pass 1: Populate the cluster-index-to-device name map for each existing
  // cluster. PIGGYBACKING BACKEND TEST HERE, THEY WILL GET COMBINED INTO ONE
  for (auto node : graph->op_nodes()) {
    int cluster_idx;

    if (GetNodeCluster(node, &cluster_idx) != Status::OK()) {
      continue;
    }

    string node_backend;
    if (GetNodeBackend(node, &node_backend) != Status::OK()) {
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
      NGRAPH_VLOG(3) << "setting cluster " << cluster_idx
                     << " requested device to '" << node->assigned_device_name()
                     << "'";
      device_name_map[cluster_idx] = node->assigned_device_name();
    }

    auto itr = backend_name_map.find(cluster_idx);

    if (itr != backend_name_map.end()) {
      if (itr->second != node_backend) {
        std::stringstream ss_err;
        ss_err << "Node " << node->name() << " in cluster " << cluster_idx
               << " has assigned backend " << node_backend
               << " but another node with assigned backend " << it->second
               << " has already been seen in the same cluster";

        return errors::Internal(ss_err.str());
      }
    } else {
      NGRAPH_VLOG(3) << "setting cluster " << cluster_idx
                     << " requested backend to '" << node_backend << "'";
      backend_name_map[cluster_idx] = node_backend;
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

    NGRAPH_VLOG(4) << "found " << flow_kind << ": " << src->name() << "["
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

      auto new_input_node_def =
          NGraphClusterManager::GetClusterGraph(dst_cluster_idx)->add_node();
      new_input_node_def->set_name(new_input_name);
      new_input_node_def->set_op("_Arg");
      edge_is_arg = true;

      SetAttrValue(dt, &((*(new_input_node_def->mutable_attr()))["T"]));
      SetAttrValue(arg_index_count[dst_cluster_idx],
                   &((*(new_input_node_def->mutable_attr()))["index"]));

      arg_index_count[dst_cluster_idx]++;

      cluster_input_map[dst_cluster_idx].push_back(
          std::make_tuple(src->id(), edge->src_output(), dt));
    }

    if (config::IsLoggingPlacement()) {
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

  if (config::IsLoggingPlacement()) {
    int computed_edge_number = count_arg + count_retval +
                               count_both_arg_retval + count_free +
                               count_encapsulated;
    std::cout << "NGTF_SUMMARY: Types of edges:: args: " << count_arg
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

  // Pass 3: Create encapsulation nodes for all clusters.
  for (auto& kv : device_name_map) {
    int cluster_idx = kv.first;
    string cluster_backend = backend_name_map[cluster_idx];

    std::stringstream ss;
    ss << "ngraph_cluster_" << cluster_idx;

    std::vector<DataType> input_types;
    std::vector<NodeBuilder::NodeOut> inputs;

    for (auto& tup : cluster_input_map[cluster_idx]) {
      int src_node_id;
      int src_output_idx;
      DataType dt;
      std::tie(src_node_id, src_output_idx, dt) = tup;

      input_types.push_back(dt);

      inputs.push_back(
          NodeBuilder::NodeOut(graph->FindNodeId(src_node_id), src_output_idx));
    }

    Node* n;
    NodeBuilder nb =
        NodeBuilder(ss.str(), "NGraphEncapsulate")
            .Attr("ngraph_cluster", cluster_idx)
            .Attr("ngraph_backend",
                  BackendManager::GetBackendAttributeValues(cluster_backend)
                      .at("ngraph_backend"))
            .Attr("Targuments", input_types)
            .Attr("Tresults", cluster_output_dt_map[cluster_idx])
            .Attr("ngraph_graph_id", graph_id)
            .Device(device_name_map[cluster_idx])
            .Input(inputs);
    if (!device_config.empty()) {
      NGRAPH_VLOG(3) << "Device config is not empty";
      for (auto const& i : device_config) {
        // Adding the optional attributes
        NGRAPH_VLOG(3) << "Attaching Attribute " << i.first << " Val "
                       << i.second;
        nb.Attr(i.first, i.second);
      }
    }
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

      int cluster_idx;
      int cluster_output;
      std::tie(cluster_idx, cluster_output) = it->second;

      Status status =
          graph->UpdateEdge(cluster_node_map[cluster_idx], cluster_output,
                            edge->dst(), edge->dst_input());
      TF_RETURN_IF_ERROR(status);
    }
  }

  // Pass 5: Make copies of all clustered nodes inside the cluster graphs,
  // rewiring the inputs in their NodeDefs as we go.
  std::set<int> cluster_indices_for_this_graph;
  for (auto node : graph->op_nodes()) {
    int cluster_idx;

    if (GetNodeAttr(node->attrs(), "_ngraph_cluster", &cluster_idx) !=
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
        inputs.push_back(edge);
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
    cluster_indices_for_this_graph.insert(cluster_idx);
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

  // Pass 6: Remove clustered nodes from the graph.
  for (auto node : graph->op_nodes()) {
    int cluster_idx;

    if (GetNodeAttr(node->attrs(), "_ngraph_cluster", &cluster_idx) !=
        Status::OK()) {
      continue;
    }

    graph->RemoveNode(node);
  }

  // Pass 7: Insert to function library
  // Note: We loop over cluster_indices_for_this_graph and not all the
  // contents of ClusterManager
  for (const auto& cluster_idx : cluster_indices_for_this_graph) {
    // The transformation happening inside this loop is:
    // graphdef --> graph --> functiondef
    // NGraphClusterManager::GetClusterGraph(cluster_idx)-->subgraph-->fdef
    // TODO: whats the right flib to use in subgraph's constructor?
    Graph subgraph(graph->flib_def());
    // TODO: When this works, NGraphClusterManager can go away
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        GraphConstructorOptions(),
        *(NGraphClusterManager::GetClusterGraph(cluster_idx)), &subgraph));
    FunctionDef* fdef = fdeflib->add_function();
    // TODO: if func lib has func with same name etc?
    TF_RETURN_IF_ERROR(GraphToFunctionDef(
        subgraph, strings::StrCat("ngraph_cluster_", to_string(cluster_idx)),
        fdef));
  }

  // Pass 8:
  bool aot_requested;
  set<string> performed_aot_on_enc;
  std::set<std::map<std::string, vector<int>>> node_shapes_hints_sets;
  std::tie(aot_requested, node_shapes_hints_sets) = aot_info;
  if (aot_requested) {
    NGRAPH_VLOG(3) << "AOT requested";
    if (!ngraph_tf_is_grappler_enabled()) {
      return errors::Internal(
          "AOT requested for non grappler build. Please use grappler build if "
          "AOT is required");
    }
    string input_node_type = "Placeholder";
    // In case of grappler, we have Placeholder, which might contain shape info,
    // so it is possible we can aot without any provided shapes
    // in normal pass its args. unless shapes are provided there is no chance of
    // reading shapes from args.

    auto get_shape_for_node_from_shape_hint = [](Node* node,
                                                 ShapeHintMap single_hint) {
      auto find_itr = single_hint.find(node->name());
      return find_itr == single_hint.end() ? PartialShape()
                                           : PartialShape(find_itr->second);
    };

    auto hint_as_string = [](ShapeHintMap single_hint) {
      string hint_str;
      for (auto itr_node : single_hint) {
        hint_str +=
            ((itr_node.first) + ":[" + ng::join(itr_node.second) + "],");
      }
      return hint_str;
    };

    std::map<std::string, vector<int>> inputs_node_shapes_for_compilation;
    // map between node name and the PartialShape it contains
    std::map<std::string, PartialShape> node_partial_shape_map;
    // This is a map of placeholder names and the shapes we can infer from them
    std::map<std::string, vector<int>> shape_from_placeholders_as_hints;
    for (auto node : graph->op_nodes()) {
      if (node->type_string() == input_node_type) {
        NGRAPH_VLOG(5) << "Checking input for AOT: " << node->name() << "("
                       << node->type_string()
                       << "): " << node->attrs().SummarizeNode();
        // TODO: need to confirm if its _output_shapes or shape
        auto shape_field = node->attrs().Find("_output_shapes");
        if (shape_field == nullptr) {
          shape_field = node->attrs().Find("shape");
        }
        // It seems that _output_shapes is not found and hence the shape is
        // inferred only from the hints. however if "shape" is present, it is
        // empty, and in that case the empty shape and the rank!=0 hint fuse
        // to give an invalid shape according to our current logic. have to
        // modify that
        PartialShape partial_shape_from_node;
        if (shape_field != nullptr) {
          // Get shape from the node
          partial_shape_from_node = PartialShape(shape_field->shape());
        }
        NGRAPH_VLOG(5) << "For node " << node->name()
                       << " got shape from nose: "
                       << partial_shape_from_node.to_string();
        node_partial_shape_map.insert({node->name(), partial_shape_from_node});
        shape_from_placeholders_as_hints.insert(
            {node->name(), partial_shape_from_node.get_shape_vector()});
      }
    }

    // If no shape hints are provided but the placeholders contain complete
    // shape, then we still need to enter the for loop below to compute AOT.
    // Hence adding the shapes from placeholders as hints.
    if (node_shapes_hints_sets.size() == 0) {
      NGRAPH_VLOG(5) << "Using shapes from placeholders as hint";
      node_shapes_hints_sets.insert(shape_from_placeholders_as_hints);
    }
    // TODO: .....CHECK ABOVE IF

    // Iterate over each shape hint and see if they can be used
    for (ShapeHintMap single_hint : node_shapes_hints_sets) {
      // A boolean to determine if we can AOT for this single_hint
      bool can_aot = true;

      for (auto itr_single_hint : single_hint) {
        if (shape_from_placeholders_as_hints.find(itr_single_hint.first) ==
            shape_from_placeholders_as_hints.end()) {
          return errors::Internal("Passed hint for node ",
                                  itr_single_hint.first,
                                  " but there is no input with that name");
        }
      }

      for (auto node : graph->op_nodes()) {
        if (node->type_string() == input_node_type) {
          PartialShape partial_shape_from_node =
              node_partial_shape_map.at(node->name());

          PartialShape shape_hint_for_node =
              get_shape_for_node_from_shape_hint(node, single_hint);

          // If a shape has been found in the input node, match with
          // shape_hints if they exist
          PartialShape combined_shape_info;
          if (shape_hint_for_node.is_valid()) {
            NGRAPH_VLOG(5) << "For node " << node->name() << " shape hint (",
                hint_as_string(single_hint),
                ") for node is valid and is: " +
                    shape_hint_for_node.to_string();
            if (partial_shape_from_node.is_valid()) {
              NGRAPH_VLOG(5) << "Partial shape from node is also valid. So "
                                "will attempt to concretize if possible";
              if (partial_shape_from_node.size() == 0) {
                // TODO: revisit this if-else
                NGRAPH_VLOG(5) << "Partial shape from node is empty, so will "
                                  "use shape from hint";
                combined_shape_info = shape_hint_for_node;
              } else {
                NGRAPH_VLOG(5) << "Concretizing shape " +
                                      partial_shape_from_node.to_string() +
                                      "from node with hint for node, " +
                                      shape_hint_for_node.to_string();
                partial_shape_from_node.concretize(shape_hint_for_node);
                combined_shape_info = partial_shape_from_node;
              }
            } else {
              NGRAPH_VLOG(5) << "Partial shape from node is invalid. So using "
                                "hint for the node as shape";
              combined_shape_info = shape_hint_for_node;
            }
          } else {
            NGRAPH_VLOG(5) << "For node " << node->name()
                           << " shape hint (" + hint_as_string(single_hint) +
                                  ") for node is invalid";
            if (partial_shape_from_node.is_valid()) {
              // No shape hints found. But the node itself has some shape info
              NGRAPH_VLOG(5) << "Partial shape from node is valid and is: " +
                                    partial_shape_from_node.to_string();
              combined_shape_info = partial_shape_from_node;
            } else {
              NGRAPH_VLOG(5) << "Partial shape from node is invalid";
              combined_shape_info = PartialShape();
            }
          }

          can_aot = combined_shape_info.is_valid() &&
                    combined_shape_info.is_concrete();
          if (can_aot) {
            inputs_node_shapes_for_compilation[node->name()] =
                combined_shape_info.get_shape_vector();
          } else {
            // TODO: necessarily break? Maybe some things can be AOT, others
            // maybe not
            string fail_reason =
                (combined_shape_info.is_valid()
                     ? (node->name() + " could not be concretized")
                     : "it is invalid for " + node->name());
            return errors::Internal("Cannot AOT using this hint (",
                                    hint_as_string(single_hint), ") as ",
                                    fail_reason);
            break;
          }
        }  // end of if (node->type_string() == input_node_type)
      }    // End of for loop that goes through all nodes

      // Did we manage to concretize all input shapes?
      for (auto itr : node_partial_shape_map) {  // iterate over all inputs
        if (inputs_node_shapes_for_compilation.find(itr.first) ==
            inputs_node_shapes_for_compilation.end()) {
          can_aot = false;
          // TODO: print "this" hint
          return errors::Internal("Cannot AOT using this hint (",
                                  hint_as_string(single_hint), ") for ",
                                  (itr.first), " was not concretized");
        }
      }

      if (!can_aot) {
        return errors::Internal("AOT requested, but could not perform AOT");
      }
      for (auto node : graph->op_nodes()) {
        if (node->type_string() == "NGraphEncapsulate") {
          // Check inputs of the encapsulates. They can only be fed by fully
          // concrete shapes (after going through the shape hints) or consts
          std::vector<int32> st_inputs;
          GetStaticInputs(node, &st_inputs);
          // Current assumption is that only encapsulates without static
          // inputs are AOT
          if (st_inputs.size() != 0) {
            return errors::Internal(
                "AOT requested. Found an encapsulate with static inputs, but "
                "that is not supported");
          }

          std::vector<TensorShape> input_shapes;
          std::stringstream signature_ss;
          for (auto in_node : node->in_nodes()) {
            if (!in_node->IsSource()) {
              auto itr_shape =
                  inputs_node_shapes_for_compilation.find(in_node->name());
              if (itr_shape == inputs_node_shapes_for_compilation.end()) {
                // TODO: this error could potentially happen due to 2 reasons:
                // 1. Enough valid shape hints were not passed
                // 2. It is an encapsulate that has atleast 1 input fed by a
                // non-placeholder (like another TF node or another
                // encapsulate)
                // Later provide more explicit debug message (reason 1 or 2 or
                // anything else)
                return errors::Internal(
                    "AOT requested. Found an encapsulate that has a "
                    "non-concrete input");
              } else {
                std::vector<int64> converted_to_int64(itr_shape->second.begin(),
                                                      itr_shape->second.end());
                input_shapes.push_back(TensorShape(converted_to_int64));
                for (auto itr1 : itr_shape->second) {
                  signature_ss << itr1 << ",";
                }
                signature_ss << ";";
              }
            }
          }

          signature_ss << "/";
          string signature = signature_ss.str();
          NGRAPH_VLOG(3) << "Performing AOT for " << node->name()
                         << " for signature = " << signature << "\n";

          std::vector<const Tensor*> static_input_map;
          std::shared_ptr<ngraph::Function> ng_function;
          int cluster_idx;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(node->attrs(), "ngraph_cluster", &cluster_idx));
          GraphDef* gdef_for_current_encapsulate;
          gdef_for_current_encapsulate =
              NGraphClusterManager::GetClusterGraph(cluster_idx);
          GraphConstructorOptions opts;
          opts.allow_internal_ops = true;
          Graph graph_for_current_encapsulate(OpRegistry::Global());
          TF_RETURN_IF_ERROR(
              ConvertGraphDefToGraph(opts, *gdef_for_current_encapsulate,
                                     &graph_for_current_encapsulate));

          // get backend.
          // TODO: Note that this is code duplication of some stuff present
          // in NGraphEncapsulateOp
          // Once NGraphEncapsulateOp is refactored, this code should be
          // removed and a common function should be used

          // TODO: these sections can be hoisted out of the main loop
          std::string backend_name;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(node->attrs(), "ngraph_backend", &backend_name));
          std::string device_id;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(node->attrs(), "ngraph_device_id", &device_id));

          string op_backend_name;
          try {
            op_backend_name = BackendManager::GetBackendCreationString(
                backend_name, device_id);
          } catch (const std::exception& exp) {
            return errors::Internal(
                "Caught exception while creating backend string ", exp.what(),
                "\n");
          }
          TF_RETURN_IF_ERROR(BackendManager::CreateBackend(
              op_backend_name));  // Created a backend here. must free it
          // TranslateGraph must be called AFTER CreateBackend because some TF
          // ops like CNMS and gather use backend specific nodes
          TF_RETURN_IF_ERROR(Builder::TranslateGraph(
              input_shapes, static_input_map, &graph_for_current_encapsulate,
              ng_function));
          int json_indentation = 4;
          string serialized_ngfunc(
              ngraph::serialize(ng_function, json_indentation));
          std::unordered_map<std::string, std::string> additional_attribute_map;
          for (auto itr : node->attrs()) {
            // Find the optional attributes to be sent to the backend.
            // The optional attributes have '_ngraph_' appended to the start
            // so we need to get rid of that and only send the remaining
            // string
            // since the backend will only look for that.
            // '_ngraph_' is only appended for the bridge.
            // For e.g. _ngraph_ice_cores --> ice_cores
            if (itr.first.find("_ngraph_") != std::string::npos) {
              // leave out _ngraph_aot_requested
              if (itr.first.find("_ngraph_aot_requested") ==
                  std::string::npos) {
                additional_attribute_map.insert(
                    {itr.first.substr(strlen("_ngraph_")), itr.second.s()});
              }
            }
          }
          BackendManager::SetConfig(op_backend_name, additional_attribute_map);
          ng::runtime::Backend* op_backend = nullptr;
          try {
            op_backend = BackendManager::GetBackend(op_backend_name);
          } catch (const std::out_of_range& e) {
            NGRAPH_VLOG(5) << "Exception: " << e.what();
            BackendManager::ReleaseBackend(op_backend_name);
            throw;
          }
          BackendManager::LockBackend(op_backend_name);
          std::shared_ptr<ngraph::runtime::Executable> ng_exec;
          try {
            ng_exec = op_backend->compile(ng_function);
          } catch (...) {
            BackendManager::UnlockBackend(op_backend_name);
            Status st =
                NgraphSerialize("tf_function_error_aot.json", ng_function);
            BackendManager::ReleaseBackend(op_backend_name);
            return errors::Internal(
                "Failed to compile ng_function for AOT.",
                (st.ok() ? ""
                         : " Failed to serialize as well with error: " +
                               st.error_message()));
          }
          BackendManager::UnlockBackend(op_backend_name);
          BackendManager::ReleaseBackend(op_backend_name);

          stringstream exec_dump;
          ng_exec->save(exec_dump);
          // ng function attached as debugging information
          node->AddAttr("_ngraph_aot_ngfunction_" + signature,
                        serialized_ngfunc);
          // Compute will use this ngexec
          node->AddAttr("_ngraph_aot_ngexec_" + signature, exec_dump.str());
          // We do not need to add "_ngraph_aot_requested" attribute since it
          // already is already present in device_config and inserted into the
          // currently created NGraphEncapsulate
          // TODO: create a separate namespace of node attributes for backend
          // and for bridge
          performed_aot_on_enc.insert(node->name());
          NGRAPH_VLOG(5) << "Performed AOT on " << node->name();
        }
      }
    }  // end of for (ShapeHintMap single_hint : node_shapes_hints_sets)

    // In the end assert that all encapsulates have performed AOT
    for (auto node : graph->op_nodes()) {
      if (node->type_string() == "NGraphEncapsulate") {
        if (performed_aot_on_enc.find(node->name()) ==
            performed_aot_on_enc.end()) {
          return errors::Internal("Requested AOT, but did not perform AOT on ",
                                  node->name());
        }
      }
    }
  }  // end of if (aot_requested)

  // Pass 9 (optional, only run if environment variable
  // NGRAPH_TF_DUMP_CLUSTERS is set): validate the graph def, and
  // make sure we can construct a graph from it.
  if (std::getenv("NGRAPH_TF_DUMP_CLUSTERS")) {
    for (auto& kv : device_name_map) {
      int cluster_idx = kv.first;
      TF_RETURN_IF_ERROR(graph::ValidateGraphDef(
          *NGraphClusterManager::GetClusterGraph(cluster_idx),
          *OpRegistry::Global()));

      Graph g(OpRegistry::Global());
      GraphConstructorOptions opts;
      opts.allow_internal_ops = true;
      TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
          opts, *NGraphClusterManager::GetClusterGraph(cluster_idx), &g));

      std::stringstream ss;
      ss << "ngraph_cluster_" << cluster_idx;
      std::string filename_prefix = ss.str();

      GraphToPbTextFile(&g, filename_prefix + ".pbtxt");
      GraphToDotFile(&g, filename_prefix + ".dot",
                     "nGraph Cluster Dump: " + filename_prefix);
    }
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
