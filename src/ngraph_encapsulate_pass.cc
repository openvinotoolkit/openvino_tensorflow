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
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
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

#include "ngraph_cluster_manager.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"
#include "tf_graph_writer.h"

using namespace std;
namespace ngraph_bridge {

class NGraphEncapsulatePass : public tensorflow::GraphOptimizationPass {
 public:
  tf::Status Run(const tf::GraphOptimizationPassOptions& options) {
    if (std::getenv("NGRAPH_TF_SKIP_ENCAPSULATION") != nullptr) {
      NGRAPH_VLOG(0)
          << "NGRAPH_TF_SKIP_ENCAPSULATION is set. Skipping encapsulation "
             "step.";
      return tf::Status::OK();
    }

    return EncapsulateFunctions(options.graph->get());
  }

 private:
  // TODO(amprocte): integrate this into the input names
  // static std::string Mangle(std::string name) {
  //   std::stringstream ss;
  //
  //   for (char c : name) {
  //     if (!std::isalpha(c) && !std::isdigit(c)) {
  //       ss << "_" << std::setw(2) << std::setfill('0') << int(c);
  //     } else {
  //       ss << c;
  //     }
  //   }
  //
  //   return ss.str();
  // }

  static bool GetClusterId(const tf::Node* node, int* cluster_id) {
    if (tf::GetNodeAttr(node->attrs(), "_ngraph_cluster", cluster_id) !=
        tf::Status::OK()) {
      *cluster_id = -1;
      return false;
    } else {
      return true;
    }
  }

  // begin code copied and pasted (and modified) from graph.cc...
  static void AddInput(tf::NodeDef* dst, tf::StringPiece src_name,
                       int src_slot) {
    if (src_slot == tf::Graph::kControlSlot) {
      dst->add_input(tf::strings::StrCat("^", src_name));
    } else if (src_slot == 0) {
      dst->add_input(src_name.data(), src_name.size());
    } else {
      dst->add_input(tf::strings::StrCat(src_name, ":", src_slot));
    }
  }
  // ...end code copied and pasted (and modified) from graph.cc

  tf::Status EncapsulateFunctions(tf::Graph* graph) {
    // A map from cluster indices to the expected device name for nodes
    // in that cluster.
    std::map<int, std::string> device_name_map;

    // As we build the graph we will be tracking the.. TODO(amprocte): finish
    // this comment.
    std::map<std::tuple<int, int>, std::tuple<int, int>> output_remap_map;
    std::map<std::tuple<int, int, int>, int> input_remap_map;
    std::map<std::tuple<int, std::string, int>, string> input_rename_map;

    // A map from cluster indices to a vector of input data types.
    std::map<int, std::vector<std::tuple<int, int, tf::DataType>>>
        cluster_input_map;
    // A map from cluster indices to a vector of output data types.
    std::map<int, std::vector<tf::DataType>> cluster_output_dt_map;

    // A map from cluster indices to corresponding NGraphEncapsulate nodes.
    std::map<int, tf::Node*> cluster_node_map;

    // Pass 1: Populate the cluster-index-to-device name map for each existing
    // cluster.
    for (auto node : graph->op_nodes()) {
      int cluster_idx;

      if (!GetClusterId(node, &cluster_idx)) {
        continue;
      }

      auto it = device_name_map.find(cluster_idx);

      if (it != device_name_map.end()) {
        if (it->second != node->requested_device()) {
          std::stringstream ss_err;
          ss_err << "Node " << node->name() << " in cluster " << cluster_idx
                 << " has requested device " << node->requested_device()
                 << " but another node with requested device " << it->second
                 << " has already been seen in the same cluster";

          return tf::errors::Internal(ss_err.str());
        }
      } else {
        NGRAPH_VLOG(3) << "setting cluster " << cluster_idx
                       << " requested device to '" << node->requested_device()
                       << "'";
        device_name_map[cluster_idx] = node->requested_device();
      }
    }

    // Pass 2: Find all nodes that are feeding into/out of each cluster, and
    // add inputs for them to the corresponding FunctionDef(s).
    std::map<int, int> retval_index_count;
    std::map<int, int> arg_index_count;

    for (auto edge : graph->edges()) {
      // TODO(amprocte): should actually keep of these. During clustering we
      // will already have identified any intra-cluster control deps. Should
      // maintain inter-cluster control deps.
      if (edge->IsControlEdge()) {
        continue;
      }

      tf::Node* src = edge->src();
      tf::Node* dst = edge->dst();

      // TODO(amprocte): the following rejects edges involving source/sink. Is
      // that what we want to do?
      if (!src->IsOp() || !dst->IsOp()) {
        continue;
      }

      int dst_cluster_idx;
      bool dst_clustered = GetClusterId(dst, &dst_cluster_idx);

      int src_cluster_idx;
      bool src_clustered = GetClusterId(src, &src_cluster_idx);

      // Ignore edges within a cluster. (Note that this test also works when
      // both nodes are unclustered; GetClusterId gives us -1 in that case.
      if (dst_cluster_idx == src_cluster_idx) {
        continue;
      }

      // Some debug logging...
      tf::DataType dt = dst->input_type(edge->dst_input());
      std::string flow_kind = dst_clustered && src_clustered
                                  ? "cross-flow"
                                  : dst_clustered ? "in-flow" : "out-flow";

      NGRAPH_VLOG(4) << "found " << flow_kind << ": " << src->name() << "["
                     << edge->src_output() << "] in " << src_cluster_idx
                     << " to " << dst->name() << "[" << edge->dst_input()
                     << "] in " << dst_cluster_idx << ", datatype: " << dt;

      // If the source node lies within a cluster, we must create an output for
      // it from the source cluster. For the moment we will just store this
      // fact in the output_remap_map.
      if (src_clustered &&
          output_remap_map.find(std::make_tuple(
              src->id(), edge->src_output())) == output_remap_map.end()) {
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
          input_remap_map.find(std::make_tuple(dst_cluster_idx, src->id(),
                                               edge->src_output())) ==
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

        SetAttrValue(dt, &((*(new_input_node_def->mutable_attr()))["T"]));
        SetAttrValue(arg_index_count[dst_cluster_idx],
                     &((*(new_input_node_def->mutable_attr()))["index"]));

        arg_index_count[dst_cluster_idx]++;

        cluster_input_map[dst_cluster_idx].push_back(
            std::make_tuple(src->id(), edge->src_output(), dt));
      }
    }

    // Pass 3: Create encapsulation nodes for all clusters.
    for (auto& kv : device_name_map) {
      int cluster_idx = kv.first;

      std::stringstream ss;
      ss << "ngraph_cluster_" << cluster_idx;

      std::vector<tf::DataType> input_types;
      std::vector<tf::NodeBuilder::NodeOut> inputs;

      for (auto& tup : cluster_input_map[cluster_idx]) {
        int src_node_id;
        int src_output_idx;
        tf::DataType dt;
        std::tie(src_node_id, src_output_idx, dt) = tup;

        input_types.push_back(dt);

        inputs.push_back(tf::NodeBuilder::NodeOut(
            graph->FindNodeId(src_node_id), src_output_idx));
      }

      tf::Node* n;
      tf::Status status =
          tf::NodeBuilder(ss.str(), "NGraphEncapsulate")
              .Attr("ngraph_cluster", cluster_idx)
              .Attr("Targuments", input_types)
              .Attr("Tresults", cluster_output_dt_map[cluster_idx])
              .Device(device_name_map[cluster_idx])
              .Input(inputs)
              .Finalize(graph, &n);
      TF_RETURN_IF_ERROR(status);
      n->set_assigned_device_name(device_name_map[cluster_idx]);

      cluster_node_map[cluster_idx] = n;
    }

    // Pass 4: Remap all non-clustered inputs that are reading from
    // encapsulated edges, and all control edges that cross cluster
    // boundaries.
    for (auto edge : graph->edges()) {
      int src_cluster_idx;
      bool src_clustered = GetClusterId(edge->src(), &src_cluster_idx);
      int dst_cluster_idx;
      bool dst_clustered = GetClusterId(edge->dst(), &dst_cluster_idx);

      if (src_cluster_idx == dst_cluster_idx) {
        continue;
      }

      if (edge->IsControlEdge()) {
        if (src_clustered && dst_clustered) {
          graph->RemoveControlEdge(edge);
          graph->AddControlEdge(cluster_node_map[src_cluster_idx],
                                cluster_node_map[dst_cluster_idx]);
        } else if (src_clustered) {
          tf::Node* dst = edge->dst();
          graph->RemoveControlEdge(edge);
          graph->AddControlEdge(cluster_node_map[src_cluster_idx], dst);
        } else if (dst_clustered) {
          tf::Node* src = edge->src();
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

        graph->UpdateEdge(cluster_node_map[cluster_idx], cluster_output,
                          edge->dst(), edge->dst_input());
      }
    }

    // Pass 5: Make copies of all clustered nodes inside the cluster graphs,
    // rewiring the inputs in their NodeDefs as we go.
    for (auto node : graph->op_nodes()) {
      int cluster_idx;

      if (tf::GetNodeAttr(node->attrs(), "_ngraph_cluster", &cluster_idx) !=
          tf::Status::OK()) {
        continue;
      }

      // Because the input names may have changed from the original node def,
      // we will need to borrow some code from Graph::ToGraphDefSubRange in
      // tensorflow/core/graph/graph.cc that rewrites the node's input list.

      // begin code copied and pasted (and modified) from graph.cc...
      tf::NodeDef original_def = node->def();

      // Get the inputs for this Node.  We make sure control inputs are
      // after data inputs, as required by GraphDef.
      std::vector<const tf::Edge*> inputs;
      inputs.resize(node->num_inputs(), nullptr);
      for (const tf::Edge* edge : node->in_edges()) {
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
        const tf::Edge* edge = inputs[i];
        if (edge == nullptr) {
          if (i < node->requested_inputs().size()) {
            original_def.add_input(node->requested_inputs()[i]);
          } else {
            original_def.add_input("");
          }
        } else {
          const tf::Node* src = edge->src();
          if (!src->IsOp()) continue;
          AddInput(&original_def, src->name(), edge->src_output());
        }
      }
      // ...end code copied and pasted (and modified) from graph.cc

      auto node_def =
          NGraphClusterManager::GetClusterGraph(cluster_idx)->add_node();
      *node_def = original_def;

      for (auto& input : *(node_def->mutable_input())) {
        tf::TensorId tensor_id = tf::ParseTensorName(input);

        auto it = input_rename_map.find(std::make_tuple(
            cluster_idx, tensor_id.first.ToString(), tensor_id.second));

        if (it != input_rename_map.end()) {
          input = it->second;
        }
      }
    }

    // Pass 6: Remove clustered nodes from the graph.
    for (auto node : graph->op_nodes()) {
      int cluster_idx;

      if (tf::GetNodeAttr(node->attrs(), "_ngraph_cluster", &cluster_idx) !=
          tf::Status::OK()) {
        continue;
      }

      graph->RemoveNode(node);
    }

    // Pass 7 (optional, only run if environment variable
    // NGRAPH_TF_VALIDATE_CLUSTER_GRAPHS is set): validate the graph def, and
    // make sure we can construct a graph from it.
    if (std::getenv("NGRAPH_TF_VALIDATE_CLUSTER_GRAPHS")) {
      for (auto& kv : device_name_map) {
        int cluster_idx = kv.first;
        TF_RETURN_IF_ERROR(tf::graph::ValidateGraphDef(
            *NGraphClusterManager::GetClusterGraph(cluster_idx),
            *tf::OpRegistry::Global()));

        tf::Graph g(tf::OpRegistry::Global());
        tf::GraphConstructorOptions opts;
        opts.allow_internal_ops = true;
        TF_RETURN_IF_ERROR(tf::ConvertGraphDefToGraph(
            opts, *NGraphClusterManager::GetClusterGraph(cluster_idx), &g));

        std::stringstream ss;
        ss << "ngraph_cluster_" << cluster_idx;
        std::string filename_prefix = ss.str();

        GraphToPbTextFile(&g, filename_prefix + ".pbtxt");
        GraphToDotFile(&g, filename_prefix + ".dot",
                       "nGraph Cluster Dump: " + filename_prefix, false);
      }
    }

    return tf::Status::OK();
  }
};
}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 110,
                      ngraph_bridge::NGraphEncapsulatePass);
}  // namespace tensorflow
