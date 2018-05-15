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
#include "ngraph_utils.h"

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
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/device_name_utils.h"

using namespace std;
namespace ngraph_bridge {

class NGraphEncapsulatePass : public tensorflow::GraphOptimizationPass {
public:
  tf::Status Run(const tf::GraphOptimizationPassOptions &options) {
    return EncapsulateFunctions(options.graph->get());
  }

private:
  static std::string Mangle(std::string name) {
    std::stringstream ss;

    for (char c : name) {
      if (!std::isalpha(c) && !std::isdigit(c)) {
        ss << "_" << std::setw(2) << std::setfill('0') << int(c);
      } else {
        ss << c;
      }
    }

    return ss.str();
  }

  static std::string MakeCrossFlowName(std::string src_node_name,
                                       int src_output) {
    std::stringstream ss;

    ss << "ngraph_crossflow_" << Mangle(src_node_name) << "_" << src_output;
    return ss.str();
  }

  static bool GetClusterId(const tf::Node *node, int *cluster_id) {
    if (tf::GetNodeAttr(node->attrs(), "_ngraph_cluster", cluster_id) !=
        tf::Status::OK()) {
      *cluster_id = -1;
      return false;
    } else {
      return true;
    }
  }

  tf::Status EncapsulateFunctions(tf::Graph *graph) {
    // A map from cluster indices to function definitions.
    std::map<int, tf::FunctionDef> fdef_map;

    // As we build the graph we will be tracking the.. TODO(amprocte): finish
    // this comment.
    std::map<std::tuple<int, int>, std::tuple<int, int>> output_remap_map;
    std::map<std::tuple<int, int, int>, int> input_remap_map;

    // A map from cluster indices to a vector of input data types.
    std::map<int, std::vector<std::tuple<int, int, tf::DataType>>>
        cluster_input_map;
    // A map from cluster indices to a vector of output data types.
    std::map<int, std::vector<tf::DataType>> cluster_output_dt_map;

    // A map from cluster indices to corresponding NGraphEncapsulate nodes.
    std::map<int, tf::Node *> cluster_node_map;

    // Pass 1: Create FunctionDefs for each existing cluster.
    for (auto node : graph->op_nodes()) {
      int cluster_idx;

      if (!GetClusterId(node, &cluster_idx)) {
        continue;
      }

      if (fdef_map.find(cluster_idx) != fdef_map.end()) {
        continue;
      }

      std::stringstream ss;
      ss << "_NGraphCluster" << cluster_idx;

      fdef_map[cluster_idx].mutable_signature()->set_name(ss.str());
    }

    // Pass 2: Find all nodes that are feeding into/out of each cluster, and
    // add inputs for them to the corresponding FunctionDef(s).
    for (auto edge : graph->edges()) {
      // TODO(amprocte): should actually keep of these. During clustering we
      // will already have identified any intra-cluster control deps. Should
      // maintain inter-cluster control deps.
      if (edge->IsControlEdge()) {
        continue;
      }

      tf::Node *src = edge->src();
      tf::Node *dst = edge->dst();

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

      VLOG(0) << "found " << flow_kind << ": " << src->name() << "["
              << edge->src_output() << "] in " << src_cluster_idx << " to "
              << dst->name() << "[" << edge->dst_input() << "] in "
              << dst_cluster_idx << ", datatype: " << dt;

      std::string cross_flow_name =
          MakeCrossFlowName(src->name(), edge->src_output());

      // If the source node lies within a cluster, we must create an output for
      // it from the source cluster. For the moment we will just store this
      // fact in the output_remap_map.
      if (src_clustered &&
          output_remap_map.find(std::make_tuple(
              src->id(), edge->src_output())) == output_remap_map.end()) {
        output_remap_map[std::make_tuple(src->id(), edge->src_output())] =
            std::make_tuple(src_cluster_idx,
                            cluster_output_dt_map[src_cluster_idx].size());
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
        cluster_input_map[dst_cluster_idx].push_back(
            std::make_tuple(src->id(), edge->src_output(), dt));
      }
    }

    // Pass 3: Create the function library and add in all the function defs.
    tf::FunctionDefLibrary flib_def_for_encaps;

    // Add a stub definition for the "_NGraphEncapsulate" op itself.
    auto fdef_encaps = flib_def_for_encaps.add_function();
    fdef_encaps->mutable_signature()->set_name("NGraphEncapsulate");

    auto attr_ngraph_cluster = fdef_encaps->mutable_signature()->add_attr();
    attr_ngraph_cluster->set_name("ngraph_cluster");
    attr_ngraph_cluster->set_type("int");
    attr_ngraph_cluster->set_description(
        "Index of the nGraph cluster that is being encapsulated");

    auto attr_targuments = fdef_encaps->mutable_signature()->add_attr();
    attr_targuments->set_name("Targuments");
    attr_targuments->set_type("list(type)");
    attr_targuments->set_description("List of types for each argument");

    auto attr_tresults = fdef_encaps->mutable_signature()->add_attr();
    attr_tresults->set_name("Tresults");
    attr_tresults->set_type("list(type)");
    attr_tresults->set_description("List of types for each result");

    tf::OpDef::ArgDef &input_arg_def =
        *(fdef_encaps->mutable_signature()->add_input_arg());
    input_arg_def.set_name("arguments");
    input_arg_def.set_type_list_attr("Targuments");

    tf::OpDef::ArgDef &output_arg_def =
        *(fdef_encaps->mutable_signature()->add_output_arg());
    output_arg_def.set_name("results");
    output_arg_def.set_type_list_attr("Tresults");

    TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(flib_def_for_encaps));

    // Pass 4: Create encapsulation nodes for all clusters.
    for (auto &kv : fdef_map) {
      int cluster_idx = kv.first;

      std::stringstream ss;
      ss << "ngraph_cluster_" << cluster_idx;

      std::vector<tf::DataType> input_types;
      std::vector<tf::NodeBuilder::NodeOut> inputs;

      for (auto &tup : cluster_input_map[cluster_idx]) {
        int src_node_id;
        int src_output_idx;
        tf::DataType dt;
        std::tie(src_node_id, src_output_idx, dt) = tup;

        input_types.push_back(dt);

        inputs.push_back(tf::NodeBuilder::NodeOut(
            graph->FindNodeId(src_node_id), src_output_idx));
      }

      tf::Node *n;
      tf::Status status =
          tf::NodeBuilder(ss.str(), &fdef_encaps->signature())
              .Attr("ngraph_cluster", cluster_idx)
              .Attr("Targuments", input_types)
              .Attr("Tresults", cluster_output_dt_map[cluster_idx])
              .Input(inputs)
              .Finalize(graph, &n);
      TF_RETURN_IF_ERROR(status);
      cluster_node_map[cluster_idx] = n;
    }

    // Pass 5: Remap all inputs that are reading from encapsulated nodes, and
    // control edges that involve clustered nodes.
    for (auto edge : graph->edges()) {
      if (edge->IsControlEdge()) {
        int src_cluster_idx;
        bool src_clustered = GetClusterId(edge->src(), &src_cluster_idx);
        int dst_cluster_idx;
        bool dst_clustered = GetClusterId(edge->dst(), &dst_cluster_idx);

        if (src_clustered && dst_clustered) {
          if (src_cluster_idx != dst_cluster_idx) {
            graph->RemoveControlEdge(edge);
            graph->AddControlEdge(cluster_node_map[src_cluster_idx],
                                  cluster_node_map[dst_cluster_idx]);
          }
        } else if (src_clustered) {
          tf::Node *dst = edge->dst();
          graph->RemoveControlEdge(edge);
          graph->AddControlEdge(cluster_node_map[src_cluster_idx], dst);
        } else if (dst_clustered) {
          tf::Node *src = edge->src();
          graph->RemoveControlEdge(edge);
          graph->AddControlEdge(src, cluster_node_map[dst_cluster_idx]);
        }
      } else {
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

    // Pass 6: Encapsulate all clustered nodes, removing them from graph
    // and adding them to the corresponding fdef as we go.
    for (auto node : graph->op_nodes()) {
      int cluster_idx;

      if (tf::GetNodeAttr(node->attrs(), "_ngraph_cluster", &cluster_idx) !=
          tf::Status::OK()) {
        continue;
      }

      // TODO(amprocte): this is the "original" node def per the docs. Is
      // there a way to "convert" the possibly-updated node to a node def
      // (and is this what we want to do?)
      // GraphToGraphDefSubrange in graph.cc has an example of how they do
      // it.
      *fdef_map[cluster_idx].add_node_def() = node->def();
      graph->RemoveNode(node);
    }

    // Pass 7: Create and add the function library.
    tf::FunctionDefLibrary flib_def;

    // Add definitions for each cluster function.
    for (auto kv : fdef_map) {
      auto &fdef = kv.second;
      TF_RETURN_IF_ERROR(ValidateOpDef(fdef.signature()));
      *flib_def.add_function() = fdef;
    }

    TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(flib_def));

    return tf::Status::OK();
  }
};
} // namespace ngraph_bridge

namespace tensorflow {
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 110,
                      ngraph_bridge::NGraphEncapsulatePass);
} // namespace tensorflow
