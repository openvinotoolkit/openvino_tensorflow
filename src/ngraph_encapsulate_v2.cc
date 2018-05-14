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
#include <fstream>
#include <iostream>
#include <sstream>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
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
  tf::Status EncapsulateFunctions(tf::Graph *graph) {
    // A map from cluster names to function definitions.
    std::map<std::string, tf::FunctionDef> fdef_map;

    // Pass 1: Create FunctionDefs for each named cluster.
    for (auto node : graph->op_nodes()) {
      std::string cluster_name;

      if (tf::GetNodeAttr(node->attrs(), "_ngraph_cluster", &cluster_name) !=
          tf::Status::OK()) {
        continue;
      }

      if (fdef_map.find(cluster_name) != fdef_map.end()) {
        continue;
      }

      fdef_map[cluster_name].mutable_signature()->set_name(
          "ngraph_encapsulate_" + cluster_name);
      VLOG(0) << "New fdef: " << fdef_map[cluster_name].signature().name();
    }

    // Pass 2: Find all nodes that are feeding into each cluster, and add inputs
    // for them.
    for (auto edge : graph->edges()) {
      if (edge->IsControlEdge()) {
        continue;
      }

      tf::Node *src = edge->src();
      tf::Node *dst = edge->dst();

      if (!src->IsOp() || !dst->IsOp()) {
        continue;
      }

      std::string dst_cluster_name;
      bool dst_clustered = true;
      if (tf::GetNodeAttr(dst->attrs(), "_ngraph_cluster", &dst_cluster_name) !=
          tf::Status::OK()) {
        dst_cluster_name = "<<unclustered>>";
        dst_clustered = false;
      }

      std::string src_cluster_name;
      bool src_clustered = true;
      if (tf::GetNodeAttr(src->attrs(), "_ngraph_cluster", &src_cluster_name) !=
          tf::Status::OK()) {
        src_cluster_name = "<<unclustered>>";
        src_clustered = false;
      }

      if (dst_cluster_name == src_cluster_name) {
        continue;
      }

      tf::DataType dt = dst->input_type(edge->dst_input());

      std::string flow_kind = dst_clustered && src_clustered ? "cross-flow" : dst_clustered ? "in-flow" : "out-flow";

      VLOG(0) << "found " << flow_kind << ": " << src->name() << "[" << edge->src_output() << "] in "
              << src_cluster_name << " to " << dst->name() << "[" << edge->dst_input() << "] in "
              << dst_cluster_name << ", datatype: " << dt;

      if (dst_clustered) {
        tf::FunctionDef& fdef = fdef_map[dst_cluster_name];
        tf::OpDef::ArgDef& input_arg_def = *fdef.mutable_signature()->add_input_arg();
        // TODO(amprocte): will need a fresh name here, "should" match regex [a-z][a-z0-9_]* per op_def.proto
        input_arg_def.set_name(src->name());
        input_arg_def.set_type(dt);
      }

      if (src_clustered) {
        tf::FunctionDef& fdef = fdef_map[src_cluster_name];
        tf::OpDef::ArgDef& output_arg_def = *fdef.mutable_signature()->add_output_arg();
        // TODO(amprocte): will need a fresh name here, "should" match regex [a-z][a-z0-9_]* per op_def.proto
        output_arg_def.set_name(dst->name());
        output_arg_def.set_type(dt);
      }
    }

    // Pass 3: Encapsulate all clustered nodes, removing them from graph
    // and adding them to the corresponding fdef as we go.
    for (auto node : graph->op_nodes()) {
      std::string cluster_name;

      if (tf::GetNodeAttr(node->attrs(), "_ngraph_cluster", &cluster_name) !=
          tf::Status::OK()) {
        continue;
      }

      // TODO(amprocte): this is the "original" node def per the docs. Is
      // there a way to "convert" the possibly-updated node to a node def
      // (and is this what we want to do?)
      // GraphToGraphDefSubrange in graph.cc has an example of how they do
      // it.
      // TODO(amprocte): there should definitely be an item in the map, but
      // maybe double-check/assert
      *fdef_map[cluster_name].add_node_def() = node->def();
      // graph->RemoveNode(node);
    }

    // Add the function defs to the graph's library.
    tf::FunctionDefLibrary flib_def;

    for (auto kv : fdef_map) {
      *flib_def.add_function() = kv.second;
    }

    TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(flib_def));

    /*    // Create the nGraph launch op and add its inputs.
        tf::NodeDef launch_node_def;
        launch_node_def.set_name(encapsulated_name);
        launch_node_def.set_op(encapsulated_name);
        for (auto node : recv_nodes) {
          launch_node_def.add_input(node->name());
        }
        launch_node_def.set_device(device);

        tf::Status status;
        tf::Node* launch_node;
        launch_node = graph->AddNode(launch_node_def, &status);
        TF_RETURN_IF_ERROR(status);

        // Add edges to/from the recv/sends.
        int pos;

        pos = 0;
        for (auto node : recv_nodes) {
          graph->AddEdge(node,0,launch_node,pos++);
        }

        pos = 0;
        for (auto node : send_nodes) {
          graph->AddEdge(launch_node,pos++,node,0);
        }*/

    return tf::Status::OK();
  }
};
} // namespace ngraph_bridge

namespace tensorflow {
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 110,
                      ngraph_bridge::NGraphEncapsulatePass);
} // namespace tensorflow
