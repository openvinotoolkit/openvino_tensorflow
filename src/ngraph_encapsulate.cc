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
//
// !!!!!!!!! THIS SOURCE FILE IS OBSOLETE AND NOT USED !!!!!!!!!
//
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
    if (options.partition_graphs != nullptr) {
      for (auto &pg : *options.partition_graphs) {
        if (!ShouldEncapsulate(pg.first)) {
          continue;
        }

        tf::Graph *graph = pg.second.get();
        tf::FunctionDef fdef;

        TF_RETURN_IF_ERROR(EncapsulateFunction(pg.first, graph));
      }
    }

    return tf::Status::OK();
  }

private:
  // TODO(amprocte): do we need to look at job name, replica, task?
  bool ShouldEncapsulate(const std::string &device_string) {
    tf::DeviceNameUtils::ParsedName parsed;

    // TODO(amprocte): should this error out?
    if (!tf::DeviceNameUtils::ParseFullName(device_string, &parsed)) {
      return false;
    }

    // TODO(amprocte): change to DEVICE_NGRAPH constant
    return (parsed.has_type && parsed.type == "NGRAPH_CPU");
  }

  tf::Status EncapsulateFunction(const std::string& device, tf::Graph *graph) {
    tf::FunctionDef fdef;
    tf::OpDef &signature = *fdef.mutable_signature();

    std::stringstream ss;
    ss << "ngraph_encapsulate_" << function_counter++;
    string encapsulated_name = ss.str();

    signature.set_name(encapsulated_name);

    // Pass 1: Find send and receive nodes, and create corresponding arguments
    // in the function signature
    // TODO(amprocte): use something other than vector?
    std::vector<tf::Node *> recv_nodes;
    std::vector<tf::Node *> send_nodes;

    for (auto node : graph->op_nodes()) {
      if (node->IsRecv() &&
          std::find(recv_nodes.begin(), recv_nodes.end(), node) ==
              recv_nodes.end()) {
        recv_nodes.push_back(node);

        tf::OpDef::ArgDef& arg_def = *signature.add_input_arg();
        arg_def.set_name(node->name());
        arg_def.set_type(node->output_type(0));
      } else if (node->IsSend() &&
                 std::find(send_nodes.begin(), send_nodes.end(), node) ==
                     send_nodes.end()) {
        send_nodes.push_back(node);

        tf::OpDef::ArgDef& arg_def = *signature.add_output_arg();
        arg_def.set_name(node->name());
        arg_def.set_type(node->input_type(0));
      }
    }

    // Pass 2: Encapsulate all non-send/receive nodes, removing them from graph
    // and adding them to the fdef as we go.
    for (auto node : graph->op_nodes()) {
      if (!node->IsRecv() && !node->IsSend()) {
        // TODO(amprocte): this is the "original" node def per the docs. Is
        // there a way to "convert" the possibly-updated node to a node def
        // (and is this what we want to do?)
        // GraphToGraphDefSubrange in graph.cc has an example of how they do
        // it.
        *fdef.add_node_def() = node->def();
        graph->RemoveNode(node);
      }
    }

    // Add the function def to the graph's library.
    tf::FunctionDefLibrary flib_def;
    *flib_def.add_function() = fdef;
    TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(flib_def));

    // Create the nGraph launch op and add its inputs.
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
    }

    return tf::Status::OK();
  }

  int function_counter = 0;
};
} // namespace ngraph_bridge

//namespace tensorflow {
//REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 105,
//                      ngraph_bridge::NGraphEncapsulatePass);
//} // namespace tensorflow
