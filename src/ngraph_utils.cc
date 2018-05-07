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

using namespace std;

namespace ngraph_bridge {

bool GraphToPbTextFile(tf::Graph* graph, const string& filename) {
  tf::GraphDef g_def;
  graph->ToGraphDef(&g_def);

  string graph_pb_str;
  tf::protobuf::TextFormat::PrintToString(g_def, &graph_pb_str);
  std::ofstream ostrm_out(filename, std::ios_base::trunc);
  ostrm_out << graph_pb_str;
}

void SummarizeOp(tf::OpKernelConstruction* ctx, std::ostream& out) {
  auto node_def = ctx->def();
  out << "Node name: " << node_def.name() << " Op: " << node_def.op() << "\n";
  out << "Inputs: " << node_def.input().size() << "\n    ";
  for (const std::string& input : node_def.input()) {
    out << input << "\n    ";
  }
  out << "\n";
}

void GraphToDotFile(tf::Graph* graph, const std::string& filename,
                    const std::string& title, bool annotate_device) {
  std::string dot = GraphToDot(graph, title, annotate_device);
  std::ofstream ostrm_out(filename, std::ios_base::trunc);
  ostrm_out << dot;
}

std::string GraphToDot(tf::Graph* graph, const std::string& title,
                       bool annotate_device) {
  //
  // Output containing the DOT representation of the Graph
  std::ostringstream dot_string;
  dot_string << "digraph G {\n";
  dot_string << "labelloc=\"t\";\n";
  dot_string << "label=<<b>TensorFlow Graph: " << title << "</b><br/><br/>>;\n";

  // Input edges
  std::vector<const tf::Edge*> inputs;
  for (auto id = 0; id < graph->num_node_ids(); ++id) {
    const tf::Node* node = graph->FindNodeId(id);
    if (node == nullptr) continue;
    // Sample node:
    // node_name [label=<<b>convolution.1</b><br/>window={size=5x5
    //      pad=2_2x2_2}<br/>dim_labels=b01f_01io-&gt;b01f<br/>f32[10000,14,14,64]{3,2,1,0}>,
    //      shape=rect, tooltip=" ", style="filled", fontcolor="white",
    //      color="#003c8f", fillcolor="#1565c0"];
    dot_string << "node_" << node;
    dot_string << " [label=<<b>" << node->type_string() << "</b><br/>";
    dot_string << node->name() << "<br/>";
    // Print the data type if this node an op node
    tf::DataType datatype;
    if (GetNodeAttr(node->def(), "T", &datatype) == tf::Status::OK()) {
      dot_string << tf::DataTypeString(datatype) << "<br/>";
      if (annotate_device) {
        // For some reason the assigned_device_name results in a crash
        // dot_string << " Device: " << node->assigned_device_name() << "<br/>";
        dot_string << " Device: " << node->requested_device() << "<br/>";
      }
    }

    dot_string << ">, shape=rect, style=\"filled\", fontcolor=\"black\", "
                  "color=\"#003c8f\", fillcolor=\"#C3E4FD\"";
    dot_string << " ];\n";

    // printf("Node: %s Type: %s Id: %d\n", node->name().c_str(),
    //        node->type_string().c_str(), node->id());
    // printf("  %s\n", tf::SummarizeNode(*node).c_str());

    // Get the inputs for this Node.  We make sure control inputs are
    // after data inputs, as required by GraphDef.
    inputs.clear();
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
    for (size_t i = 0; i < inputs.size(); ++i) {
      const tf::Edge* edge = inputs[i];
      if (edge != nullptr) {
        const tf::Node* src = edge->src();
        if (!src->IsOp()) continue;
        const tf::Node* dst = edge->dst();
        if (!dst->IsOp()) continue;
        dot_string << "node_" << src << " -> "
                   << "node_" << dst << "\n";
        // printf("Edge: %s -> %s\n", src->name().c_str(), dst->name().c_str());
      }
    }
  }

  dot_string << "}\n";

  return dot_string.str();
}

}  // namespace ngraph_bridge