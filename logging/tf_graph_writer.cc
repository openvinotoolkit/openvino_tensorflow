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
#include "tf_graph_writer.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"

using namespace std;
namespace ngraph_bridge {
const char* const DEVICE_NGRAPH = "NGRAPH";

//-----------------------------------------------------------------------------
// GraphToPbTextFile
//-----------------------------------------------------------------------------
void GraphToPbTextFile(tf::Graph* graph, const string& filename) {
  tf::GraphDef g_def;
  graph->ToGraphDef(&g_def);

  string graph_pb_str;
  tf::protobuf::TextFormat::PrintToString(g_def, &graph_pb_str);
  std::ofstream ostrm_out(filename, std::ios_base::trunc);
  ostrm_out << graph_pb_str;
}

//-----------------------------------------------------------------------------
// GraphToDotFile
//-----------------------------------------------------------------------------
void GraphToDotFile(tf::Graph* graph, const std::string& filename,
                    const std::string& title, bool annotate_device) {
  std::string dot = GraphToDot(graph, title, annotate_device);
  std::ofstream ostrm_out(filename, std::ios_base::trunc);
  ostrm_out << dot;
}

static std::string color_string(unsigned int color) {
  std::stringstream ss;
  ss << "#" << std::setfill('0') << std::setw(6) << std::hex
     << (color & 0xFFFFFF);
  return ss.str();
}

//-----------------------------------------------------------------------------
// GraphToDot
//-----------------------------------------------------------------------------
std::string GraphToDot(tf::Graph* graph, const std::string& title,
                       bool annotate_device) {
  //
  // Output containing the DOT representation of the Graph
  std::ostringstream dot_string;
  dot_string << "digraph G {\n";
  dot_string << "labelloc=\"t\";\n";
  dot_string << "label=<<b>TensorFlow Graph: " << title << "</b><br/><br/>>;\n";

  const int num_cluster_colors = 22;
  unsigned int cluster_bg_colors[num_cluster_colors]{
      0xF3C300, 0x875692, 0xF38400, 0xA1CAF1, 0xBE0032, 0xC2B280,
      0x848482, 0x008856, 0xE68FAC, 0x0067A5, 0xF99379, 0x604E97,
      0xF6A600, 0xB3446C, 0xDCD300, 0x882D17, 0x8DB600, 0x654522,
      0xE25822, 0x2B3D26, 0xF2F3F4, 0x222222};

  // https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
  auto make_fg_color = [](unsigned int bg_color) {
    unsigned int red = (bg_color & 0xFF0000) >> 16;
    unsigned int green = (bg_color & 0x00FF00) >> 8;
    unsigned int blue = (bg_color & 0x0000FF);

    if (red * 0.299 + green * 0.587 + blue * 0.114 > 186) {
      return 0x000000;
    } else {
      return 0xFFFFFF;
    }
  };

  int seen_cluster_count = 0;
  std::map<int, unsigned int> cluster_color_map;

  // Input edges
  std::vector<const tf::Edge*> inputs;
  for (auto id = 0; id < graph->num_node_ids(); ++id) {
    string fill_color = "#f2f2f2";
    string style = "filled";

    const tf::Node* node = graph->FindNodeId(id);
    if (node == nullptr) continue;
    if (node->IsSource()) continue;
    if (node->IsSink()) continue;
    // Sample node:
    // node_name [label=<<b>convolution.1</b><br/>window={size=5x5
    //      pad=2_2x2_2}<br/>dim_labels=b01f_01io-&gt;b01f<br/>f32[10000,14,14,64]{3,2,1,0}>,
    //      shape=rect, tooltip=" ", style="filled", fontcolor="white",
    //      color="#003c8f", fillcolor="#1565c0"];
    //
    dot_string << "node_" << node;
    dot_string << " [label=<<b>" << node->type_string() << "</b><br/>";
    dot_string << node->name() << "<br/>";

    // Print the data type if this node an op node
    tf::DataType datatype;
    if (GetNodeAttr(node->def(), "T", &datatype) == tf::Status::OK()) {
      dot_string << tf::DataTypeString(datatype) << "<br/>";
    }

    string text_color = "#000000";

    if (annotate_device) {
      // For some reason the assigned_device_name results in a crash
      // dot_string << " Device: " << node->assigned_device_name() << "<br/>";
      auto device_name = node->requested_device();
      if (tf::str_util::StrContains(device_name, DEVICE_NGRAPH)) {
        fill_color = "#cee9fd";
      } else if (tf::str_util::StrContains(device_name, "XLA_CPU")) {
        fill_color = "#ffcc99";
      } else if (tf::str_util::StrContains(device_name, "CPU")) {
        fill_color = "#ffffcc";
      }
      dot_string << " Device: " << device_name << "<br/>";
    } else {
      if (node->type_string() == "_Recv" ||
          node->type_string() == "_HostRecv") {
        fill_color = "#f29999";
        style = "rounded,filled";
      } else if (node->type_string() == "_Send" ||
                 node->type_string() == "_HostSend") {
        fill_color = "#99f299";
        style = "rounded,filled";
      } else {
        int cluster_idx;
        if (tf::GetNodeAttr(node->attrs(), "_ngraph_cluster", &cluster_idx) ==
            tf::Status::OK()) {
          unsigned int bg_color;
          unsigned int fg_color;
          if (cluster_color_map.find(cluster_idx) == cluster_color_map.end()) {
            bg_color = cluster_bg_colors[seen_cluster_count];
            cluster_color_map[cluster_idx] = bg_color;
            if (seen_cluster_count < num_cluster_colors - 1) {
              seen_cluster_count++;
            }
          } else {
            bg_color = cluster_color_map[cluster_idx];
          }
          fg_color = make_fg_color(bg_color);

          fill_color = color_string(bg_color);
          text_color = color_string(fg_color);
        } else {
          style = "dashed,filled";
        }
      }
    }

    dot_string << ">, shape=rect, style=\"" << style << "\", fontcolor=\""
               << text_color
               << "\", "
                  "color=\"black\", fillcolor=\""
               << fill_color << "\"";
    dot_string << " ];\n";

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

        string arrow_color = "#000000";

        if (edge->IsControlEdge()) {
          arrow_color = "#ff0000";
        }

        dot_string << "node_" << src << " -> "
                   << "node_" << dst << " [color=\"" << arrow_color << "\"]\n";
        // printf("Edge: %s -> %s\n", src->name().c_str(), dst->name().c_str());
      }
    }
  }

  dot_string << "}\n";

  return dot_string.str();
}
}  // namespace ngraph_bridge
