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

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "ngraph_bridge/enable_variable_ops/ngraph_replace_op_utilities.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_prefetch_shared_data.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// Utility function to check if placement on the NGRAPH device has been
// requested.
//
// FIXME(amprocte): stubbed out for now because NGRAPH device is gone.
//
static bool NGraphPlacementRequested(const Node* node) { return true; }

Status ReplacePrefetch(Graph* graph, Node* prefetch_node) {
  NodeBuilder::NodeOut input_dataset;
  NodeBuilder::NodeOut buffer_size;

  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(prefetch_node->input_edges(&input_edges));

  input_dataset =
      NodeBuilder::NodeOut(input_edges[0]->src(), input_edges[0]->src_output());
  buffer_size =
      NodeBuilder::NodeOut(input_edges[1]->src(), input_edges[1]->src_output());

  std::vector<DataType> output_types;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(prefetch_node->attrs(), "output_types", &output_types));

  std::vector<PartialTensorShape> output_shapes;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(prefetch_node->attrs(), "output_shapes", &output_shapes));

  int slack_period = 0;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(prefetch_node->attrs(), "slack_period", &slack_period));

  Node* replacement;
  TF_RETURN_IF_ERROR(NodeBuilder("NGraphPrefetchNode", "NGraphPrefetchDataset")
                         .Input(input_dataset)
                         .Input(buffer_size)
                         .Attr("output_types", output_types)
                         .Attr("output_shapes", output_shapes)
                         .Attr("slack_period", slack_period)
                         .Device(prefetch_node->assigned_device_name())
                         .Finalize(graph, &replacement));
  replacement->set_assigned_device_name(prefetch_node->assigned_device_name());

  string new_name = graph->NewName("NGraph" + prefetch_node->name());
  replacement->set_name(new_name);

  std::vector<const Edge*> edges;

  // Remove all the input edges of the existing prefetch node
  std::vector<const Edge*> edges_to_remove;
  std::vector<std::tuple<Node*, int, Node*, int>> edges_to_add;
  for (auto edge : prefetch_node->in_edges()) {
    edges_to_remove.push_back(edge);
  }

  for (auto edge : prefetch_node->out_edges()) {
    NGRAPH_VLOG(0) << "Replacing: OutEdge " << edge->DebugString();
    // Collect new output edges between the new prefetch node and the next node
    edges_to_add.push_back(std::tuple<Node*, int, Node*, int>(
        replacement, edge->src_output(), edge->dst(), edge->dst_input()));
    // Remove the output edge from the current prefetch node
    edges_to_remove.push_back(edge);
  }

  // Now add the new output edges
  // The input edges to the new node is added during the node creation
  for (const auto& i : edges_to_add) {
    NGRAPH_VLOG(0) << "Adding: " << get<0>(i)->name() << "  " << get<1>(i)
                   << "  " << get<2>(i)->name() << " " << get<3>(i);
    graph->AddEdge(get<0>(i), get<1>(i), get<2>(i), get<3>(i));
  }

  // Though edges will be removed when we remove the prefetch_node
  // we specifically remove the edges to be sure
  for (auto edge : edges_to_remove) {
    NGRAPH_VLOG(0) << "Removing: " << edge->DebugString();
    graph->RemoveEdge(edge);
  }

  // FInally remove the current preftetch node
  graph->RemoveNode(prefetch_node);
  NGRAPH_VLOG(4) << "Replaced TF Prefetch Node " << prefetch_node->name()
                 << " with NGraphPrefetch Node " << replacement->name();
  return Status::OK();
}

//
// Main entry point for the variable-capture.
//
Status CaptureVariables(Graph* graph, std::set<string> skip_these_nodes) {
  const static std::map<
      const string,
      const pair<string,
                 function<Status(
                     Graph * graph, Node * node, Node * *replacement,
                     const string replacement_node_name,
                     const string replacement_op_type, const bool just_looking,
                     const bool outputs_ng_supported, const int graph_id,
                     const bool is_backend_set)>>>
      CAPTURE_REPLACE_OP_MAP{
          {"ApplyGradientDescent",
           std::make_pair("NGraphApplyGradientDescent", ReplaceOptimizer)},
          {"ApplyMomentum",
           std::make_pair("NGraphApplyMomentum", ReplaceOptimizer)},

          {"Assign", std::make_pair("NGraphAssign", ReplaceAssign)},
          {"AssignAdd", std::make_pair("NGraphAssignAdd", ReplaceAssign)},
          {"AssignSub", std::make_pair("NGraphAssignSub", ReplaceAssign)},
          {"VariableV2", std::make_pair("NGraphVariable", ReplaceVariable)}};

  std::set<Node*> nodes_to_capture;
  Node* prefetch_node = nullptr;
  for (auto node : graph->op_nodes()) {
    std::set<Node*> ref_list;
    if (NGraphPlacementRequested(node)) {
      // Check if the node is a VariableV2
      if (node->type_string() == "VariableV2") {
        NGRAPH_VLOG(4) << "Found Variable: " << node->name();
        // Add the Variable node to the ref list
        ref_list.insert(node);

        // go over all the nodes leading from VariableV2 and store them
        // in a list if they are ref type
        StoreRefTypeOutputs(node, &ref_list);

        if (ref_list.size()) {
          for (auto n : ref_list) {
            auto itr = CAPTURE_REPLACE_OP_MAP.find(n->type_string());
            if (itr != CAPTURE_REPLACE_OP_MAP.end()) {
              nodes_to_capture.insert(n);
            }
          }
          ref_list.clear();
        }
      } else if (node->type_string() == "PrefetchDataset") {
        // Collect the prefetch_node so that we can add
        // the NGraphWriteToDevice Op after this one
        prefetch_node = node;
      }
    }
  }

  for (auto node : nodes_to_capture) {
    Node* replacement;
    auto itr = CAPTURE_REPLACE_OP_MAP.find(node->type_string());
    // Create the replacement node
    TF_RETURN_IF_ERROR((itr->second.second)(graph, node, &replacement,
                                            node->name(), itr->second.first,
                                            true, false, 0, false));
    NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                   << replacement->DebugString();
    TF_RETURN_IF_ERROR(ReplaceInputControlEdges(graph, node, replacement));
    TF_RETURN_IF_ERROR(ReplaceOutputEdges(graph, node, replacement));
  }  // end of looping through nodes in the capture list

  for (auto node : nodes_to_capture) {
    NGRAPH_VLOG(4) << "Removing: " << node->name();
    graph->RemoveNode(node);
  }

  if (std::getenv(NGraphPrefetchSharedResouce::NGRAPH_TF_USE_PREFETCH) !=
      nullptr) {
    if (prefetch_node != nullptr) {
      return ReplacePrefetch(graph, prefetch_node);
    }
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
