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

//
// Main entry point for the variable-capture.
//
Status CaptureVariables(Graph* graph, std::set<string> skip_these_nodes) {
  const static std::map<
      const string,
      const pair<
          string,
          function<Status(
              Graph * graph, Node * node, Node * *replacement,
              const string replacement_node_name,
              const string replacement_op_type, const bool just_looking,
              const bool is_tf_just_looking, const bool outputs_ng_supported,
              const int graph_id, const bool is_backend_set)>>>
      CAPTURE_REPLACE_OP_MAP{
          {"ApplyGradientDescent", std::make_pair("NGraphApplyGradientDescent",
                                                  ReplaceApplyGradientDescent)},
          {"Assign", std::make_pair("NGraphAssign", ReplaceAssign)},
          {"AssignAdd", std::make_pair("NGraphAssignAdd", ReplaceAssign)},
          {"AssignSub", std::make_pair("NGraphAssignSub", ReplaceAssign)},
          {"VariableV2", std::make_pair("NGraphVariable", ReplaceVariable)}};

  std::vector<Node*> nodes_to_capture;

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
              nodes_to_capture.push_back(n);
            }
          }
          ref_list.clear();
        }
      }
    }
  }

  for (auto node : nodes_to_capture) {
    Node* replacement;
    auto itr = CAPTURE_REPLACE_OP_MAP.find(node->type_string());
    // Create the replacement node
    TF_RETURN_IF_ERROR((itr->second.second)(graph, node, &replacement,
                                            node->name(), itr->second.first,
                                            true, false, false, 0, false));
    NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                   << replacement->DebugString();

    TF_RETURN_IF_ERROR(ReplaceInputControlEdges(graph, node, replacement));
    TF_RETURN_IF_ERROR(ReplaceOutputEdges(graph, node, replacement));
  }  // end of looping through nodes in the capture list

  for (auto node : nodes_to_capture) {
    NGRAPH_VLOG(4) << "Removing: " << node->name();
    graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
