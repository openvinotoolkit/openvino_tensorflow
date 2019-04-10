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
#include "tensorflow/core/graph/types.h"

#include "ngraph_replace_op_utilities.h"
#include "ngraph_rewrite_for_tracking.h"
#include "ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// Main entry point for rewrite-for-tracking.
//
Status RewriteForTracking(Graph* graph, int graph_id) {
  const static std::map<
      const string,
      const function<Status(
          Graph * graph, Node * node, Node * *replacement,
          const string replacement_node_name, const string replacement_op_type,
          const bool just_looking, const bool outputs_ng_supported,
          const int graph_id, const bool is_backend_set)>>
      REWRITE_REPLACE_OP_MAP{{"NGraphAssign", ReplaceAssign},
                             {"NGraphVariable", ReplaceVariable}};

  std::vector<Node*> replaced_nodes;
  for (auto node : graph->op_nodes()) {
    auto itr = REWRITE_REPLACE_OP_MAP.find(node->type_string());
    if (itr != REWRITE_REPLACE_OP_MAP.end()) {
      NGRAPH_VLOG(1) << "Checking: " << DebugNode(node) << " " << node->name();

      bool just_looking = true;
      bool outputs_ng_supported = true;

      // Check if all the outputs of this node are supported by nGraph
      for (auto edge : node->out_edges()) {
        auto dst = edge->dst();
        NGRAPH_VLOG(1) << "dst node " << DebugNode(dst);
        if (dst->IsOp() && !edge->IsControlEdge() &&
            !IsNGSupportedType(dst->type_string())) {
          NGRAPH_VLOG(1) << "Dst node ngraph doesn't support ";
          outputs_ng_supported = false;
          break;
        }
      }

      // If any of the nodes reading from this Variable node read the data as
      // reference then we dont track it, else we do
      for (auto edge : node->out_edges()) {
        if (edge->dst()->IsOp() && !edge->IsControlEdge() &&
            IsRefType(edge->dst()->input_type(edge->dst_input()))) {
          // if the output reference is read by NGraph supported ops, do not
          // turn off just_looking
          if (!IsNGVariableType(edge->dst()->type_string())) {
            NGRAPH_VLOG(1) << DebugNode(edge->dst())
                           << "needs reference, setting just_looking to false";
            just_looking = false;
            break;
          }
        }
      }

      NGRAPH_VLOG(1) << "Just Looking: " << PrintBool(just_looking);
      NGRAPH_VLOG(1) << "Outputs supported by nGraph: "
                     << PrintBool(outputs_ng_supported);
      NGRAPH_VLOG(1) << "Requires Replacement "
                     << PrintBool(just_looking || !outputs_ng_supported);

      std::string node_new_name = node->name();

      if (just_looking) {
        node_new_name += "/peek";
      }

      if (!outputs_ng_supported) {
        node_new_name += "/non_ng_outputs";
      }

      node_new_name += "/gid_" + to_string(graph_id);
      NGRAPH_VLOG(1) << "Replacing " << node->name() << " New Node name "
                     << node_new_name;

      Node* replacement;

      // Create and add the replacement node
      TF_RETURN_IF_ERROR((itr->second)(graph, node, &replacement, node_new_name,
                                       node->type_string(), just_looking,
                                       outputs_ng_supported, graph_id, true));

      TF_RETURN_IF_ERROR(ReplaceInputControlEdges(graph, node, replacement));
      TF_RETURN_IF_ERROR(ReplaceOutputEdges(graph, node, replacement));

      replaced_nodes.push_back(node);

    }  // end of checking if it is NGVariableType
  }    // end of looping through the nodes in the graph
  for (auto node : replaced_nodes) {
    graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
