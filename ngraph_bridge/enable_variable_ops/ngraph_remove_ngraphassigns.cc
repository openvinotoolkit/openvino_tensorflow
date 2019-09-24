/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include "ngraph_bridge/enable_variable_ops/ngraph_remove_ngraphassigns.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status RemoveNGraphAssigns(Graph* graph) {
  vector<Node*> remove_nodes;

  for (auto node : graph->op_nodes()) {
    if (node->type_string() == "NGraphAssign" && NodeIsMarkedForRemoval(node)) {
      NGRAPH_VLOG(3) << "Removing NGraphAssign " << node->name();
      Node *input_0, *input_1;
      // input_0 is of type NGraphVariable/NGraphAssign
      // input_1 is of type NGraphEncapsulateOp
      TF_RETURN_IF_ERROR(node->input_node(0, &input_0));
      TF_RETURN_IF_ERROR(node->input_node(1, &input_1));
      if (!IsNGVariableType(input_0->type_string())) {
        return errors::Internal(
            "Got NGraphAssign with input variable tensor from ",
            input_0->type_string());
      }

      if (input_1->type_string() != "NGraphEncapsulate") {
        return errors::Internal(
            "Trying to remove NGraphAssign with input value computed from ",
            input_1->type_string());
      }

      // Add control edge between the Op providing the variable and the Op
      // updating it
      graph->AddEdge(input_0, Graph::kControlSlot, input_1,
                     Graph::kControlSlot);

      // Handle input edges
      NGRAPH_VLOG(3) << "Handling input edges ";
      vector<const Edge*> remove_edges;
      for (auto edge : node->in_edges()) {
        // attach incoming control edge to input_1, as that's where update
        // will happen
        if (edge->IsControlEdge()) {
          // avoid cycles
          if (edge->src() == input_1) continue;
          graph->AddEdge(edge->src(), edge->src_output(), input_1,
                         edge->dst_input());
        }
        remove_edges.push_back(edge);
      }

      // Handle output edges
      NGRAPH_VLOG(3) << "Handling output edges ";
      for (auto edge : node->out_edges()) {
        if (edge->IsControlEdge()) {
          // Add control edge from Encap to the dst node
          graph->AddEdge(input_1, edge->src_output(), edge->dst(),
                         edge->dst_input());
        } else {
          // Note: TF takes care whether the edge to be added is ref-type or not
          // For e.g. if we add edge Var -> Add/Encap (Add's input is
          // non-ref-type)
          // if we add edge Var -> Assign (Assign's input is ref-type)

          // Add edge from Variable to the dst node
          graph->AddEdge(input_0, edge->src_output(), edge->dst(),
                         edge->dst_input());
          // Add control edge from Encap to the dst node, the variable should be
          // used only after the update is completed by Encapsulate op
          graph->AddEdge(input_1, Graph::kControlSlot, edge->dst(),
                         Graph::kControlSlot);
        }
        remove_edges.push_back(edge);
      }

      for (auto edge : remove_edges) {
        graph->RemoveEdge(edge);
      }

      // Add the node for removal
      remove_nodes.push_back(node);
    }
  }

  // Remove Nodes
  for (auto node : remove_nodes) {
    graph->RemoveNode(node);
  }

  return Status::OK();
}

bool NodeIsMarkedForRemoval(const Node* node) {
  bool is_marked;
  return (GetNodeAttr(node->attrs(), "_ngraph_remove", &is_marked) ==
              Status::OK() &&
          is_marked);
}

}  // namespace ngraph_bridge

}  // namespace tensorflow