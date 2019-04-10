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
#include "ngraph_rewrite_for_tracking.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/types.h"

#include "ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// Main entry point for rewrite-for-tracking.
//
Status RewriteForTracking(Graph* graph, int graph_id) {
  std::vector<Node*> replaced_nodes;

  for (auto node : graph->op_nodes()) {
    if (node->type_string() == "NGraphVariable") {
      NGRAPH_VLOG(4) << "Checking: " << node->name();

      bool just_looking = true;

      // If any of the nodes reading from this Variable node read the data as
      // reference then we dont track it, else we do
      for (auto edge : node->out_edges()) {
        if (edge->dst()->IsOp() && !edge->IsControlEdge() &&
            IsRefType(edge->dst()->input_type(edge->dst_input()))) {
          just_looking = false;
          break;
        }
      }

      if (just_looking) {
        NGRAPH_VLOG(4) << "Just looking: " << node->name();

        TensorShape shape;
        DataType dtype;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shape", &shape));
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "dtype", &dtype));

        std::string container;
        std::string shared_name;
        if (GetNodeAttr(node->attrs(), "container", &container) !=
            Status::OK()) {
          container = "";
        }
        if (GetNodeAttr(node->attrs(), "shared_name", &shared_name) !=
            Status::OK()) {
          shared_name = "";
        }

        Node* replacement;

        // TODO(amprocte): Do we need to copy "_" attributes?
        TF_RETURN_IF_ERROR(
            NodeBuilder(graph->NewName(node->name() + "/peek"),
                        "NGraphVariable")
                .Attr("shape", shape)
                .Attr("dtype", dtype)
                .Attr("container", container)
                .Attr("shared_name",
                      (shared_name.empty() ? node->name() : shared_name))
                .Attr("just_looking", true)
                .Device(node->assigned_device_name())
                .Finalize(graph, &replacement));

        replacement->set_assigned_device_name(node->assigned_device_name());

        // Add edge from the input nodes (to the variable node (NGraphVariable))
        // to the new replacement node (also of type NGraphVariable)
        NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                       << replacement->DebugString();

        // Though edges will be removed when we remove the node
        // we specifically remove the edges to be sure
        for (auto edge : node->in_edges()) {
          NGRAPH_VLOG(4) << "Replacing: " << edge->DebugString();
          graph->AddEdge(edge->src(), edge->src_output(), replacement,
                         edge->dst_input());
          graph->RemoveEdge(edge);
        }

        std::vector<const Edge*> edges;
        for (auto edge : node->out_edges()) {
          edges.push_back(edge);
        }
        for (auto edge : edges) {
          NGRAPH_VLOG(4) << "Replacing: " << edge->DebugString();
          graph->AddEdge(replacement, edge->src_output(), edge->dst(),
                         edge->dst_input());
          graph->RemoveEdge(edge);
        }

        replaced_nodes.push_back(node);
      } else {
        NGRAPH_VLOG(4) << "Not just looking: " << node->name();
      }
    }
  }

  for (auto node : replaced_nodes) {
    NGRAPH_VLOG(4) << "Removing: " << node->name();
    graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
