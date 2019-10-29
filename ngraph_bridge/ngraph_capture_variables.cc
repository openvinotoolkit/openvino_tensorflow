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

#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// Utility function to check if it is an output node
// Skip capturing it, if yes.
static bool IsOutputNode(const Node* node,
                         const std::set<string> skip_these_nodes) {
  bool found = skip_these_nodes.find(node->name()) != skip_these_nodes.end();
  if (found) {
    NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Found Output Node: " << node->name()
                   << " - skip capturing it";
  }
  return found;
}

// Status AddWriteToDeviceOp(Graph* input_graph, std::set<string> skip_these_nodes) {
//   for (auto node : input_graph->op_nodes()) {
//     bool fetch_node = false;
//     bool ref_type = false;
//     fetch_node = skip_these_nodes.find(node->name()) != skip_these_nodes.end();
//     if (fetch_node) {
//       NGRAPH_VLOG(0) << "AddWriteToDeviceOp: Fetch Node " << node->name();
//       // Check the number of outputs of the 'fetch_node'
//       // Only move further to create an IdentityN node
//       // if it is greater than 0
//       // Also, make sure that none of the output types is
//       // a ref type because IdentityN does not support
//       // an input of type ref type
//       if (node->num_outputs()) {
//         std::vector<NodeBuilder::NodeOut> inputs;
//         std::vector<DataType> input_types;
//         for (int i = 0; i < node->num_outputs(); i++) {
//           if (IsRefType(node->output_type(i))) {
//             NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: "
//                            << "Datatype for the node output"
//                            << " at index " << i << " is ref type";
//             ref_type = true;
//             break;
//           }
//           input_types.push_back(node->output_type(i));
//           inputs.push_back(NodeBuilder::NodeOut(node, i));
//         }

//         if (ref_type) {
//           NGRAPH_VLOG(5)
//               << "NGTF_OPTIMIZER: Cannot construct an IdentityN node";
//           continue;
//         }

//       //------------------------------------------
//         NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Creating an IdentityN node";
//         Node* write_to_device_node;
//         TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "IdentityN")
//                                .Attr("T", input_types)
//                                .Input(inputs)
//                                .Device(node->assigned_device_name())
//                                .Finalize(input_graph, &write_to_device_node));

//         TF_RETURN_IF_ERROR(NodeBuilder("ng_write_to_device", "NGraphWriteToDevice")
//                                 .Input(input_node)
//                                 .Device(prefetch_node->assigned_device_name())
//                                 .Finalize(graph, &write_to_device_node));


//         write_to_device_node->set_assigned_device_name(node->assigned_device_name());

//         // Rename the skip node
//         // Get a new name for the node with the given prefix
//         // We will use the 'original-node-name_ngraph' as the prefix
//         string new_name = input_graph->NewName(node->name() + "_ngraph");
//         // TODO: Use (guaranteed) unique name here
//         node->set_name(new_name);
//         NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: New name for fetch node "
//                        << node->name();
//       } else {
//         NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: num outputs " << node->num_outputs();
//         NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Cannot construct an IdentityN node";
//       }
//     }
//   }
//   return Status::OK();
// }

//
// Main entry point for the variable-capture.
//
Status CaptureVariables(Graph* graph, const std::set<string> skip_these_nodes) {
  if (config::IsEnabled() == false) {
    return Status::OK();
  }

  std::vector<Node*> replaced_nodes;

  Node* prefetch_node = nullptr;

  for (auto node : graph->op_nodes()) {
    if (!IsOutputNode(node, skip_these_nodes)) {
      if (node->type_string() == "VariableV2") {
        NGRAPH_VLOG(4) << "Capturing: " << node->name();

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
        TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "NGraphVariable")
                               .Attr("shape", shape)
                               .Attr("dtype", dtype)
                               .Attr("container", container)
                               .Attr("shared_name", shared_name)
                               .Device(node->assigned_device_name())
                               .Finalize(graph, &replacement));

        replacement->set_assigned_device_name(node->assigned_device_name());

        std::vector<const Edge*> edges;

        // Add edge from the input nodes (to the variable node (VariableV2))
        // to the replacement node (NGraphVariable)
        NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                       << replacement->DebugString();

        std::vector<const Edge*> edges_to_remove;
        std::vector<std::tuple<Node*, int, Node*, int>> edges_to_add;
        for (auto edge : node->in_edges()) {
          NGRAPH_VLOG(4) << "Replacing: In Edge " << edge->DebugString();
          edges_to_add.push_back(std::tuple<Node*, int, Node*, int>(
              edge->src(), edge->src_output(), replacement, edge->dst_input()));
          edges_to_remove.push_back(edge);
        }

        for (auto edge : node->out_edges()) {
          NGRAPH_VLOG(4) << "Replacing: OutEdge " << edge->DebugString();
          edges_to_add.push_back(std::tuple<Node*, int, Node*, int>(
              replacement, edge->src_output(), edge->dst(), edge->dst_input()));
          edges_to_remove.push_back(edge);
        }

        for (const auto& i : edges_to_add) {
          NGRAPH_VLOG(4) << "Adding: " << get<0>(i) << "  " << get<1>(i) << "  "
                         << get<2>(i) << " " << get<3>(i);
          graph->AddEdge(get<0>(i), get<1>(i), get<2>(i), get<3>(i));
        }

        // Though edges will be removed when we remove the node
        // we specifically remove the edges to be sure
        for (auto edge : edges_to_remove) {
          NGRAPH_VLOG(4) << "Removing: " << edge->DebugString();
          graph->RemoveEdge(edge);
        }

        replaced_nodes.push_back(node);
      }
      else if (node->type_string() == "PrefetchDataset"){
        // Collect the prefetch_node so that we can add
        // the NGraphWriteToDevice Op after this one
        prefetch_node = node;
      }
    }
  }

  for (auto node : replaced_nodes) {
    NGRAPH_VLOG(4) << "Removing: " << node->name();
    graph->RemoveNode(node);
  }

  // Now add the NGraphWriteToDevice node
  if (prefetch_node != nullptr){
    std::vector<const Edge*> edges;

    // PrefetchDataset should have only one outgoing edge
    // Assert otherwise?
    
    std::vector<const Edge*> edges_to_remove;
    std::vector<std::tuple<Node*, int, Node*, int>> edges_to_add;

    // Get the target nide of this prefetch node
    Node* prefetch_target = nullptr;
    for (auto edge: prefetch_node->out_edges()){
      prefetch_target = edge->dst();
      std::cout << "Out Edge: " << edge->DebugString() << std::endl;
      std::cout << "Target: " << prefetch_target->name() << std::endl;
      edges_to_remove.push_back(edge);
    }

    // First remove the existing edgex
    for (auto edge : edges_to_remove) {
      NGRAPH_VLOG(0) << "Removing: " << edge->DebugString();
      graph->RemoveEdge(edge);
    }

    Node* write_to_device_node = nullptr;
    auto input_node = NodeBuilder::NodeOut(prefetch_node,0);
    TF_RETURN_IF_ERROR(NodeBuilder("ng_write_to_device", "NGraphWriteToDevice")
                            .Input(input_node)
                            .Device(prefetch_node->assigned_device_name())
                            .Finalize(graph, &write_to_device_node));
    write_to_device_node->set_assigned_device_name(prefetch_node->assigned_device_name());

    // Add edge from the input nodes (to the variable node (VariableV2))
    // to the replacement node (NGraphVariable)
    std::cout << "Inserting Node " << write_to_device_node->DebugString() << " after "
                    << prefetch_node->DebugString() << std::endl << std::endl;
    
    // for (auto edge:edges_to_remove){
      // graph->AddEdge(prefetch_node, edge->src_output(), write_to_device_node, edge->dst_input());
      // graph->AddEdge(write_to_device_node, edge->src_output(), prefetch_target, edge->dst_input());
      //graph->AddEdge(prefetch_node, 0, write_to_device_node, 0);
      graph->AddEdge(write_to_device_node, 0, prefetch_target, 0);
    // }

  }
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
