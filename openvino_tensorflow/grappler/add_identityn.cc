/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#include "openvino_tensorflow/grappler/ngraph_add_identityn.h"

using namespace std;

namespace tensorflow {

namespace openvino_tensorflow {

Status AddIdentityN(Graph* input_graph, std::set<string> skip_these_nodes) {
  for (auto node : input_graph->op_nodes()) {
    bool fetch_node = false;
    bool ref_type = false;
    fetch_node = skip_these_nodes.find(node->name()) != skip_these_nodes.end();
    if (fetch_node) {
      NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Fetch Node " << node->name();
      // Check the number of outputs of the 'fetch_node'
      // Only move further to create an IdentityN node
      // if it is greater than 0
      // Also, make sure that none of the output types is
      // a ref type because IdentityN does not support
      // an input of type ref type
      if (node->num_outputs()) {
        std::vector<NodeBuilder::NodeOut> inputs;
        std::vector<DataType> input_types;
        for (int i = 0; i < node->num_outputs(); i++) {
          if (IsRefType(node->output_type(i))) {
            NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: "
                           << "Datatype for the node output"
                           << " at index " << i << " is ref type";
            ref_type = true;
            break;
          }
          input_types.push_back(node->output_type(i));
          inputs.push_back(NodeBuilder::NodeOut(node, i));
        }

        if (ref_type) {
          NGRAPH_VLOG(5)
              << "NGTF_OPTIMIZER: Cannot construct an IdentityN node";
          continue;
        }

        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Creating an IdentityN node";
        Node* identityN_node;
        TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "IdentityN")
                               .Attr("T", input_types)
                               .Input(inputs)
                               .Device(node->assigned_device_name())
                               .Finalize(input_graph, &identityN_node));

        identityN_node->set_assigned_device_name(node->assigned_device_name());

        // Rename the skip node
        // Get a new name for the node with the given prefix
        // We will use the 'original-node-name_ngraph' as the prefix
        string new_name = input_graph->NewName(node->name() + "_ngraph");
        // TODO: Use (guaranteed) unique name here
        node->set_name(new_name);
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: New name for fetch node "
                       << node->name();
      } else {
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: num outputs " << node->num_outputs();
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Cannot construct an IdentityN node";
      }
    }
  }
  return Status::OK();
}

}  // namespace openvino_tensorflow

}  // namespace tensorflow
