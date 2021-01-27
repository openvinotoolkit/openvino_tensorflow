/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include <iomanip>
#include <iostream>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

#include "api.h"
#include "assign_clusters.h"
#include "cluster_manager.h"
#include "deassign_clusters.h"
#include "encapsulate_clusters.h"
#include "log.h"
#include "ngraph_optimizer.h"
#include "ngraph_rewrite_pass.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

Status AddIdentityN(Graph* input_graph, std::set<string> skip_these_nodes) {
  for (auto node : input_graph->op_nodes()) {
    bool fetch_node = false;
    bool ref_type = false;
    fetch_node = skip_these_nodes.find(node->name()) != skip_these_nodes.end();
    if (fetch_node) {
      NGRAPH_VLOG(5) << "ngraph-optimizer: Fetch Node " << node->name();
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
            NGRAPH_VLOG(5) << "ngraph-optimizer: "
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
              << "ngraph-optimizer: Cannot construct an IdentityN node";
          continue;
        }

        NGRAPH_VLOG(5) << "ngraph-optimizer: Creating an IdentityN node";
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
        NGRAPH_VLOG(5) << "ngraph-optimizer: New name for fetch node "
                       << node->name();
      } else {
        NGRAPH_VLOG(5) << "ngraph-optimizer: num outputs "
                       << node->num_outputs();
        NGRAPH_VLOG(5)
            << "ngraph-optimizer: Cannot construct an IdentityN node";
      }
    }
  }
  return Status::OK();
}

Status NgraphOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  const auto params = config->parameter_map();
  for (auto i : params) {
    m_config_map["_ngraph_" + i.first] = i.second.s();
    NGRAPH_VLOG(3) << "Attribute: " << i.first
                   << " Value: " << m_config_map["_ngraph_" + i.first];
  }
  return Status::OK();
}

Status NgraphOptimizer::Optimize(tensorflow::grappler::Cluster* cluster,
                                 const tensorflow::grappler::GrapplerItem& item,
                                 GraphDef* output) {
  NGRAPH_VLOG(5) << "ngraph-optimizer: grappler item id " << item.id;

  // Convert the GraphDef to Graph
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = true;
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, item.graph, &graph));

  // TODO: Find out a better way to preserve feed nodes, init_ops and
  // keep_ops instead of just skipping those from clustering.
  // Get nodes to be preserved/skipped
  std::set<string> nodes_to_preserve;

  // Feed Nodes
  for (size_t i = 0; i < item.feed.size(); i++) {
    nodes_to_preserve.insert(item.feed[i].first);
  }

  // Keep Ops
  nodes_to_preserve.insert(item.keep_ops.begin(), item.keep_ops.end());

  // Init Ops
  nodes_to_preserve.insert(item.init_ops.begin(), item.init_ops.end());

  // Find a list of nodes that are of the types that are disabled
  std::set<string> disabled_nodes;
  std::set<string> disabled_ops_set = api::GetDisabledOps();
  for (auto itr : graph.nodes()) {
    if (disabled_ops_set.find(itr->type_string()) != disabled_ops_set.end()) {
      disabled_nodes.insert(itr->name());
    }
  }

  // Fetch Nodes
  std::set<string> fetch_nodes;
  for (const string& f : item.fetch) {
    int pos = f.find(":");
    fetch_nodes.insert(f.substr(0, pos));
  }

  // nodes_to_add_identity_to = fetch_nodes - disabled_nodes
  std::set<string> nodes_to_add_identity_to;
  std::set_difference(fetch_nodes.begin(), fetch_nodes.end(),
                      disabled_nodes.begin(), disabled_nodes.end(),
                      std::inserter(nodes_to_add_identity_to,
                                    nodes_to_add_identity_to.begin()));

  // Rewrite graph to add IdentityN node so the fetch node can be encapsulated
  // as well
  // If the fetch node in question has 0 outputs or any of the outputs
  // has ref type as a data type then don't add IdentityN node, but the fetch
  // node will be skipped from capturing and marking for clustering.
  TF_RETURN_IF_ERROR(AddIdentityN(&graph, nodes_to_add_identity_to));

  nodes_to_preserve.insert(nodes_to_add_identity_to.begin(),
                           nodes_to_add_identity_to.end());
  std::set<string>& skip_these_nodes = nodes_to_preserve;

  NGraphRewritePass rwp;
  rwp.Rewrite(&graph, skip_these_nodes, m_config_map);

  // Convert the graph back to Graphdef
  graph.ToGraphDef(output);
  return Status::OK();
}

void NgraphOptimizer::Feedback(tensorflow::grappler::Cluster* cluster,
                               const tensorflow::grappler::GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(NgraphOptimizer, "ngraph-optimizer");

}  // end namespace ngraph_bridge
}  // end namespace tensorflow
