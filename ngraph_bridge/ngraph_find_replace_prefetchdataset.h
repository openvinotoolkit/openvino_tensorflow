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
#ifndef NGRAPH_FIND_REPLACE_PREFETCHDATASET_H_
#define NGRAPH_FIND_REPLACE_PREFETCHDATASET_H_
#pragma once

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_prefetch_shared_data.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Node* FindPrefetch(Node* makeiterator_node) {
  Node* prefetch_node = nullptr;
  for (auto e1 : makeiterator_node->in_edges()) {
    Node* n = e1->src();
    // resnet case
    if (n->type_string() == "PrefetchDataset") {
      prefetch_node = n;
      break;
    } else if (n->type_string() == "ModelDataset") {  // axpy case
      for (auto e2 : n->in_edges()) {
        if (e2->src()->type_string() == "OptimizeDataset") {
          for (auto e3 : e2->src()->in_edges()) {
            if (e3->src()->type_string() == "PrefetchDataset") {
              prefetch_node = e3->src();
              break;
            }
          }
        } else {
          // this is neither axpy nor resnet
          break;
        }
      }
    }
  }
  return prefetch_node;
}

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
    NGRAPH_VLOG(4) << "Replacing: OutEdge " << edge->DebugString();
    // Collect new output edges between the new prefetch node and the next node
    edges_to_add.push_back(std::tuple<Node*, int, Node*, int>(
        replacement, edge->src_output(), edge->dst(), edge->dst_input()));
    // Remove the output edge from the current prefetch node
    edges_to_remove.push_back(edge);
  }

  // Now add the new output edges
  // The input edges to the new node is added during the node creation
  for (const auto& i : edges_to_add) {
    NGRAPH_VLOG(4) << "Adding: " << get<0>(i)->name() << "  " << get<1>(i)
                   << "  " << get<2>(i)->name() << " " << get<3>(i);
    graph->AddEdge(get<0>(i), get<1>(i), get<2>(i), get<3>(i));
  }

  // Though edges will be removed when we remove the prefetch_node
  // we specifically remove the edges to be sure
  for (auto edge : edges_to_remove) {
    NGRAPH_VLOG(4) << "Removing: " << edge->DebugString();
    graph->RemoveEdge(edge);
  }

  // Finally remove the current preftetch node
  graph->RemoveNode(prefetch_node);
  NGRAPH_VLOG(4) << "Replaced TF Prefetch Node " << prefetch_node->name()
                 << " with NG Prefetch Node " << replacement->name();
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_FIND_REPLACE_PREFETCHDATASET_H_
