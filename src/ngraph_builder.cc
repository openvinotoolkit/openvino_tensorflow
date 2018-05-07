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

#include "ngraph_builder.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb_text.h"
#include "tensorflow/core/graph/algorithm.h"

namespace ngraph_bridge {
unique_ptr<ng::Function> Builder::TransformGraph(const tf::Graph* input_graph) {
  //   for (const tf::Node* n : input_graph->nodes()) {
  //     cout << "Node: " << n->name() << " Type: " << n->type_string() << endl;
  //     for (const tf::Edge* edge : n->in_edges()) {
  //       cout << "\tEdge " << edge->src()->name() << " --> " <<
  //       edge->dst()->name()
  //            << endl;
  //     }
  //   }
  // Do a topological sort
  // GetReversePostOrder will give us topological sort.
  vector<tf::Node*> ordered;
  tf::GetReversePostOrder(*input_graph, &ordered);

  cout << "After Topological sort\n";
  //   for (auto n : ordered) {
  //     cout << "Node: " << n->name() << " Type: " << n->type_string() << endl;
  //     for (const tf::Edge* edge : n->in_edges()) {
  //       cout << "\tEdge " << edge->src()->name() << " --> " <<
  //       edge->dst()->name()
  //            << endl;
  //     }
  //   }

  // Now start building the nGraph.
  unordered_map<string, shared_ptr<ng::op::Parameter>> parameter_map;
  for (auto n : ordered) {
    tf::AttrSlice n_attrs = n->attrs();
    if (n->type_string() == "Placeholder") {
      tf::DataType dtype;
      if (tf::GetNodeAttr(n_attrs, "dtype", &dtype) != tf::Status::OK()) {
        cout << "Error: No data type defined for Parameter!\n";
        return nullptr;
      }
      tf::TensorShapeProto shape_proto;
      if (tf::GetNodeAttr(n_attrs, "shape", &shape_proto) != tf::Status::OK()) {
        cout << "Error: No shape type defined for Parameter!\n";
        return nullptr;
      }

      // Get the name of the node
      cout << "Parameter: " << n->name() << "[" << tf::DataTypeString(dtype)
           << "] Shape: ";
      tf::PartialTensorShape shape(shape_proto);

      for (int d = 0; d < shape.dims(); ++d) {
        if (d > 0) cout << ", ";
        cout << shape.dim_size(d);
      }

      // Create the nGraph::op::Parameter
      // TODO
      //<< " Shape: " << tf::ProtoShortDebugString(shape) << endl;
      cout << endl;
    } else {
      // cout << "Node: " << n->name() << " Type: " << n->type_string() << endl;
      for (const tf::Edge* edge : n->in_edges()) {
        cout << "\tEdge " << edge->src()->name() << "["
             << edge->src()->type_string() << "] --> " << edge->dst()->name()
             << "[" << edge->dst()->type_string() << "]" << endl;
      }
    }
  }

  return nullptr;
}

}  // namespace ngraph_bridge