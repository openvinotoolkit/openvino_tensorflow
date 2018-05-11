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
#include "tensorflow/core/graph/edgeset.h"

namespace ngraph_bridge {
unique_ptr<ng::Function> Builder::TranslateGraph(const tf::Graph* input_graph) {
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
  vector<tf::Node*> tf_params;
  vector<tf::Node*> tf_consts;
  vector<tf::Node*> tf_ret_vals;
  vector<tf::Node*> tf_ops;

  for (auto n : ordered) {
    if (n->IsSink() || n->IsSource() || n->IsControlFlow()) {
      continue;
    }

    if (n->IsConstant()) {
      std::cout << "Constant Node: " << n->type_string() << endl;
    }
    if (n->IsVariable()) {
      std::cout << "Variable Node: " << n->type_string() << endl;
    }
    if (n->type_string() == "_Arg") {
      tf_params.push_back(n);
    } else if (n->type_string() == "_Retval") {
      tf_ret_vals.push_back(n);
    } else if (!n->IsConstant()) {
      tf_ops.push_back(n);
    }
  }

  cout << "Parameters\n";
  for (auto parm : tf_params) {
    tf::DataType dtype;
    if (tf::GetNodeAttr(parm->attrs(), "T", &dtype) != tf::Status::OK()) {
      cout << "Error: No data type defined for _Arg!\n";
      return nullptr;
    }
    int index;
    if (tf::GetNodeAttr(parm->attrs(), "index", &index) != tf::Status::OK()) {
      cout << "Error: No index defined for _Arg!\n";
      return nullptr;
    }
    cout << "Param: " << index << endl;
  }

  cout << "Return Values\n";
  for (auto n : tf_ret_vals) {
    tf::DataType dtype;
    if (tf::GetNodeAttr(n->attrs(), "T", &dtype) != tf::Status::OK()) {
      cout << "Error: No data type defined for _Arg!\n";
      return nullptr;
    }
    int index;
    if (tf::GetNodeAttr(n->attrs(), "index", &index) != tf::Status::OK()) {
      cout << "Error: No index defined for _Arg!\n";
      return nullptr;
    }
    cout << "RetVal: " << index << endl;
    for (const tf::Edge* edge : n->in_edges()) {
      cout << "\tSrc: " << edge->src()->name() << "("
           << edge->src()->type_string() << ")" << endl;
    }
  }

  cout << "Ops\n";
  for (auto op : tf_ops) {
    cout << "Op: " << op->name() << "(" << op->type_string() << ")" << endl;
    for (const tf::Edge* edge : op->in_edges()) {
      cout << "\tSrc: " << edge->src()->name() << "("
           << edge->src()->type_string() << ")" << endl;
    }
  }

  // for (auto n : ordered) {
  //   tf::AttrSlice n_attrs = n->attrs();
  //   if (n->type_string() == "_Arg") {
  //     tf::DataType dtype;
  //     if (tf::GetNodeAttr(n_attrs, "T", &dtype) != tf::Status::OK()) {
  //       cout << "Error: No data type defined for _Arg!\n";
  //       return nullptr;
  //     }
  //     int index;
  //     if (tf::GetNodeAttr(n_attrs, "index", &index) != tf::Status::OK()) {
  //       cout << "Error: No index defined for _Arg!\n";
  //       return nullptr;
  //     }

  //     // tf::TensorShapeProto shape_proto;
  //     // if (tf::GetNodeAttr(n_attrs, "shape", &shape_proto) !=
  //     // tf::Status::OK()) {
  //     //   cout << "Error: No shape type defined for Parameter!\n";
  //     //   return nullptr;
  //     // }

  //     // Get the name of the node
  //     cout << "_Arg: " << n->name() << "[" << tf::DataTypeString(dtype)
  //          << "] Index: " << index << endl;
  //     // tf::PartialTensorShape shape(shape_proto);
  //     // if (!shape.IsFullyDefined()) {
  //     //   cout << "Error: cannot use partially defined shapes" << endl;
  //     // }

  //     // for (int d = 0; d < shape.dims(); ++d) {
  //     //   if (d > 0) cout << ", ";
  //     //   cout << shape.dim_size(d);
  //     // }

  //     // Create the nGraph::op::Parameter
  //     // TODO

  //     //<< " Shape: " << tf::ProtoShortDebugString(shape) << endl;
  //     // cout << endl;
  //   } else {
  //     // cout << "Node: " << n->name() << " Type: " << n->type_string() <<
  //     endl; for (const tf::Edge* edge : n->in_edges()) {
  //       cout << "\tEdge " << edge->src()->name() << "["
  //            << edge->src()->type_string() << "] --> " << edge->dst()->name()
  //            << "[" << edge->dst()->type_string() << "]" << endl;
  //     }
  //   }
  // }

  return nullptr;
}

}  // namespace ngraph_bridge