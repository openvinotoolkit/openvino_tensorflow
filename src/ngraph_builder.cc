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
#include "ngraph_utils.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb_text.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/edgeset.h"

namespace ngraph_bridge {
unique_ptr<ng::Function> Builder::TranslateGraph(
    const std::vector<tf::TensorShape>& inputs, const tf::Graph* input_graph) {
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
  vector<tf::Node*> tf_params;
  vector<tf::Node*> tf_consts;
  vector<tf::Node*> tf_ret_vals;
  vector<tf::Node*> tf_ops;

  for (auto n : ordered) {
    if (n->IsSink() || n->IsSource() || n->IsControlFlow()) {
      continue;
    }

    // if (n->IsConstant()) {
    //   std::cout << "Constant Node: " << n->type_string() << endl;
    // }
    if (n->IsVariable()) {
      std::cout << "Variable Node: " << n->type_string() << endl;
    }
    if (n->type_string() == "_Arg") {
      tf_params.push_back(n);
    } else if (n->type_string() == "_Retval") {
      tf_ret_vals.push_back(n);
    } else {
      tf_ops.push_back(n);
    }
  }

  unordered_map<string, shared_ptr<ng::Node>> ng_op_map;
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

    const tf::TensorShape& tf_shape = inputs[index];
    cout << "Param: " << index << " Name: " << parm->name()
         << " Rank: " << tf_shape.dims() << " Shape: " << tf_shape << endl;
    // TODO: Convert tf::Type to ng::type
    ng::Shape ng_shape(tf_shape.dims());
    for (int i = 0; i < tf_shape.dims(); ++i) {
      ng_shape[i] = tf_shape.dim_size(i);
    }

    auto ng_param = make_shared<ng::op::Parameter>(ng::element::f32, ng_shape);
    ng_op_map[parm->name()] = ng_param;
  }

  cout << "Ops\n";
  for (auto op : tf_ops) {
    cout << "Op: " << op->name() << "(" << op->type_string() << ")" << endl;
    for (const tf::Edge* edge : op->in_edges()) {
      cout << "\tSrc: " << edge->src()->name() << "("
           << edge->src()->type_string() << ")" << endl;
    }

    // Now create the ng ops from tf ops
    if (op->type_string() == "Const") {
      // Data type first
      tf::DataType dtype;
      if (tf::GetNodeAttr(op->attrs(), "dtype", &dtype) != tf::Status::OK()) {
        cout << "Error: No data type defined for Constant!\n";
        return nullptr;
      }

      // TODO: Create a type based on the DataType
      tf::TensorShapeProto shape_proto;
      vector<float> const_values;
      if (!ValuesFromConstNode<float>(op->def(), &shape_proto, &const_values)) {
        cout << "Error: Cannot get values from Constant!" << endl;
      }
      // Shape of the tensor
      // if (tf::GetNodeAttr(op->attrs(), "tensor_shape", &shape_proto) !=
      //     tf::Status::OK()) {
      //   cout << "Error: No shape type defined for Constant!\n";
      //   return nullptr;
      // }
      tf::TensorShape const_shape(shape_proto);
      ng::Shape ng_shape(const_shape.dims());
      for (int i = 0; i < const_shape.dims(); ++i) {
        ng_shape[i] = const_shape.dim_size(i);
      }

      vector<float> float_t(ng::shape_size(ng_shape), 0);
      auto ng_node =
          make_shared<ng::op::Constant>(ng::element::f32, ng_shape, float_t);
      ng_op_map[op->name()] = ng_node;

    } else if ((op->type_string() == "Mul") || (op->type_string() == "Add")) {
      if (op->num_inputs() != 2) {
        cout << "Error: Number of inputs is not 2 for elementwise op\n";
        return nullptr;
      }
      tf::Node* tf_lhs;
      if (op->input_node(0, &tf_lhs) != tf::Status::OK()) {
        cout << "Cannot get the input node 0";
        return nullptr;
      }

      tf::Node* tf_rhs;
      if (op->input_node(1, &tf_rhs) != tf::Status::OK()) {
        cout << "Cannot get the input node 1";
        return nullptr;
      }

      auto ng_lhs = ng_op_map.find(tf_lhs->name())->second;
      auto ng_rhs = ng_op_map.find(tf_rhs->name())->second;

      cout << "NG LHS node: " << ng_lhs->get_name() << endl;
      cout << "NG RHS node: " << ng_rhs->get_name() << endl;

      shared_ptr<ng::Node> ng_op;

      if (op->type_string() == "Mul") {
        ng_op = make_shared<ngraph::op::Multiply>(ng_lhs, ng_rhs);
      } else if (op->type_string() == "Add") {
        ng_op = make_shared<ngraph::op::Add>(ng_lhs, ng_rhs);
      }

      // Add to the map
      ng_op_map[op->name()] = ng_op;

      // Get the inputs for this node
    }
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