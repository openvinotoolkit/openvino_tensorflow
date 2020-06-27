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

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "ngraph_bridge/enable_variable_ops/ngraph_capture_variables.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_replace_op_utilities.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_replace_variable_modifiers.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
namespace ng = ngraph;
namespace tensorflow {

namespace ngraph_bridge {

// TODO (malikshr) :: Currently we are not checking whether the new op name is
// unique
// New Op names are added in
// 1. ReplaceModifiers
// 2. RewriteForTracking

Status CreateBinaryOpNode(NodeBuilder::NodeOut* ndef_gen_op,
                          std::string new_name_op, std::string op_type,
                          NodeBuilder::NodeOut input1,
                          NodeBuilder::NodeOut input2, DataType dtype,
                          Node*& node, Graph*& graph) {
  Node* generic_op;

  TF_RETURN_IF_ERROR(NodeBuilder(new_name_op, op_type)
                         .Input(input1)
                         .Input(input2)
                         .Attr("T", dtype)
                         .Device(node->assigned_device_name())
                         .Finalize(graph, &(generic_op)));
  generic_op->set_assigned_device_name(node->assigned_device_name());

  *ndef_gen_op = NodeBuilder::NodeOut(generic_op, 0);
  return Status::OK();
}

Status ReplaceModifiers(Graph* graph, int graph_id) {
  // Go over the nodes and replace variable modifiers
  // Each Modifier is replaced with the corresponding computational TF
  // graph followed by NGraphAssign Op
  // If there is an incoming control edge to the Modifier Op
  // It is attached to the first op in the series of the computation TF graph
  vector<Node*> remove_nodes;
  for (auto node : graph->op_nodes()) {
    if (node->type_string() == "NGraphAssignSub" ||
        node->type_string() == "NGraphAssignAdd") {
      NodeBuilder::NodeOut input_ref;
      NodeBuilder::NodeOut input_val;

      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

      for (auto edge : node->in_edges()) {
        if (edge == NULL) {
          continue;
        }
        if (edge->dst()->IsOp() && !edge->IsControlEdge() &&
            IsRefType(edge->dst()->input_type(edge->dst_input()))) {
          input_ref = NodeBuilder::NodeOut(edge->src(), edge->src_output());
        } else {
          input_val = NodeBuilder::NodeOut(edge->src(), edge->src_output());
        }
      }

      string op_type =
          (node->type_string() == "NGraphAssignSub") ? "Sub" : "Add";
      string op_name_suffix = "_" + op_type;
      Node* compute_op;
      string new_name_sub = node->name() + op_name_suffix;
      TF_RETURN_IF_ERROR(NodeBuilder(new_name_sub, op_type)
                             .Input(input_ref)
                             .Input(input_val)
                             .Attr("T", dtype)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &(compute_op)));
      compute_op->set_assigned_device_name(node->assigned_device_name());
      NodeBuilder::NodeOut ndef_compute_op =
          NodeBuilder::NodeOut(compute_op, 0);

      NGRAPH_VLOG(1) << "Compute op name: " << compute_op->name();
      NGRAPH_VLOG(1) << "Compute op assigned device: "
                     << compute_op->assigned_device_name();

      Node* ngraphassign_op;
      string new_name_ngassign = node->name() + "_NGraphAssign";

      TF_RETURN_IF_ERROR(NodeBuilder(new_name_ngassign, "NGraphAssign")
                             .Attr("validate_shape", true)
                             .Attr("use_locking", true)
                             .Attr("T", dtype)
                             .Attr("ngraph_graph_id", 0)
                             .Input(input_ref)
                             .Input(ndef_compute_op)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &ngraphassign_op));
      ngraphassign_op->set_assigned_device_name(node->assigned_device_name());
      NGRAPH_VLOG(1) << "Assign op name: " << ngraphassign_op->name();
      NGRAPH_VLOG(1) << "Assign op assigned device: "
                     << ngraphassign_op->assigned_device_name();

      TF_RETURN_IF_ERROR(ReplaceInputControlEdges(graph, node, compute_op));
      TF_RETURN_IF_ERROR(ReplaceOutputEdges(graph, node, ngraphassign_op));

      remove_nodes.push_back(node);
      NGRAPH_VLOG(1) << "Removing node";

    }  // AssignSub + Assign Add
    else if (node->type_string() == "NGraphApplyGradientDescent") {
      NodeBuilder::NodeOut input_var;
      NodeBuilder::NodeOut input_alpha;
      NodeBuilder::NodeOut input_delta;

      std::vector<const Edge*> input_edges;
      TF_RETURN_IF_ERROR(node->input_edges(&input_edges));

      NGRAPH_VLOG(1) << "No of input edges to ApplyGradientDescent "
                     << input_edges.size();

      input_var = NodeBuilder::NodeOut(input_edges[0]->src(),
                                       input_edges[0]->src_output());
      input_alpha = NodeBuilder::NodeOut(input_edges[1]->src(),
                                         input_edges[1]->src_output());
      input_delta = NodeBuilder::NodeOut(input_edges[2]->src(),
                                         input_edges[2]->src_output());

      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

      Node* mul_op;
      string new_name_mul = node->name() + "_Mul";
      TF_RETURN_IF_ERROR(NodeBuilder(new_name_mul, "Mul")
                             .Input(input_alpha)
                             .Input(input_delta)
                             .Attr("T", dtype)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &(mul_op)));
      mul_op->set_assigned_device_name(node->assigned_device_name());
      NodeBuilder::NodeOut ndef_mul_op = NodeBuilder::NodeOut(mul_op, 0);

      Node* sub_op;
      string new_name_sub = node->name() + "_Sub";
      TF_RETURN_IF_ERROR(NodeBuilder(new_name_sub, "Sub")
                             .Input(input_var)
                             .Input(ndef_mul_op)
                             .Attr("T", dtype)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &(sub_op)));
      sub_op->set_assigned_device_name(node->assigned_device_name());
      NodeBuilder::NodeOut ndef_sub_op = NodeBuilder::NodeOut(sub_op, 0);

      Node* ngraphassign_op;
      string new_name_ngassign = node->name() + "_NGraphAssign";

      TF_RETURN_IF_ERROR(NodeBuilder(new_name_ngassign, "NGraphAssign")
                             .Attr("validate_shape", true)
                             .Attr("use_locking", true)
                             .Attr("T", dtype)
                             .Attr("ngraph_graph_id", 0)
                             .Input(input_var)
                             .Input(ndef_sub_op)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &ngraphassign_op));
      ngraphassign_op->set_assigned_device_name(node->assigned_device_name());
      NGRAPH_VLOG(1) << "Assign op name: " << ngraphassign_op->name();
      NGRAPH_VLOG(1) << "Assign op assigned device: "
                     << ngraphassign_op->assigned_device_name();

      TF_RETURN_IF_ERROR(ReplaceInputControlEdges(graph, node, mul_op));
      TF_RETURN_IF_ERROR(ReplaceOutputEdges(graph, node, ngraphassign_op));

      remove_nodes.push_back(node);

      NGRAPH_VLOG(1) << "Replaced ApplyGradientDescent";
    }  // Apply Gradient Descent
    else if (node->type_string() == "NGraphApplyMomentum") {
      std::vector<const Edge*> input_edges;
      TF_RETURN_IF_ERROR(node->input_edges(&input_edges));

      NGRAPH_VLOG(1) << "No of input edges to ApplyMomentum "
                     << input_edges.size();

      auto input_var = NodeBuilder::NodeOut(input_edges[0]->src(),
                                            input_edges[0]->src_output());
      auto input_accum = NodeBuilder::NodeOut(input_edges[1]->src(),
                                              input_edges[1]->src_output());
      auto input_lr = NodeBuilder::NodeOut(input_edges[2]->src(),
                                           input_edges[2]->src_output());
      auto input_grad = NodeBuilder::NodeOut(input_edges[3]->src(),
                                             input_edges[3]->src_output());
      auto input_momentum = NodeBuilder::NodeOut(input_edges[4]->src(),
                                                 input_edges[4]->src_output());
      // Computing accumulator accum.device(d) = accum * momentum() + grad;
      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

      string new_name_mul = node->name() + "_Mul";
      NodeBuilder::NodeOut ndef_mul_op;
      TF_RETURN_IF_ERROR(CreateBinaryOpNode(&ndef_mul_op, new_name_mul, "Mul",
                                            input_accum, input_momentum, dtype,
                                            node, graph));
      TF_RETURN_IF_ERROR(
          ReplaceInputControlEdges(graph, node, ndef_mul_op.node));
      string new_name_add = node->name() + "_Add";
      NodeBuilder::NodeOut ndef_add_op;
      TF_RETURN_IF_ERROR(CreateBinaryOpNode(&ndef_add_op, new_name_add, "Add",
                                            input_grad, ndef_mul_op, dtype,
                                            node, graph));

      Node* accumassign_op;
      string new_name_accumassign = node->name() + "_AccumAssign";

      TF_RETURN_IF_ERROR(NodeBuilder(new_name_accumassign, "NGraphAssign")
                             .Attr("validate_shape", true)
                             .Attr("use_locking", true)
                             .Attr("T", dtype)
                             .Attr("ngraph_graph_id", 0)
                             .Input(input_accum)
                             .Input(ndef_add_op)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &accumassign_op));
      accumassign_op->set_assigned_device_name(node->assigned_device_name());
      NodeBuilder::NodeOut ndef_accumassign_op =
          NodeBuilder::NodeOut(accumassign_op, 0);

      string new_name_mul_1 = node->name() + "_Mul1";
      NodeBuilder::NodeOut ndef_mul_op_1;
      TF_RETURN_IF_ERROR(CreateBinaryOpNode(&ndef_mul_op_1, new_name_mul_1,
                                            "Mul", ndef_accumassign_op,
                                            input_lr, dtype, node, graph));

      bool use_nesterov;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(node->attrs(), "use_nesterov", &use_nesterov));
      Node* ngraphassign_op;
      if (use_nesterov) {
        // Equation in this case:
        //  var.device(d) - = grad * lr() + accum * momentum() * lr();

        string new_name_mul_2 = node->name() + "_Mul2";
        NodeBuilder::NodeOut ndef_mul_op_2;
        TF_RETURN_IF_ERROR(CreateBinaryOpNode(&ndef_mul_op_2, new_name_mul_2,
                                              "Mul", input_grad, input_lr,
                                              dtype, node, graph));

        string new_name_mul_3 = node->name() + "_Mul3";
        NodeBuilder::NodeOut ndef_mul_op_3;
        TF_RETURN_IF_ERROR(CreateBinaryOpNode(
            &ndef_mul_op_3, new_name_mul_3, "Mul", ndef_mul_op_1,
            input_momentum, dtype, node, graph));

        string new_name_add_1 = node->name() + "_Add_1";
        NodeBuilder::NodeOut ndef_add_op_1;
        TF_RETURN_IF_ERROR(CreateBinaryOpNode(
            &ndef_add_op_1, new_name_add_1, "Add", ndef_mul_op_2, ndef_mul_op_3,
            dtype, node, graph));

        string new_name_sub = node->name() + "_Sub";
        NodeBuilder::NodeOut ndef_sub_op;
        TF_RETURN_IF_ERROR(CreateBinaryOpNode(&ndef_sub_op, new_name_sub, "Sub",
                                              input_var, ndef_add_op_1, dtype,
                                              node, graph));

        string new_name_ngraphassign = node->name() + "_NGraphAssign";
        TF_RETURN_IF_ERROR(NodeBuilder(new_name_ngraphassign, "NGraphAssign")
                               .Attr("validate_shape", true)
                               .Attr("use_locking", true)
                               .Attr("T", dtype)
                               .Attr("ngraph_graph_id", 0)
                               .Input(input_var)
                               .Input(ndef_sub_op)
                               .Device(node->assigned_device_name())
                               .Finalize(graph, &ngraphassign_op));
        ngraphassign_op->set_assigned_device_name(node->assigned_device_name());
      }  // use_nesterov
      else {
        // Equation in this case:
        // var.device(d) -= accum * lr();

        string new_name_sub = node->name() + "_Sub";
        NodeBuilder::NodeOut ndef_sub_op;
        TF_RETURN_IF_ERROR(CreateBinaryOpNode(&ndef_sub_op, new_name_sub, "Sub",
                                              input_var, ndef_mul_op_1, dtype,
                                              node, graph));

        string new_name_ngraphassign = node->name() + "_NGraphAssign";
        TF_RETURN_IF_ERROR(NodeBuilder(new_name_ngraphassign, "NGraphAssign")
                               .Attr("validate_shape", true)
                               .Attr("use_locking", true)
                               .Attr("T", dtype)
                               .Attr("ngraph_graph_id", 0)
                               .Input(input_var)
                               .Input(ndef_sub_op)
                               .Device(node->assigned_device_name())
                               .Finalize(graph, &ngraphassign_op));
        ngraphassign_op->set_assigned_device_name(node->assigned_device_name());
      }  // else

      NGRAPH_VLOG(1) << "Assign op name: " << ngraphassign_op->name();
      NGRAPH_VLOG(1) << "Assign op assigned device: "
                     << ngraphassign_op->assigned_device_name();
      TF_RETURN_IF_ERROR(ReplaceOutputEdges(graph, node, ngraphassign_op));

      remove_nodes.push_back(node);

      NGRAPH_VLOG(1) << "Replaced ApplyMomentum";
    }
  }

  for (auto node : remove_nodes) {
    graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
