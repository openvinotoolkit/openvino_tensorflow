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

#include "ngraph_api.h"
#include "ngraph_utils.h"
#include "ngraph_version_utils.h"
#include "tensorflow/core/graph/graph.h"
#include "tf_deadness_analysis.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// The "marking" pass checks every node with requested placement on nGraph,
// and either rejects the placement request, or tags it with suitable metadata.
//
// For now we assume that every node has nGraph placement requested, unless the
// environment variable NGRAPH_TF_DISABLE is set. (TODO(amprocte): implement
// something better.)
//
// Each TensorFlow op supported by nGraph has a "confirmation function"
// associated with it. When the confirmation pass encounters a node of op "Op",
// the confirmation function for "Op" first checks if this particular instance
// of the op can be placed on nGraph, possibly attaching extra metadata to the
// node for later use, and returns "true" if placement is allowed. Every
// confirmed op has the attribute "_ngraph_marked_for_clustering" set to
// "true".
//
// See the body of "MarkForClustering" for more details on what a "confirmation
// function" does.
//

using ConfirmationFunction = std::function<Status(Node*, bool*)>;
using TypeConstraintMap =
    std::map<std::string, std::map<std::string, gtl::ArraySlice<DataType>>>;

// Different Checks before we mark for clustering
//
// Utility function to check if placement on the NGRAPH device has been
// requested.
//
// FIXME(amprocte): stubbed out for now because NGRAPH device is gone.
//
static Status NGraphPlacementRequested(Node* node, bool& placement_ok) {
  placement_ok = true;
  return Status::OK();
}

#if (TF_VERSION_GEQ_1_11)
// Checks if the node's inputs have mismatching deadness
static Status DeadnessOk(Node* node,
                         std::unique_ptr<DeadnessAnalysis>* deadness_analyzer,
                         bool& deadness_ok) {
  deadness_ok =
      !(node->IsMerge() ||
        (*deadness_analyzer)->HasInputsWithMismatchingDeadness(*node));
  return Status::OK();
}
#endif

// Checks if the node's inputs meet all the type constraints
static Status TypeConstraintOk(Node* node,
                               TypeConstraintMap& type_constraint_map,
                               bool& type_constraints_ok) {
  type_constraints_ok = true;
  for (auto& name_and_set : type_constraint_map[node->type_string()]) {
    auto& type_attr_name = name_and_set.first;
    auto& allowed_types = name_and_set.second;

    DataType dt;

    if (GetNodeAttr(node->attrs(), type_attr_name, &dt) != Status::OK() ||
        std::find(allowed_types.begin(), allowed_types.end(), dt) ==
            allowed_types.end()) {
      type_constraints_ok = false;
      break;
    }
  }
  return Status::OK();
}

// Checks if the node meets the confirmation constraints
static Status ConfirmationOk(
    Node* node,
    std::map<std::string, ConfirmationFunction>& confirmation_functions,
    bool& confirmation_ok) {
  auto it = confirmation_functions.find(node->type_string());

  if (it != confirmation_functions.end()) {
    TF_RETURN_IF_ERROR(it->second(node, &confirmation_ok));
  }
  return Status::OK();
}

//
// Marks the input indices in "inputs" as static, i.e., inputs that must be
// driven either by an _Arg or by a Const in the encapsulated graph.
//
static inline void SetStaticInputs(Node* n, std::vector<int32> inputs) {
  n->AddAttr("_ngraph_static_inputs", inputs);
}

// Generates a "simple" confirmation function which always returns true, and
// tags the input indices given in static_input_indices as static. A negative
// value in static_input_indices indicates that the input index is counted from
// the right.
static ConfirmationFunction SimpleConfirmationFunction(
    const std::vector<int32>& static_input_indices = {}) {
  auto cf = [static_input_indices](Node* n, bool* result) {
    // Adjust negative input indices.
    auto indices = static_input_indices;
    std::transform(indices.begin(), indices.end(), indices.begin(),
                   [n](int x) { return x >= 0 ? x : n->num_inputs() + x; });
    SetStaticInputs(n, indices);
    *result = true;
    return Status::OK();
  };
  return cf;
};

//
// Main entry point for the marking pass.
//
Status MarkForClustering(Graph* graph) {
  if (config::IsEnabled() == false) {
    return Status::OK();
  }

  //
  // If NGRAPH_TF_DISABLE is set we will not mark anything; all subsequent
  // passes become a no-op.
  //
  if (std::getenv("NGRAPH_TF_DISABLE") != nullptr) {
    return Status::OK();
  }

  //
  // A map of op types (e.g. "Add") to type constraint maps. For (fake)
  // example:
  //
  //  type_constraint_map["Cast"]["SrcT"] = {DT_FLOAT, DT_BOOL};
  //  type_constraint_map["Cast"]["DstT"] = {DT_DOUBLE, DT_INT16};
  //
  // ...would mean that for the "Cast" op, the "SrcT" type variable can be
  // DT_FLOAT or DT_BOOL, and the "DstT" type variable can be DT_DOUBLE or
  // DT_INT16.
  //
  static TypeConstraintMap type_constraint_map;

  //
  // A map of op types (e.g. "Add") to confirmation functions. These can be
  // used to check arbitrary constraints, and attach information to the node
  // in the process. For example:
  //
  //    confirmation_functions["MyOp"] = [](Node* n, bool* confirmed) {
  //      int dummy;
  //      if (GetAttr(n->attrs(),"my_unsupported_attr",&dummy).ok()) {
  //        *confirmed = false;
  //        return Status::OK();
  //      }
  //
  //      SetStaticInputs(n, {0});
  //      *confirmed = true;
  //      return Status::OK();
  //    };
  //
  // The foregoing function checks every "MyOp" node to make sure that it does
  // not have the attribute "my_unsupported_attr", and rejects placement if it
  // does. Otherwise, it marks the zeroth input to the node as static (meaning
  // that its value must be known at translation-to-nGraph time, and accepts
  // placement.
  //
  static std::map<std::string, ConfirmationFunction> confirmation_functions;

  mutex init_mu;
  static bool initialized = false;

  // If the type constraint and confirmation function maps have not been
  // initialized, initialize them.
  //
  // IF YOU ARE ADDING A NEW OP IMPLEMENTATION, ADD TYPE CONSTRAINTS AND A
  // CONFIRMATION FUNCTION FOR THE OP HERE. The constraint function should
  // refuse placement if the node is not supported in the builder, and tag
  // the node with any data that will be needed in case the graph is broken
  // up in a later rewrite pass (for example, constant data).
  {
    mutex_lock l(init_mu);

    if (!initialized) {
      //
      // Initialize type constraint map.
      //
      type_constraint_map["Abs"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Add"]["T"] = NGraphNumericDTypes();
      type_constraint_map["AddN"]["T"] = NGraphNumericDTypes();
      type_constraint_map["AvgPool"]["T"] = NGraphNumericDTypes();
      type_constraint_map["AvgPoolGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["BatchMatMul"]["T"] = NGraphNumericDTypes();
      type_constraint_map["BiasAdd"]["T"] = NGraphNumericDTypes();
      type_constraint_map["BiasAddGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Cast"]["SrcT"] = NGraphDTypes();
      type_constraint_map["Cast"]["DstT"] = NGraphDTypes();
      type_constraint_map["ConcatV2"]["T"] = NGraphDTypes();
      type_constraint_map["ConcatV2"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["Const"]["dtype"] = NGraphDTypes();
      type_constraint_map["Conv2D"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Conv2DBackpropInput"]["T"] = NGraphNumericDTypes();
      type_constraint_map["DepthwiseConv2dNative"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Equal"]["T"] = NGraphDTypes();
      type_constraint_map["Exp"]["T"] = NGraphNumericDTypes();
      type_constraint_map["ExpandDims"]["T"] = NGraphDTypes();
      type_constraint_map["Floor"]["T"] = NGraphNumericDTypes();
      type_constraint_map["FloorDiv"]["T"] = NGraphNumericDTypes();
      type_constraint_map["FloorMod"]["T"] = NGraphNumericDTypes();
      type_constraint_map["FusedBatchNorm"]["T"] = NGraphNumericDTypes();
      type_constraint_map["FusedBatchNormGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Greater"]["T"] = NGraphDTypes();
      type_constraint_map["GreaterEqual"]["T"] = NGraphDTypes();
      type_constraint_map["Identity"]["T"] = NGraphDTypes();
      type_constraint_map["L2Loss"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Less"]["T"] = NGraphDTypes();
      type_constraint_map["LessEqual"]["T"] = NGraphDTypes();
      type_constraint_map["Log"]["T"] = NGraphNumericDTypes();
      // LogicalAnd and LogicalNot have no type attributes ("T", if it existed,
      // would always be bool).
      type_constraint_map["MatMul"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Maximum"]["T"] = NGraphNumericDTypes();
      type_constraint_map["MaxPool"]["T"] = NGraphNumericDTypes();
      type_constraint_map["MaxPoolGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Mean"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Mean"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["Minimum"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Mul"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Neg"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Pack"]["T"] = NGraphDTypes();
      type_constraint_map["Pad"]["T"] = NGraphDTypes();
      type_constraint_map["Pad"]["Tpaddings"] = NGraphIndexDTypes();
      type_constraint_map["Pow"]["T"] = NGraphNumericDTypes();
      type_constraint_map["PreventGradient"]["T"] = NGraphDTypes();
      type_constraint_map["Prod"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Prod"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["RealDiv"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Reciprocal"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Relu"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Relu6"]["T"] = NGraphNumericDTypes();
      type_constraint_map["ReluGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Reshape"]["T"] = NGraphDTypes();
      type_constraint_map["Reshape"]["Tshape"] = NGraphIndexDTypes();
      type_constraint_map["Rsqrt"]["T"] = NGraphDTypes();
      type_constraint_map["Shape"]["T"] = NGraphDTypes();
      type_constraint_map["Shape"]["out_type"] = NGraphIndexDTypes();
      type_constraint_map["Sigmoid"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Sign"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Slice"]["T"] = NGraphDTypes();
      type_constraint_map["Slice"]["Index"] = NGraphIndexDTypes();
      type_constraint_map["Snapshot"]["T"] = NGraphDTypes();
      type_constraint_map["Softmax"]["T"] = NGraphNumericDTypes();
      type_constraint_map["SparseSoftmaxCrossEntropyWithLogits"]["T"] =
          NGraphNumericDTypes();
      type_constraint_map["SparseSoftmaxCrossEntropyWithLogits"]["Tlabels"] =
          NGraphNumericDTypes();
      type_constraint_map["Split"]["T"] = NGraphDTypes();
      type_constraint_map["SplitV"]["T"] = NGraphDTypes();
      type_constraint_map["SplitV"]["Tlen"] = NGraphIndexDTypes();
      type_constraint_map["Square"]["T"] = NGraphDTypes();
      type_constraint_map["SquaredDifference"]["T"] = NGraphDTypes();
      type_constraint_map["Squeeze"]["T"] = NGraphDTypes();
      type_constraint_map["StridedSlice"]["T"] = NGraphDTypes();
      type_constraint_map["StridedSlice"]["Index"] = NGraphIndexDTypes();
      type_constraint_map["Sub"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Sum"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Sum"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["Tanh"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Tile"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Tile"]["Tmultiples"] = NGraphIndexDTypes();
      type_constraint_map["Transpose"]["T"] = NGraphDTypes();
      type_constraint_map["Transpose"]["Tperm"] = NGraphIndexDTypes();
      type_constraint_map["Unpack"]["T"] = NGraphDTypes();

      //
      // Initialize confirmation function map.
      //
      // Please keep these in alphabetical order by op name.
      //
      confirmation_functions["Abs"] = SimpleConfirmationFunction();
      confirmation_functions["Add"] = SimpleConfirmationFunction();
      confirmation_functions["AddN"] = SimpleConfirmationFunction();
      confirmation_functions["AvgPool"] = SimpleConfirmationFunction();
      confirmation_functions["AvgPoolGrad"] = SimpleConfirmationFunction({0});
      confirmation_functions["BatchMatMul"] = SimpleConfirmationFunction();
      confirmation_functions["BiasAdd"] = SimpleConfirmationFunction();
      confirmation_functions["BiasAddGrad"] = SimpleConfirmationFunction();
      confirmation_functions["Cast"] = SimpleConfirmationFunction();
      confirmation_functions["ConcatV2"] = SimpleConfirmationFunction({-1});
      confirmation_functions["Const"] = SimpleConfirmationFunction();
      confirmation_functions["Conv2D"] = SimpleConfirmationFunction();
      confirmation_functions["Conv2DBackpropFilter"] =
          SimpleConfirmationFunction({1});
      confirmation_functions["Conv2DBackpropInput"] =
          SimpleConfirmationFunction({0});
      confirmation_functions["DepthwiseConv2dNative"] =
          SimpleConfirmationFunction();
      confirmation_functions["Equal"] = SimpleConfirmationFunction();
      confirmation_functions["Exp"] = SimpleConfirmationFunction();
      confirmation_functions["ExpandDims"] = SimpleConfirmationFunction({1});
      confirmation_functions["Fill"] = SimpleConfirmationFunction({0});
      confirmation_functions["Floor"] = SimpleConfirmationFunction();
      confirmation_functions["FloorDiv"] = SimpleConfirmationFunction();
      confirmation_functions["FloorMod"] = SimpleConfirmationFunction();
      confirmation_functions["FusedBatchNorm"] = SimpleConfirmationFunction();
      confirmation_functions["FusedBatchNormGrad"] = [](Node* n, bool* result) {
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "is_training", result));
        return Status::OK();
      };
      confirmation_functions["Greater"] = SimpleConfirmationFunction();
      confirmation_functions["GreaterEqual"] = SimpleConfirmationFunction();
      confirmation_functions["Identity"] = SimpleConfirmationFunction();
      confirmation_functions["L2Loss"] = SimpleConfirmationFunction();
      confirmation_functions["Less"] = SimpleConfirmationFunction();
      confirmation_functions["LessEqual"] = SimpleConfirmationFunction();
      confirmation_functions["Log"] = SimpleConfirmationFunction();
      confirmation_functions["LogicalAnd"] = SimpleConfirmationFunction();
      confirmation_functions["LogicalNot"] = SimpleConfirmationFunction();
      confirmation_functions["MatMul"] = SimpleConfirmationFunction();
      confirmation_functions["Maximum"] = SimpleConfirmationFunction();
      confirmation_functions["MaxPool"] = SimpleConfirmationFunction();
      confirmation_functions["MaxPoolGrad"] = SimpleConfirmationFunction();
      confirmation_functions["Mean"] = SimpleConfirmationFunction({1});
      confirmation_functions["Minimum"] = SimpleConfirmationFunction();
      confirmation_functions["Mul"] = SimpleConfirmationFunction();
      confirmation_functions["Neg"] = SimpleConfirmationFunction();
      confirmation_functions["Pad"] = SimpleConfirmationFunction({1});
      confirmation_functions["Pow"] = SimpleConfirmationFunction();
      confirmation_functions["PreventGradient"] = SimpleConfirmationFunction();
      confirmation_functions["Prod"] = SimpleConfirmationFunction({1});
      confirmation_functions["RealDiv"] = SimpleConfirmationFunction();
      confirmation_functions["Reciprocal"] = SimpleConfirmationFunction();
      confirmation_functions["Relu"] = SimpleConfirmationFunction();
      confirmation_functions["Relu6"] = SimpleConfirmationFunction();
      confirmation_functions["ReluGrad"] = SimpleConfirmationFunction();
      confirmation_functions["Reshape"] = SimpleConfirmationFunction({1});
      confirmation_functions["Rsqrt"] = SimpleConfirmationFunction();
      confirmation_functions["Shape"] = SimpleConfirmationFunction();
      confirmation_functions["Sigmoid"] = SimpleConfirmationFunction();
      confirmation_functions["Sign"] = SimpleConfirmationFunction();
      confirmation_functions["Slice"] = SimpleConfirmationFunction({1, 2});
      confirmation_functions["Snapshot"] = SimpleConfirmationFunction();
      confirmation_functions["Softmax"] = SimpleConfirmationFunction();
      confirmation_functions["SparseSoftmaxCrossEntropyWithLogits"] =
          SimpleConfirmationFunction();
      confirmation_functions["Split"] = SimpleConfirmationFunction({0});
      confirmation_functions["SplitV"] = SimpleConfirmationFunction({1, 2});
      confirmation_functions["Square"] = SimpleConfirmationFunction();
      confirmation_functions["SquaredDifference"] =
          SimpleConfirmationFunction();
      confirmation_functions["Squeeze"] = SimpleConfirmationFunction();
      confirmation_functions["StridedSlice"] = [](Node* n, bool* result) {
        // Reject if "new_axis_mask" is set.
        int tf_new_axis_mask;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(n->attrs(), "new_axis_mask", &tf_new_axis_mask));
        if (tf_new_axis_mask != 0) {
          *result = false;
          return Status::OK();
        }

        return SimpleConfirmationFunction({1, 2, 3})(n, result);
      };
      confirmation_functions["Pack"] = SimpleConfirmationFunction();
      confirmation_functions["Sub"] = SimpleConfirmationFunction();
      confirmation_functions["Sum"] = SimpleConfirmationFunction({1});
      confirmation_functions["Tanh"] = SimpleConfirmationFunction();
      confirmation_functions["Tile"] = SimpleConfirmationFunction({1});
      confirmation_functions["Transpose"] = SimpleConfirmationFunction({1});
      confirmation_functions["Unpack"] = SimpleConfirmationFunction();

      initialized = true;
    }
  }

// If TF Version >= 1.11 do deadness analysis on the node
#if (TF_VERSION_GEQ_1_11)
  std::unique_ptr<DeadnessAnalysis> deadness_analyzer;
  TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(*graph, &deadness_analyzer));
#endif

  for (auto node : graph->op_nodes()) {
    bool mark_for_clustering = false;

    do {
      // check placement
      bool placement_ok = false;
      TF_RETURN_IF_ERROR(NGraphPlacementRequested(node, placement_ok));
      if (!placement_ok) {
        NGRAPH_VLOG(5) << "Placement not requested: " << node->name();
        break;
      }

#if (TF_VERSION_GEQ_1_11)
      // check deadness
      bool deadness_ok = false;
      TF_RETURN_IF_ERROR(DeadnessOk(node, &deadness_analyzer, deadness_ok));
      if (!deadness_ok) {
        NGRAPH_VLOG(5) << "Node Inputs have mismatching deadness or Node is of "
                          "type Merge: "
                       << node->name();
        break;
      }
#endif

      // check input type constraints
      bool type_constraint_ok = false;
      TF_RETURN_IF_ERROR(
          TypeConstraintOk(node, type_constraint_map, type_constraint_ok));
      if (!type_constraint_ok) {
        NGRAPH_VLOG(5) << "Inputs do not meet type constraints: "
                       << node->name();
        break;
      }

      // check node's confirmation constraints
      bool confirmation_constraint_ok = false;
      TF_RETURN_IF_ERROR(ConfirmationOk(node, confirmation_functions,
                                        confirmation_constraint_ok));
      if (!confirmation_constraint_ok) {
        NGRAPH_VLOG(5) << "Node does not meet confirmation constraints: "
                       << node->name();
        break;
      }

      // if all constraints are met, mark for clustering
      mark_for_clustering = true;
    } while (false);

    // Set the _ngraph_marked_for_clustering attribute if all constraints
    // are satisfied
    if (mark_for_clustering) {
      NGRAPH_VLOG(4) << "Accepting: " << node->name() << "["
                     << node->type_string() << "]";
      // TODO(amprocte): move attr name to a constant
      node->AddAttr("_ngraph_marked_for_clustering", true);
    } else {
      NGRAPH_VLOG(4) << "Rejecting: " << node->name() << "["
                     << node->type_string() << "]";
    }
  }

  return Status::OK();
}

bool NodeIsMarkedForClustering(const Node* node) {
  bool is_marked;
  // TODO(amprocte): move attr name to a constant
  return (GetNodeAttr(node->attrs(), "_ngraph_marked_for_clustering",
                      &is_marked) == Status::OK() &&
          is_marked);
}

void GetStaticInputs(const Node* node, std::vector<int32>* inputs) {
  if (GetNodeAttr(node->attrs(), "_ngraph_static_inputs", inputs) !=
      Status::OK()) {
    *inputs = std::vector<int32>{};
  }
}

bool InputIsStatic(const Node* node, int index) {
  std::vector<int32> inputs;
  GetStaticInputs(node, &inputs);
  return std::find(inputs.begin(), inputs.end(), index) != inputs.end();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
