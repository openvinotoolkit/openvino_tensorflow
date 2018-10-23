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
#include "ngraph_mark_for_clustering.h"
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
// of the op can be placed on nGraph, and returns "true" if placement is
// allowed. This is followed by checks for deadness and input datatype of the
// op.

// Each op that passes all the checks, has the attribute
// "_ngraph_marked_for_clustering" set to "true". Additional metadata (Static
// Inputs) for the op is also set.

using ConfirmationFunction = std::function<Status(Node*, bool*)>;
using TypeConstraintMap =
    std::map<std::string, std::map<std::string, gtl::ArraySlice<DataType>>>;
using SetAttributesFunction = std::function<Status(Node*)>;

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

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
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
    std::map<std::string, ConfirmationFunction>& confirmation_function_map,
    bool& confirmation_ok) {
  auto it = confirmation_function_map.find(node->type_string());

  if (it != confirmation_function_map.end()) {
    TF_RETURN_IF_ERROR(it->second(node, &confirmation_ok));
  }
  return Status::OK();
}

//
// Marks the input indices in "inputs" as static
static inline void SetStaticInputs(Node* n, std::vector<int32> inputs) {
  n->AddAttr("_ngraph_static_inputs", inputs);
}

// Marks the input indices given in static_input_indices as static, i.e., inputs
// that must be driven either by an _Arg or by a Const in the encapsulated
// graph (meaning that its value must be known at translation-to-nGraph time). A
// negative value in static_input_indices indicates that the input index is
// counted from the right.
static SetAttributesFunction SetStaticInputs(
    const std::vector<int32>& static_input_indices = {}) {
  auto cf = [static_input_indices](Node* n) {
    // Adjust negative input indices.
    auto indices = static_input_indices;
    std::transform(indices.begin(), indices.end(), indices.begin(),
                   [n](int x) { return x >= 0 ? x : n->num_inputs() + x; });
    SetStaticInputs(n, indices);
    return Status::OK();
  };
  return cf;
};

// Generates a "simple" confirmation function which always returns true,
static ConfirmationFunction SimpleConfirmationFunction() {
  auto cf = [](Node* n, bool* result) {
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
  // used to check arbitrary constraints. For example:
  //
  //    confirmation_function_map["MyOp"] = [](Node* n, bool* confirmed) {
  //      int dummy;
  //      if (GetAttr(n->attrs(),"my_unsupported_attr",&dummy).ok()) {
  //        *confirmed = false;
  //        return Status::OK();
  //      }
  //      *confirmed = true;
  //      return Status::OK();
  //    };
  //
  // The foregoing function checks every "MyOp" node to make sure that it does
  // not have the attribute "my_unsupported_attr", and rejects placement if it
  // does.

  static std::map<std::string, ConfirmationFunction> confirmation_function_map;

  //
  // A map of op types (e.g. "Add") to set_attribute functions. These can be
  // used to set any additional attributes. For example:
  //
  //    confirmation_function_map["MyOp"] = [](Node* n) {
  //     if(n->condition()){
  //        int dummy=5;
  //        n->AddAttr("_ngraph_dummy_attr", dummy);
  //      }
  //
  //      vector<int32> static_input_index =5;
  //      n->AddAttr("_ngraph_static_inputs", static_input_index);
  //      return Status::OK();
  //    };
  //

  static std::map<std::string, SetAttributesFunction> set_attributes_map;

  mutex init_mu;
  static bool initialized = false;

  // If the type constraint and confirmation function maps have not been
  // initialized, initialize them.
  //
  // IF YOU ARE ADDING A NEW OP IMPLEMENTATION, YOU MUST ADD A CONFIRMATION
  // FUNCTION, TYPE CONTRAINTS (IF ANY) AND STATIC INPUTS INDEXES (IF ANY) FOR
  // THE OP HERE.

  // The constraint function should refuse placement if the node is not
  // supported in the builder, and tag the node with any data that will be
  // needed in case the graph is broken up in a later rewrite pass (for example,
  // constant data).

  {
    mutex_lock l(init_mu);

    if (!initialized) {
      //
      // Initialize confirmation function map.
      //
      // Please keep these in alphabetical order by op name.
      //
      confirmation_function_map["Abs"] = SimpleConfirmationFunction();
      confirmation_function_map["Add"] = SimpleConfirmationFunction();
      confirmation_function_map["AddN"] = SimpleConfirmationFunction();
      confirmation_function_map["Any"] = SimpleConfirmationFunction();
      confirmation_function_map["All"] = SimpleConfirmationFunction();
      confirmation_function_map["ArgMax"] = SimpleConfirmationFunction();
      confirmation_function_map["ArgMin"] = SimpleConfirmationFunction();
      confirmation_function_map["AvgPool"] = SimpleConfirmationFunction();
      confirmation_function_map["AvgPoolGrad"] = SimpleConfirmationFunction();
      confirmation_function_map["BatchMatMul"] = SimpleConfirmationFunction();
      confirmation_function_map["BiasAdd"] = SimpleConfirmationFunction();
      confirmation_function_map["BiasAddGrad"] = SimpleConfirmationFunction();
      confirmation_function_map["Cast"] = SimpleConfirmationFunction();
      confirmation_function_map["ConcatV2"] = SimpleConfirmationFunction();
      confirmation_function_map["Const"] = SimpleConfirmationFunction();
      confirmation_function_map["Conv2D"] = SimpleConfirmationFunction();
      confirmation_function_map["Conv2DBackpropFilter"] =
          SimpleConfirmationFunction();
      confirmation_function_map["Conv2DBackpropInput"] =
          SimpleConfirmationFunction();
      confirmation_function_map["DepthwiseConv2dNative"] =
          SimpleConfirmationFunction();
      confirmation_function_map["Dequantize"] = SimpleConfirmationFunction();
      confirmation_function_map["Equal"] = SimpleConfirmationFunction();
      confirmation_function_map["Exp"] = SimpleConfirmationFunction();
      confirmation_function_map["ExpandDims"] = SimpleConfirmationFunction();
      confirmation_function_map["Fill"] = SimpleConfirmationFunction();
      confirmation_function_map["Floor"] = SimpleConfirmationFunction();
      confirmation_function_map["FloorDiv"] = SimpleConfirmationFunction();
      confirmation_function_map["FloorMod"] = SimpleConfirmationFunction();
      confirmation_function_map["FusedBatchNorm"] =
          SimpleConfirmationFunction();
      confirmation_function_map["FusedBatchNormGrad"] = [](Node* n,
                                                           bool* result) {
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "is_training", result));
        return Status::OK();
      };
      confirmation_function_map["Greater"] = SimpleConfirmationFunction();
      confirmation_function_map["GreaterEqual"] = SimpleConfirmationFunction();
      confirmation_function_map["HorovodAllreduce"] =
          SimpleConfirmationFunction();
      confirmation_function_map["Identity"] = SimpleConfirmationFunction();
      confirmation_function_map["L2Loss"] = SimpleConfirmationFunction();
      confirmation_function_map["Less"] = SimpleConfirmationFunction();
      confirmation_function_map["LessEqual"] = SimpleConfirmationFunction();
      confirmation_function_map["Log"] = SimpleConfirmationFunction();
      confirmation_function_map["LogicalAnd"] = SimpleConfirmationFunction();
      confirmation_function_map["LogicalNot"] = SimpleConfirmationFunction();
      confirmation_function_map["LogicalOr"] = SimpleConfirmationFunction();
      confirmation_function_map["MatMul"] = SimpleConfirmationFunction();
      confirmation_function_map["Max"] = SimpleConfirmationFunction();
      confirmation_function_map["Maximum"] = SimpleConfirmationFunction();
      confirmation_function_map["MaxPool"] = SimpleConfirmationFunction();
      confirmation_function_map["MaxPoolGrad"] = SimpleConfirmationFunction();
      confirmation_function_map["Mean"] = SimpleConfirmationFunction();
      confirmation_function_map["Min"] = SimpleConfirmationFunction();
      confirmation_function_map["Minimum"] = SimpleConfirmationFunction();
      confirmation_function_map["Mul"] = SimpleConfirmationFunction();
      confirmation_function_map["Neg"] = SimpleConfirmationFunction();
      confirmation_function_map["Pad"] = SimpleConfirmationFunction();
      confirmation_function_map["Pow"] = SimpleConfirmationFunction();
      confirmation_function_map["PreventGradient"] =
          SimpleConfirmationFunction();
      confirmation_function_map["Prod"] = SimpleConfirmationFunction();
      confirmation_function_map["QuantizeAndDequantizeV2"] = [](Node* n,
                                                                bool* result) {
        // accept only when num_bits == 8 and range is given
        bool range_given;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(n->attrs(), "range_given", &range_given));
        int num_bits;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "num_bits", &num_bits));
        *result = (num_bits == 8) && range_given;
        return Status::OK();
      };
      confirmation_function_map["QuantizeV2"] = [](Node* n, bool* result) {
        string mode;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "mode", &mode));
        *result = (mode.compare("SCALED") == 0);
        return Status::OK();
      };
      confirmation_function_map["RealDiv"] = SimpleConfirmationFunction();
      confirmation_function_map["Reciprocal"] = SimpleConfirmationFunction();
      confirmation_function_map["Relu"] = SimpleConfirmationFunction();
      confirmation_function_map["Relu6"] = SimpleConfirmationFunction();
      confirmation_function_map["ReluGrad"] = SimpleConfirmationFunction();
      confirmation_function_map["Reshape"] = SimpleConfirmationFunction();
      confirmation_function_map["Rsqrt"] = SimpleConfirmationFunction();
      confirmation_function_map["Shape"] = SimpleConfirmationFunction();
      confirmation_function_map["Sigmoid"] = SimpleConfirmationFunction();
      confirmation_function_map["SigmoidGrad"] = SimpleConfirmationFunction();
      confirmation_function_map["Sign"] = SimpleConfirmationFunction();
      confirmation_function_map["Size"] = SimpleConfirmationFunction();
      confirmation_function_map["Slice"] = SimpleConfirmationFunction();
      confirmation_function_map["Snapshot"] = SimpleConfirmationFunction();
      confirmation_function_map["Softmax"] = SimpleConfirmationFunction();
      confirmation_function_map["SparseSoftmaxCrossEntropyWithLogits"] =
          SimpleConfirmationFunction();
      confirmation_function_map["Split"] = SimpleConfirmationFunction();
      confirmation_function_map["SplitV"] = SimpleConfirmationFunction();
      confirmation_function_map["Square"] = SimpleConfirmationFunction();
      confirmation_function_map["SquaredDifference"] =
          SimpleConfirmationFunction();
      confirmation_function_map["Squeeze"] = SimpleConfirmationFunction();
      confirmation_function_map["StridedSlice"] = [](Node* n, bool* result) {
        // Reject if "new_axis_mask" is set.
        int tf_new_axis_mask;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(n->attrs(), "new_axis_mask", &tf_new_axis_mask));
        if (tf_new_axis_mask != 0) {
          *result = false;
        } else {
          *result = true;
        }
        return Status::OK();
      };
      confirmation_function_map["Pack"] = SimpleConfirmationFunction();
      confirmation_function_map["Sub"] = SimpleConfirmationFunction();
      confirmation_function_map["Sum"] = SimpleConfirmationFunction();
      confirmation_function_map["Tanh"] = SimpleConfirmationFunction();
      confirmation_function_map["TanhGrad"] = SimpleConfirmationFunction();
      confirmation_function_map["Tile"] = SimpleConfirmationFunction();
      confirmation_function_map["Transpose"] = SimpleConfirmationFunction();
      confirmation_function_map["Unpack"] = SimpleConfirmationFunction();
      confirmation_function_map["ZerosLike"] = SimpleConfirmationFunction();

      //
      // Initialize type constraint map.
      //
      type_constraint_map["Abs"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Add"]["T"] = NGraphNumericDTypes();
      type_constraint_map["AddN"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Any"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["All"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["ArgMax"]["T"] = NGraphNumericDTypes();
      type_constraint_map["ArgMax"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["ArgMin"]["T"] = NGraphNumericDTypes();
      type_constraint_map["ArgMin"]["Tidx"] = NGraphIndexDTypes();
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
      type_constraint_map["Dequantize"]["T"] = NGraphSupportedQuantizedDTypes();
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
      type_constraint_map["HorovodAllreduce"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Identity"]["T"] = NGraphDTypes();
      type_constraint_map["L2Loss"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Less"]["T"] = NGraphDTypes();
      type_constraint_map["LessEqual"]["T"] = NGraphDTypes();
      type_constraint_map["Log"]["T"] = NGraphNumericDTypes();
      // LogicalAnd and LogicalNot have no type attributes ("T", if it existed,
      // would always be bool).
      type_constraint_map["MatMul"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Max"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Max"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["Maximum"]["T"] = NGraphNumericDTypes();
      type_constraint_map["MaxPool"]["T"] = NGraphNumericDTypes();
      type_constraint_map["MaxPoolGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Mean"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Mean"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["Min"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Min"]["Tidx"] = NGraphIndexDTypes();
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
      type_constraint_map["QuantizeAndDequantizeV2"]["T"] = NGraphRealDTypes();
      type_constraint_map["QuantizeV2"]["T"] = NGraphSupportedQuantizedDTypes();
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
      type_constraint_map["SigmoidGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Sign"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Size"]["T"] = NGraphDTypes();
      type_constraint_map["Size"]["out_type"] = NGraphIndexDTypes();
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
      type_constraint_map["TanhGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Tile"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Tile"]["Tmultiples"] = NGraphIndexDTypes();
      type_constraint_map["Transpose"]["T"] = NGraphDTypes();
      type_constraint_map["Transpose"]["Tperm"] = NGraphIndexDTypes();
      type_constraint_map["Unpack"]["T"] = NGraphDTypes();

      // Set Additional Attributes (if any)
      set_attributes_map["Any"] = SetStaticInputs({1});
      set_attributes_map["All"] = SetStaticInputs({1});
      set_attributes_map["ArgMax"] = SetStaticInputs({1});
      set_attributes_map["ArgMin"] = SetStaticInputs({1});
      set_attributes_map["AvgPoolGrad"] = SetStaticInputs({0});
      set_attributes_map["ConcatV2"] = SetStaticInputs({-1});
      set_attributes_map["Conv2DBackpropFilter"] = SetStaticInputs({1});
      set_attributes_map["Conv2DBackpropInput"] = SetStaticInputs({0});
      set_attributes_map["Dequantize"] = SetStaticInputs({1, 2});
      set_attributes_map["ExpandDims"] = SetStaticInputs({1});
      set_attributes_map["Fill"] = SetStaticInputs({0});
      set_attributes_map["Max"] = SetStaticInputs({1});
      set_attributes_map["Mean"] = SetStaticInputs({1});
      set_attributes_map["Min"] = SetStaticInputs({1});
      set_attributes_map["Pad"] = SetStaticInputs({1});
      set_attributes_map["Prod"] = SetStaticInputs({1});
      set_attributes_map["QuantizeAndDequantizeV2"] = SetStaticInputs({1, 2});
      set_attributes_map["QuantizeV2"] = SetStaticInputs({1, 2});
      set_attributes_map["Reshape"] = SetStaticInputs({1});
      set_attributes_map["Slice"] = SetStaticInputs({1, 2});
      set_attributes_map["Split"] = SetStaticInputs({0});
      set_attributes_map["SplitV"] = SetStaticInputs({1, 2});
      set_attributes_map["StridedSlice"] = SetStaticInputs({1, 2, 3});
      set_attributes_map["Sum"] = SetStaticInputs({1});
      set_attributes_map["Tile"] = SetStaticInputs({1});
      set_attributes_map["Transpose"] = SetStaticInputs({1});

      initialized = true;
    }
  }

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
  std::unique_ptr<DeadnessAnalysis> deadness_analyzer;
  TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(*graph, &deadness_analyzer));
#endif

  vector<Node*> nodes_marked_for_clustering;
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

#if !defined(NGRAPH_TF_DISABLE_DEADNESS_CHECK)
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

      // check node's confirmation constraints
      bool confirmation_constraint_ok = false;
      TF_RETURN_IF_ERROR(ConfirmationOk(node, confirmation_function_map,
                                        confirmation_constraint_ok));
      if (!confirmation_constraint_ok) {
        NGRAPH_VLOG(5) << "Node does not meet confirmation constraints: "
                       << node->name();
        break;
      }

      // check input type constraints
      bool type_constraint_ok = false;
      TF_RETURN_IF_ERROR(
          TypeConstraintOk(node, type_constraint_map, type_constraint_ok));
      if (!type_constraint_ok) {
        NGRAPH_VLOG(5) << "Inputs do not meet type constraints: "
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
      nodes_marked_for_clustering.push_back(node);
    } else {
      NGRAPH_VLOG(4) << "Rejecting: " << node->name() << "["
                     << node->type_string() << "]";
    }
  }

  // Set Attributes for nodes marked for clustering
  // 1. Set Attribute "_ngraph_marked_for_clustering" as "true"
  // 2. Set any other attributes as defined in set_attribute_map
  for (auto node : nodes_marked_for_clustering) {
    // TODO(amprocte): move attr name to a constant
    node->AddAttr("_ngraph_marked_for_clustering", true);

    auto it = set_attributes_map.find(node->type_string());
    if (it != set_attributes_map.end()) {
      TF_RETURN_IF_ERROR(it->second(node));
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
