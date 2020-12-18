/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

#include "api.h"
#include "backend_manager.h"
#include "ngraph_bridge/default_opset.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/ngraph_version_utils.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

static const gtl::ArraySlice<DataType>& NGraphDTypes() {
  static gtl::ArraySlice<DataType> result{
      DT_FLOAT, DT_INT8,   DT_INT16,   DT_INT32,  DT_INT64,
      DT_UINT8, DT_UINT16, DT_UINT32,  DT_UINT64, DT_BOOL,
      DT_QINT8, DT_QUINT8, DT_BFLOAT16};
  return result;
}

static const gtl::ArraySlice<DataType>& NGraphNumericDTypes() {
  static gtl::ArraySlice<DataType> result{
      DT_FLOAT, DT_INT8,   DT_INT16,  DT_INT32,  DT_INT64,
      DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BFLOAT16};
  return result;
}

static const gtl::ArraySlice<DataType>& NGraphIndexDTypes() {
  static gtl::ArraySlice<DataType> result{DT_INT32, DT_INT64};
  return result;
}

static const gtl::ArraySlice<DataType>& NGraphRealDTypes() {
  static gtl::ArraySlice<DataType> result{DT_FLOAT, DT_DOUBLE, DT_BFLOAT16};
  return result;
}

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
// allowed. This is followed by checks for input datatype of the
// op.

// Each op that passes all the checks, has the attribute
// "_ngraph_marked_for_clustering" set to "true". Additional metadata (Static
// Inputs) for the op is also set.

// Different Checks before we mark for clustering
//
// Utility function to check if placement on the NGRAPH device has been
// requested.
//
// FIXME(amprocte): stubbed out for now because NGRAPH device is gone.
//
static Status NGraphPlacementRequested(Node*, bool& placement_ok) {
  placement_ok = true;
  return Status::OK();
}

static Status CheckIfOutputNode(const Node* node,
                                const std::set<string> skip_these_nodes,
                                bool& skip_it) {
  skip_it = skip_these_nodes.find(node->name()) != skip_these_nodes.end();
  return Status::OK();
}

// Checks if the node's inputs meet all the type constraints
static Status TypeConstraintOk(Node* node,
                               const TypeConstraintMap& type_constraint_map,
                               bool& type_constraints_ok) {
  type_constraints_ok = true;
  const auto& itr = type_constraint_map.find(node->type_string());
  if (itr != type_constraint_map.end()) {
    for (const auto& name_and_set : itr->second) {
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
  auto cf = [](Node*, bool* result) {
    *result = true;
    return Status::OK();
  };
  return cf;
};

static ConfirmationFunction FusedBatchNormConfirmationFunction() {
  auto cf = [](Node* n, bool* result) {
    bool tf_is_training;
    if (GetNodeAttr(n->attrs(), "is_training", &tf_is_training) !=
        Status::OK()) {
      tf_is_training = true;
    }
    *result = !tf_is_training;
    return Status::OK();
  };
  return cf;
};

// Check if op is supported by backend using is_supported API
Status IsSupportedByBackend(
    const Node* node, const shared_ptr<Backend> op_backend,
    const std::map<std::string, std::set<shared_ptr<ngraph::Node>>>&
        TFtoNgraphOpMap,
    bool& is_supported) {
  is_supported = true;

  auto ng_op = TFtoNgraphOpMap.find(node->type_string());
  if (ng_op == TFtoNgraphOpMap.end()) {
    return errors::Internal("TF Op is not found in the map: ",
                            node->type_string());
  }

  // Loop through the ngraph op list to query
  for (auto it = ng_op->second.begin(); it != ng_op->second.end(); it++) {
    // Pass ngraph node to check if backend supports this op
    auto ret = op_backend->IsSupported(**it);
    if (!ret) {
      is_supported = false;
      return Status::OK();
    }
  }
  return Status::OK();
}

const std::map<std::string, SetAttributesFunction>& GetAttributeSetters() {
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
  static bool initialized = false;

  if (!initialized) {
    // Set Additional Attributes (if any)
    set_attributes_map["Any"] = SetStaticInputs({1});
    set_attributes_map["All"] = SetStaticInputs({1});
    set_attributes_map["ArgMax"] = SetStaticInputs({1});
    set_attributes_map["ArgMin"] = SetStaticInputs({1});
    set_attributes_map["ConcatV2"] = SetStaticInputs({-1});
    set_attributes_map["Conv2DBackpropInput"] = SetStaticInputs({0});
    set_attributes_map["ExpandDims"] = SetStaticInputs({1});
    set_attributes_map["GatherV2"] = SetStaticInputs({2});
    set_attributes_map["Max"] = SetStaticInputs({1});
    set_attributes_map["Mean"] = SetStaticInputs({1});
    set_attributes_map["Min"] = SetStaticInputs({1});
    set_attributes_map["MirrorPad"] = SetStaticInputs({1});
    set_attributes_map["NonMaxSuppressionV2"] = SetStaticInputs({2});
    set_attributes_map["OneHot"] = SetStaticInputs({1});
    set_attributes_map["Pad"] = SetStaticInputs({1});
    set_attributes_map["PadV2"] = SetStaticInputs({1});
    set_attributes_map["Prod"] = SetStaticInputs({1});
    set_attributes_map["Reshape"] = SetStaticInputs({1});
    set_attributes_map["Slice"] = SetStaticInputs({1, 2});
    set_attributes_map["Split"] = SetStaticInputs({0});
    set_attributes_map["SplitV"] = SetStaticInputs({1, 2});
    set_attributes_map["StridedSlice"] = SetStaticInputs({1, 2, 3});
    set_attributes_map["Sum"] = SetStaticInputs({1});
    set_attributes_map["TopKV2"] = SetStaticInputs({1});
    set_attributes_map["Tile"] = SetStaticInputs({1});
    set_attributes_map["Range"] = SetStaticInputs({0, 1, 2});
    initialized = true;
  }
  return set_attributes_map;
}

const std::map<std::string, ConfirmationFunction>& GetConfirmationMap() {
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
  static bool initialized = false;
  if (!initialized) {
    //
    // Initialize confirmation function map.
    //
    // Please keep these in alphabetical order by op name.
    //
    confirmation_function_map["Abs"] = SimpleConfirmationFunction();
    confirmation_function_map["Acos"] = SimpleConfirmationFunction();
    confirmation_function_map["Acosh"] = SimpleConfirmationFunction();
    confirmation_function_map["Add"] = SimpleConfirmationFunction();
    confirmation_function_map["AddN"] = SimpleConfirmationFunction();
    confirmation_function_map["AddV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Any"] = SimpleConfirmationFunction();
    confirmation_function_map["All"] = SimpleConfirmationFunction();
    confirmation_function_map["ArgMax"] = SimpleConfirmationFunction();
    confirmation_function_map["ArgMin"] = SimpleConfirmationFunction();
    confirmation_function_map["Asin"] = SimpleConfirmationFunction();
    confirmation_function_map["Asinh"] = SimpleConfirmationFunction();
    confirmation_function_map["Atan"] = SimpleConfirmationFunction();
    confirmation_function_map["Atanh"] = SimpleConfirmationFunction();
    confirmation_function_map["AvgPool"] = SimpleConfirmationFunction();
    confirmation_function_map["BiasAdd"] = SimpleConfirmationFunction();
    confirmation_function_map["Cast"] = SimpleConfirmationFunction();
    confirmation_function_map["Ceil"] = SimpleConfirmationFunction();
    confirmation_function_map["ConcatV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Const"] = SimpleConfirmationFunction();
    confirmation_function_map["Conv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["Conv2DBackpropInput"] =
        SimpleConfirmationFunction();
    confirmation_function_map["Conv3D"] = SimpleConfirmationFunction();
    confirmation_function_map["Cos"] = SimpleConfirmationFunction();
    confirmation_function_map["Cosh"] = SimpleConfirmationFunction();
    confirmation_function_map["Cumsum"] = SimpleConfirmationFunction();
    confirmation_function_map["DepthwiseConv2dNative"] =
        SimpleConfirmationFunction();
    confirmation_function_map["DepthToSpace"] = [](Node* n, bool* result) {
      std::string tf_data_format;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->attrs(), "data_format", &tf_data_format));
      *result = tf_data_format != "NCHW_VECT_C";
      return Status::OK();
    };
    confirmation_function_map["Equal"] = SimpleConfirmationFunction();
    confirmation_function_map["Exp"] = SimpleConfirmationFunction();
    confirmation_function_map["ExpandDims"] = SimpleConfirmationFunction();
    confirmation_function_map["Fill"] = SimpleConfirmationFunction();
    confirmation_function_map["Floor"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorDiv"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorMod"] = SimpleConfirmationFunction();
    confirmation_function_map["FusedBatchNorm"] =
        FusedBatchNormConfirmationFunction();
    confirmation_function_map["FusedBatchNormV2"] =
        FusedBatchNormConfirmationFunction();
    confirmation_function_map["FusedBatchNormV3"] =
        FusedBatchNormConfirmationFunction();
    confirmation_function_map["_FusedConv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["Gather"] = SimpleConfirmationFunction();
    confirmation_function_map["GatherV2"] = SimpleConfirmationFunction();
    confirmation_function_map["_FusedMatMul"] =
        SimpleConfirmationFunction();  // TODO accept under all conditions?
                                       // check?
    confirmation_function_map["Greater"] = SimpleConfirmationFunction();
    confirmation_function_map["GreaterEqual"] = SimpleConfirmationFunction();
    confirmation_function_map["Identity"] = SimpleConfirmationFunction();
    confirmation_function_map["IsFinite"] = SimpleConfirmationFunction();
    confirmation_function_map["L2Loss"] = SimpleConfirmationFunction();
    confirmation_function_map["LogSoftmax"] = SimpleConfirmationFunction();
    confirmation_function_map["Less"] = SimpleConfirmationFunction();
    confirmation_function_map["LessEqual"] = SimpleConfirmationFunction();
    confirmation_function_map["Log"] = SimpleConfirmationFunction();
    confirmation_function_map["Log1p"] = SimpleConfirmationFunction();
    confirmation_function_map["LogicalAnd"] = SimpleConfirmationFunction();
    confirmation_function_map["LogicalNot"] = SimpleConfirmationFunction();
    confirmation_function_map["LogicalOr"] = SimpleConfirmationFunction();
    confirmation_function_map["LRN"] = SimpleConfirmationFunction();
    confirmation_function_map["MatMul"] = SimpleConfirmationFunction();
    confirmation_function_map["Max"] = SimpleConfirmationFunction();
    confirmation_function_map["Maximum"] = SimpleConfirmationFunction();
    confirmation_function_map["MaxPool"] = SimpleConfirmationFunction();
    confirmation_function_map["MaxPool3D"] = SimpleConfirmationFunction();
    confirmation_function_map["Mean"] = SimpleConfirmationFunction();
    confirmation_function_map["Min"] = SimpleConfirmationFunction();
    confirmation_function_map["Minimum"] = SimpleConfirmationFunction();
    confirmation_function_map["MirrorPad"] = SimpleConfirmationFunction();
    confirmation_function_map["Mul"] = SimpleConfirmationFunction();
    confirmation_function_map["Mod"] = SimpleConfirmationFunction();
    confirmation_function_map["Neg"] = SimpleConfirmationFunction();
    confirmation_function_map["NotEqual"] = SimpleConfirmationFunction();
    confirmation_function_map["NonMaxSuppressionV2"] =
        SimpleConfirmationFunction();
    confirmation_function_map["NoOp"] = SimpleConfirmationFunction();
    confirmation_function_map["OneHot"] = SimpleConfirmationFunction();
    confirmation_function_map["Pad"] = SimpleConfirmationFunction();
    confirmation_function_map["PadV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Pow"] = SimpleConfirmationFunction();
    confirmation_function_map["PreventGradient"] = SimpleConfirmationFunction();
    confirmation_function_map["Prod"] = SimpleConfirmationFunction();
    confirmation_function_map["Range"] = SimpleConfirmationFunction();
    confirmation_function_map["Rank"] = SimpleConfirmationFunction();
    confirmation_function_map["RealDiv"] = SimpleConfirmationFunction();
    confirmation_function_map["Reciprocal"] = SimpleConfirmationFunction();
    confirmation_function_map["Relu"] = SimpleConfirmationFunction();
    confirmation_function_map["Relu6"] = SimpleConfirmationFunction();
    confirmation_function_map["Reshape"] = SimpleConfirmationFunction();
    confirmation_function_map["Rsqrt"] = SimpleConfirmationFunction();
    confirmation_function_map["Select"] = SimpleConfirmationFunction();
    confirmation_function_map["SelectV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Shape"] = SimpleConfirmationFunction();
    confirmation_function_map["Sigmoid"] = SimpleConfirmationFunction();
    confirmation_function_map["Sign"] = SimpleConfirmationFunction();
    confirmation_function_map["Sin"] = SimpleConfirmationFunction();
    confirmation_function_map["Sinh"] = SimpleConfirmationFunction();
    confirmation_function_map["Size"] = SimpleConfirmationFunction();
    confirmation_function_map["Slice"] = SimpleConfirmationFunction();
    confirmation_function_map["Snapshot"] = SimpleConfirmationFunction();
    confirmation_function_map["Softmax"] = SimpleConfirmationFunction();
    confirmation_function_map["Softplus"] = SimpleConfirmationFunction();
    confirmation_function_map["SpaceToDepth"] =
        confirmation_function_map["DepthToSpace"];
    confirmation_function_map["Split"] = SimpleConfirmationFunction();
    confirmation_function_map["SplitV"] = SimpleConfirmationFunction();
    confirmation_function_map["Sqrt"] = SimpleConfirmationFunction();
    confirmation_function_map["Square"] = SimpleConfirmationFunction();
    confirmation_function_map["SquaredDifference"] =
        SimpleConfirmationFunction();
    confirmation_function_map["Squeeze"] = SimpleConfirmationFunction();
    confirmation_function_map["StridedSlice"] = SimpleConfirmationFunction();
    confirmation_function_map["Pack"] = SimpleConfirmationFunction();
    confirmation_function_map["Sub"] = SimpleConfirmationFunction();
    confirmation_function_map["Sum"] = SimpleConfirmationFunction();
    confirmation_function_map["Tan"] = SimpleConfirmationFunction();
    confirmation_function_map["Tanh"] = SimpleConfirmationFunction();
    confirmation_function_map["Tile"] = SimpleConfirmationFunction();
    confirmation_function_map["TopKV2"] = [](Node* n, bool* result) {
      bool sorted = true;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "sorted", &sorted));

      // sorted = false is not supported right now, it falls back to TF if set
      // to false.
      *result = sorted;
      return Status::OK();
    };
    confirmation_function_map["Transpose"] = SimpleConfirmationFunction();
    confirmation_function_map["Unpack"] = SimpleConfirmationFunction();
    confirmation_function_map["Where"] = SimpleConfirmationFunction();
    confirmation_function_map["Xdivy"] = SimpleConfirmationFunction();
    confirmation_function_map["ZerosLike"] = SimpleConfirmationFunction();
    initialized = true;
  }
  return confirmation_function_map;
}

const TypeConstraintMap& GetTypeConstraintMap() {
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
  static bool initialized = false;
  static TypeConstraintMap type_constraint_map;
  if (!initialized) {
    //
    // Initialize type constraint map.
    //
    type_constraint_map["Abs"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Acos"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Acosh"]["T"] = NGraphRealDTypes();
    type_constraint_map["Add"]["T"] = NGraphNumericDTypes();
    type_constraint_map["AddN"]["T"] = NGraphNumericDTypes();
    type_constraint_map["AddV2"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Any"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["All"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["ArgMax"]["T"] = NGraphNumericDTypes();
    type_constraint_map["ArgMax"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["ArgMin"]["T"] = NGraphNumericDTypes();
    type_constraint_map["ArgMin"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Asin"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Asinh"]["T"] = NGraphRealDTypes();
    type_constraint_map["Atan"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Atanh"]["T"] = NGraphRealDTypes();
    type_constraint_map["AvgPool"]["T"] = NGraphNumericDTypes();
    type_constraint_map["BiasAdd"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Cast"]["SrcT"] = NGraphDTypes();
    type_constraint_map["Cast"]["DstT"] = NGraphDTypes();
    type_constraint_map["Ceil"]["T"] = NGraphRealDTypes();
    type_constraint_map["ConcatV2"]["T"] = NGraphDTypes();
    type_constraint_map["ConcatV2"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Const"]["dtype"] = NGraphDTypes();
    type_constraint_map["Conv2D"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Conv2DBackpropInput"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Conv3D"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Cos"]["T"] = NGraphRealDTypes();
    type_constraint_map["Cosh"]["T"] = NGraphRealDTypes();
    type_constraint_map["Cumsum"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Cumsum"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["DepthToSpace"]["T"] = NGraphDTypes();
    type_constraint_map["DepthwiseConv2dNative"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Equal"]["T"] = NGraphDTypes();
    type_constraint_map["Exp"]["T"] = NGraphNumericDTypes();
    type_constraint_map["ExpandDims"]["T"] = NGraphDTypes();
    type_constraint_map["Floor"]["T"] = NGraphNumericDTypes();
    type_constraint_map["FloorDiv"]["T"] = {DT_FLOAT};
    type_constraint_map["FloorMod"]["T"] = {DT_FLOAT};
    type_constraint_map["FusedBatchNorm"]["T"] = NGraphNumericDTypes();
    // TODO (mingshan): FusedBatchNormV2, V3 supports DT_HALF,DT_BFLOAT16,
    // DT_FLOAT
    type_constraint_map["FusedBatchNormV2"]["T"] = {DT_FLOAT};
    type_constraint_map["FusedBatchNormV3"]["T"] = {DT_FLOAT};
    type_constraint_map["Gather"]["Tparams"] = NGraphDTypes();
    type_constraint_map["Gather"]["Tindices"] = NGraphIndexDTypes();
    type_constraint_map["GatherV2"]["Tparams"] = NGraphDTypes();
    type_constraint_map["GatherV2"]["Tindices"] = NGraphIndexDTypes();
    type_constraint_map["GatherV2"]["Taxis"] = NGraphIndexDTypes();
    type_constraint_map["_FusedConv2D"]["T"] = NGraphRealDTypes();
    type_constraint_map["_FusedMatMul"]["T"] = NGraphRealDTypes();
    type_constraint_map["Greater"]["T"] = NGraphDTypes();
    type_constraint_map["GreaterEqual"]["T"] = NGraphDTypes();
    type_constraint_map["Identity"]["T"] = NGraphDTypes();
    type_constraint_map["IsFinite"]["T"] = NGraphRealDTypes();
    type_constraint_map["L2Loss"]["T"] = NGraphNumericDTypes();
    type_constraint_map["LogSoftmax"]["T"] = NGraphRealDTypes();
    type_constraint_map["Less"]["T"] = NGraphDTypes();
    type_constraint_map["LessEqual"]["T"] = NGraphDTypes();
    type_constraint_map["Log"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Log1p"]["T"] = NGraphRealDTypes();
    type_constraint_map["LRN"]["T"] = {DT_FLOAT};  // other supported types are
                                                   // DT_HALF & DT_BFLOAT16
                                                   // which are both not
                                                   // supported by IE
    // LogicalAnd and LogicalNot have no type attributes ("T", if it existed,
    // would always be bool).
    type_constraint_map["MatMul"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Max"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Max"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Maximum"]["T"] = NGraphNumericDTypes();
    type_constraint_map["MaxPool"]["T"] = NGraphNumericDTypes();
    type_constraint_map["MaxPool3D"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Mean"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Mean"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Min"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Min"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Minimum"]["T"] = NGraphNumericDTypes();
    type_constraint_map["MirrorPad"]["T"] = NGraphDTypes();
    type_constraint_map["MirrorPad"]["Tpaddings"] = NGraphIndexDTypes();
    type_constraint_map["Mul"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Mod"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Neg"]["T"] = NGraphNumericDTypes();
    type_constraint_map["NotEqual"]["T"] = NGraphDTypes();
    type_constraint_map["NonMaxSuppressionV2"]["T"] = {
        DT_FLOAT};  // TF allows half too
    type_constraint_map["OneHot"]["T"] = NGraphDTypes();
    type_constraint_map["Pack"]["T"] = NGraphDTypes();
    type_constraint_map["Pad"]["T"] = NGraphDTypes();
    type_constraint_map["Pad"]["Tpaddings"] = NGraphIndexDTypes();
    type_constraint_map["PadV2"]["T"] = NGraphDTypes();
    type_constraint_map["PadV2"]["Tpaddings"] = NGraphIndexDTypes();
    type_constraint_map["Pow"]["T"] = NGraphNumericDTypes();
    type_constraint_map["PreventGradient"]["T"] = NGraphDTypes();
    type_constraint_map["Prod"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Prod"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Range"]["Tidx"] = NGraphNumericDTypes();
    type_constraint_map["Rank"]["T"] = NGraphNumericDTypes();
    type_constraint_map["RealDiv"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Reciprocal"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Relu"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Relu6"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Reshape"]["T"] = NGraphDTypes();
    type_constraint_map["Reshape"]["Tshape"] = NGraphIndexDTypes();
    type_constraint_map["Rsqrt"]["T"] = NGraphDTypes();
    type_constraint_map["Select"]["T"] = NGraphDTypes();
    type_constraint_map["SelectV2"]["T"] = NGraphDTypes();
    type_constraint_map["Shape"]["T"] = NGraphDTypes();
    type_constraint_map["Shape"]["out_type"] = NGraphIndexDTypes();
    type_constraint_map["Sigmoid"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Sign"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Sin"]["T"] = NGraphRealDTypes();
    type_constraint_map["Sinh"]["T"] = NGraphRealDTypes();
    type_constraint_map["Size"]["T"] = NGraphDTypes();
    type_constraint_map["Size"]["out_type"] = NGraphIndexDTypes();
    type_constraint_map["Slice"]["T"] = NGraphDTypes();
    type_constraint_map["Slice"]["Index"] = NGraphIndexDTypes();
    type_constraint_map["Snapshot"]["T"] = NGraphDTypes();
    type_constraint_map["Softmax"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Softplus"]["T"] = NGraphRealDTypes();
    type_constraint_map["SpaceToDepth"]["T"] = NGraphDTypes();
    type_constraint_map["Split"]["T"] = NGraphDTypes();
    type_constraint_map["SplitV"]["T"] = NGraphDTypes();
    type_constraint_map["SplitV"]["Tlen"] = NGraphIndexDTypes();
    type_constraint_map["Sqrt"]["T"] = NGraphDTypes();
    type_constraint_map["Square"]["T"] = NGraphDTypes();
    type_constraint_map["SquaredDifference"]["T"] = NGraphDTypes();
    type_constraint_map["Squeeze"]["T"] = NGraphDTypes();
    type_constraint_map["StridedSlice"]["T"] = NGraphDTypes();
    type_constraint_map["StridedSlice"]["Index"] = NGraphIndexDTypes();
    type_constraint_map["Sub"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Sum"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Sum"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Tan"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Tanh"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Tile"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Tile"]["Tmultiples"] = NGraphIndexDTypes();
    type_constraint_map["TopKV2"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Transpose"]["T"] = NGraphDTypes();
    type_constraint_map["Transpose"]["Tperm"] = NGraphIndexDTypes();
    type_constraint_map["Unpack"]["T"] = NGraphDTypes();
    type_constraint_map["Where"]["T"] = NGraphDTypes();
    type_constraint_map["Xdivy"]["T"] = NGraphRealDTypes();
    type_constraint_map["ZerosLike"]["T"] = NGraphNumericDTypes();
    initialized = true;
  }
  return type_constraint_map;
}

const std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>&
GetTFToNgOpMap() {
  // Constant Op does not have default Constructor
  // in ngraph, so passing a dummy node
  auto constant =
      opset::Constant::create(ngraph::element::f32, ngraph::Shape{}, {2.0f});
  // Map:: TF ops to NG Ops to track if all the Ngraph ops
  // are supported by backend
  // Update this Map if a new TF Op translation is
  // implemented or a new Ngraph Op has been added
  static std::map<std::string, std::set<shared_ptr<ngraph::Node>>>
      TFtoNgraphOpMap{
          {"Abs", {std::make_shared<opset::Abs>()}},
          {"Acos", {std::make_shared<opset::Acos>()}},
          {"Acosh", {std::make_shared<opset::Acosh>()}},
          {"Add", {std::make_shared<opset::Add>()}},
          {"AddN", {std::make_shared<opset::Add>()}},
          {"AddV2", {std::make_shared<opset::Add>()}},
          {"Any", {std::make_shared<opset::ReduceLogicalOr>(), constant}},
          {"All", {std::make_shared<opset::ReduceLogicalAnd>(), constant}},
          {"ArgMax",
           {std::make_shared<opset::TopK>(), std::make_shared<opset::Squeeze>(),
            constant}},
          {"ArgMin",
           {std::make_shared<opset::TopK>(), std::make_shared<opset::Squeeze>(),
            constant}},
          {"Asin", {std::make_shared<opset::Asin>()}},
          {"Asinh", {std::make_shared<opset::Asinh>()}},
          {"Atan", {std::make_shared<opset::Atan>()}},
          {"Atanh", {std::make_shared<opset::Atanh>()}},
          {"AvgPool", {std::make_shared<opset::AvgPool>()}},
          {"BiasAdd",
           {constant, std::make_shared<opset::Add>(),
            std::make_shared<opset::Reshape>()}},
          {"Cast", {std::make_shared<opset::Convert>()}},
          {"Ceil", {std::make_shared<opset::Ceiling>()}},
          {"ConcatV2", {std::make_shared<opset::Concat>()}},
          {"Const", {constant}},
          {"Conv2D",
           {std::make_shared<opset::Transpose>(),
            std::make_shared<opset::Convolution>()}},
          {"Conv2DBackpropInput",
           {std::make_shared<opset::ConvolutionBackpropData>(),
            std::make_shared<opset::Transpose>(), constant}},
          {"Conv3D",
           {constant, std::make_shared<opset::Convolution>(),
            std::make_shared<opset::Transpose>()}},
          {"Cos", {std::make_shared<opset::Cos>()}},
          {"Cosh", {std::make_shared<opset::Cosh>()}},
          {"Cumsum", {std::make_shared<opset::CumSum>()}},
          {"DepthToSpace", {std::make_shared<opset::DepthToSpace>()}},
          {"DepthwiseConv2dNative",
           {std::make_shared<opset::GroupConvolution>(), constant}},
          {"Equal", {std::make_shared<opset::Equal>()}},
          {"Exp", {std::make_shared<opset::Exp>()}},
          {"ExpandDims", {std::make_shared<opset::Unsqueeze>()}},
          {"Fill", {constant, std::make_shared<opset::Broadcast>()}},
          {"Floor", {std::make_shared<opset::Floor>()}},
          {"FloorDiv",
           {std::make_shared<opset::Divide>(), std::make_shared<opset::Floor>(),
            std::make_shared<opset::Broadcast>()}},
          {"FloorMod", {std::make_shared<opset::FloorMod>()}},
          {"FusedBatchNorm", {std::make_shared<opset::BatchNormInference>()}},
          {"FusedBatchNormV2",
           {constant, std::make_shared<opset::BatchNormInference>(),
            std::make_shared<opset::Transpose>()}},
          {"FusedBatchNormV3",
           {constant, std::make_shared<opset::BatchNormInference>(),
            std::make_shared<opset::Transpose>()}},
          {"Gather", {constant, std::make_shared<opset::Gather>()}},
          {"GatherV2", {constant, std::make_shared<opset::Gather>()}},
          {"_FusedConv2D",
           {std::make_shared<opset::Convolution>(), constant,
            std::make_shared<opset::Minimum>(), std::make_shared<opset::Relu>(),
            std::make_shared<opset::Add>(),
            std::make_shared<opset::BatchNormInference>()}},
          {"_FusedMatMul",
           {std::make_shared<opset::MatMul>(), std::make_shared<opset::Relu>(),
            std::make_shared<opset::Add>(), constant,
            std::make_shared<opset::Minimum>()}},
          {"Greater", {std::make_shared<opset::Greater>()}},
          {"GreaterEqual", {std::make_shared<opset::GreaterEqual>()}},
          {"Identity", {}},
          {"IsFinite",
           {constant, std::make_shared<opset::NotEqual>(),
            std::make_shared<opset::Equal>(),
            std::make_shared<opset::LogicalAnd>()}},
          {"L2Loss",
           {constant, std::make_shared<opset::Multiply>(),
            std::make_shared<opset::ReduceSum>(),
            std::make_shared<opset::Divide>()}},
          {"LogSoftmax",
           {constant, std::make_shared<opset::Exp>(),
            std::make_shared<opset::ReduceMax>(),
            std::make_shared<opset::ReduceSum>(),
            std::make_shared<opset::Subtract>(),
            std::make_shared<opset::Log>()}},
          {"Less", {std::make_shared<opset::Less>()}},
          {"LessEqual", {std::make_shared<opset::LessEqual>()}},
          {"Log", {std::make_shared<opset::Log>()}},
          {"Log1p",
           {constant, std::make_shared<opset::Add>(),
            std::make_shared<opset::Log>()}},
          {"LogicalAnd", {std::make_shared<opset::LogicalAnd>()}},
          {"LogicalNot", {std::make_shared<opset::LogicalNot>()}},
          {"LogicalOr", {std::make_shared<opset::LogicalOr>()}},
          {"LRN", {std::make_shared<opset::LRN>()}},
          {"MatMul", {std::make_shared<opset::MatMul>()}},
          {"Max", {std::make_shared<opset::ReduceMax>(), constant}},
          {"Maximum", {std::make_shared<opset::Maximum>()}},
          {"MaxPool",
           {constant, std::make_shared<opset::Transpose>(),
            std::make_shared<opset::MaxPool>()}},
          {"MaxPool3D",
           {constant, std::make_shared<opset::Transpose>(),
            std::make_shared<opset::MaxPool>()}},
          {"Mean", {std::make_shared<opset::ReduceMean>(), constant}},
          {"Min", {std::make_shared<opset::ReduceMin>(), constant}},
          {"Minimum", {std::make_shared<opset::Minimum>()}},
          {"MirrorPad", {constant, std::make_shared<opset::Pad>()}},
          {"Mul", {std::make_shared<opset::Multiply>()}},
          {"Mod", {std::make_shared<opset::Mod>()}},
          {"Neg", {std::make_shared<opset::Negative>()}},
          {"NotEqual", {std::make_shared<opset::NotEqual>()}},
          {"NonMaxSuppressionV2",
           {std::make_shared<opset::NonMaxSuppression>(), constant,
            std::make_shared<opset::Unsqueeze>(),
            std::make_shared<opset::StridedSlice>()}},
          {"OneHot", {std::make_shared<opset::OneHot>(), constant}},
          {"Pack",
           {constant, std::make_shared<opset::Concat>(),
            std::make_shared<opset::Unsqueeze>()}},
          {"Pad", {constant, std::make_shared<opset::Pad>()}},
          {"PadV2", {constant, std::make_shared<opset::Pad>()}},
          {"Pow", {std::make_shared<opset::Power>()}},
          {"Prod", {std::make_shared<opset::ReduceProd>(), constant}},
          {"Range", {std::make_shared<opset::Range>()}},
          {"Rank", {constant}},
          {"RealDiv", {std::make_shared<opset::Divide>()}},
          {"Reciprocal", {constant, std::make_shared<opset::Power>()}},
          {"Relu", {std::make_shared<opset::Relu>()}},
          {"Relu6", {std::make_shared<opset::Clamp>()}},
          {"Rsqrt", {constant, std::make_shared<opset::Power>()}},
          {"Select", {std::make_shared<opset::Select>()}},
          {"SelectV2", {std::make_shared<opset::Select>()}},
          {"Reshape", {std::make_shared<opset::Reshape>()}},
          {"Shape", {std::make_shared<opset::ShapeOf>()}},
          {"Sigmoid", {std::make_shared<opset::Sigmoid>()}},
          {"Sin", {std::make_shared<opset::Sin>()}},
          {"Sinh", {std::make_shared<opset::Sinh>()}},
          {"Size", {constant}},
          {"Sign", {std::make_shared<opset::Sign>()}},
          {"Slice", {constant, std::make_shared<opset::StridedSlice>()}},
          {"Snapshot", {}},
          {"Softmax", {std::make_shared<opset::Softmax>()}},
          {"Softplus", {std::make_shared<opset::SoftPlus>()}},
          {"SpaceToDepth", {std::make_shared<opset::SpaceToDepth>()}},
          {"Split", {std::make_shared<opset::Split>(), constant}},
          {"SplitV", {std::make_shared<opset::VariadicSplit>(), constant}},
          {"Sqrt", {std::make_shared<opset::Sqrt>()}},
          {"Square", {std::make_shared<opset::Multiply>()}},
          {"SquaredDifference", {std::make_shared<opset::SquaredDifference>()}},
          {"Squeeze", {std::make_shared<opset::Squeeze>(), constant}},
          {"StridedSlice", {constant, std::make_shared<opset::StridedSlice>()}},
          {"Sub", {std::make_shared<opset::Subtract>()}},
          {"Sum", {std::make_shared<opset::ReduceSum>(), constant}},
          {"Tan", {std::make_shared<opset::Tan>()}},
          {"Tanh", {std::make_shared<opset::Tanh>()}},
          {"Tile", {std::make_shared<opset::Tile>()}},
          {"TopKV2", {std::make_shared<opset::TopK>(), constant}},
          {"Transpose", {std::make_shared<opset::Transpose>()}},
          {"Where",
           {std::make_shared<opset::NonZero>(),
            std::make_shared<opset::Transpose>()}},
          {"Xdivy",
           {constant, std::make_shared<opset::Divide>(),
            std::make_shared<opset::Equal>(),
            std::make_shared<opset::Select>()}},
          {"Unpack", {constant, std::make_shared<opset::StridedSlice>()}},
          {"ZerosLike", {constant}},
          {"NoOp", {}},
      };

  return TFtoNgraphOpMap;
}

//
// Main entry point for the marking pass.
//
Status MarkForClustering(Graph* graph,
                         const std::set<string> skip_these_nodes) {
  const TypeConstraintMap& type_constraint_map = GetTypeConstraintMap();

  // confirmation_function_map is non-const unlike the other maps
  static std::map<std::string, ConfirmationFunction> confirmation_function_map =
      GetConfirmationMap();

  const std::map<std::string, SetAttributesFunction>& set_attributes_map =
      GetAttributeSetters();

  const std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>&
      TFtoNgraphOpMap = GetTFToNgOpMap();

  //
  // IF YOU ARE ADDING A NEW OP IMPLEMENTATION, YOU MUST ADD A CONFIRMATION
  // FUNCTION, TYPE CONTRAINTS (IF ANY) AND STATIC INPUTS INDEXES (IF ANY) FOR
  // THE OP HERE.

  // The constraint function should refuse placement if the node is not
  // supported in the builder, and tag the node with any data that will be
  // needed in case the graph is broken up in a later rewrite pass (for example,
  // constant data).

  static std::set<string> disabled_ops_set = {};

  static bool initialized = false;

  std::set<string> disabled_ops_set_current = api::GetDisabledOps();
  bool op_set_support_has_changed =
      disabled_ops_set_current != disabled_ops_set;

  if (!initialized || op_set_support_has_changed) {
    confirmation_function_map = GetConfirmationMap();
    initialized = true;
  }

  if (op_set_support_has_changed) {
    NGRAPH_VLOG(5) << "Changing op support";
    disabled_ops_set = disabled_ops_set_current;
    for (auto itr : disabled_ops_set) {
      auto conf_itr = confirmation_function_map.find(itr);
      if (conf_itr == confirmation_function_map.end()) {
        // Note: This error means, we cannot disable NGraphEncapsulate and other
        // ng ops, because they are expected to never appear in
        // confirmation_function_map
        return errors::Internal("Tried to disable ngraph unsupported op ", itr);
      } else {
        NGRAPH_VLOG(5) << "Disabling op: " << itr;
        confirmation_function_map.erase(conf_itr);
      }
    }
  }

  std::unordered_map<string, int> no_support_histogram;
  std::unordered_map<string, int> fail_confirmation_histogram;
  std::unordered_map<string, int> fail_constraint_histogram;
  vector<Node*> nodes_marked_for_clustering;

  shared_ptr<Backend> op_backend = BackendManager::GetBackend();
  for (auto node : graph->op_nodes()) {
    bool mark_for_clustering = false;

    do {
      // check if output node
      bool skip_it = false;
      TF_RETURN_IF_ERROR(CheckIfOutputNode(node, skip_these_nodes, skip_it));
      if (skip_it) {
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Found Output Node: " << node->name()
                       << " - skip marking it for clustering";
        break;
      }

      // check placement
      bool placement_ok = false;
      TF_RETURN_IF_ERROR(NGraphPlacementRequested(node, placement_ok));
      if (!placement_ok) {
        NGRAPH_VLOG(5) << "Placement not requested: " << node->name();
        break;
      }

      // check node's confirmation constraints
      bool confirmation_constraint_ok = false;
      TF_RETURN_IF_ERROR(ConfirmationOk(node, confirmation_function_map,
                                        confirmation_constraint_ok));
      if (!confirmation_constraint_ok) {
        NGRAPH_VLOG(5) << "Node does not meet confirmation constraints: "
                       << node->name();
        if (confirmation_function_map.find(node->type_string()) ==
            confirmation_function_map.end()) {
          // not found
          no_support_histogram[node->type_string()]++;
        } else {
          // found
          fail_confirmation_histogram[node->type_string()]++;
        }
        break;
      }

      // check input type constraints
      bool type_constraint_ok = false;
      TF_RETURN_IF_ERROR(
          TypeConstraintOk(node, type_constraint_map, type_constraint_ok));
      if (!type_constraint_ok) {
        NGRAPH_VLOG(5) << "Inputs do not meet type constraints: "
                       << node->name();
        fail_constraint_histogram[node->type_string()]++;
        break;
      }

      // Check if op is supported by backend
      bool is_supported = false;
      TF_RETURN_IF_ERROR(IsSupportedByBackend(node, op_backend, TFtoNgraphOpMap,
                                              is_supported));

      if (!is_supported) {
        string backend;
        BackendManager::GetBackendName(backend);
        NGRAPH_VLOG(5) << "TF Op " << node->name() << " of type "
                       << node->type_string()
                       << " is not supported by backend: " << backend;
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

  if (api::IsLoggingPlacement()) {
    std::cout << "\n=============New sub-graph logs=============\n";
    // print summary for nodes failed to be marked
    std::cout << "NGTF_SUMMARY: Op_not_supported: ";
    util::PrintNodeHistogram(no_support_histogram);
    std::cout << "NGTF_SUMMARY: Op_failed_confirmation: ";
    util::PrintNodeHistogram(fail_confirmation_histogram);
    std::cout << "NGTF_SUMMARY: Op_failed_type_constraint: ";
    util::PrintNodeHistogram(fail_constraint_histogram);
  }

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

Status GetStaticInputs(Graph* graph, std::vector<int32>* static_input_indexes) {
  static_input_indexes->clear();
  for (auto node : graph->nodes()) {
    if (node->type_string() == "_Arg") {
      int32 index;
      auto status = GetNodeAttr(node->attrs(), "index", &index);
      if (status != Status::OK()) {
        return errors::Internal("error getting node attribute index");
      }

      for (auto edge : node->out_edges()) {
        if (edge->IsControlEdge() || !edge->dst()->IsOp()) {
          continue;
        }

        NGRAPH_VLOG(5) << "For arg " << index << " checking edge "
                       << edge->DebugString();

        if (InputIsStatic(edge->dst(), edge->dst_input())) {
          NGRAPH_VLOG(5) << "Marking edge static: " << edge->DebugString();
          static_input_indexes->push_back(index);
          break;
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
