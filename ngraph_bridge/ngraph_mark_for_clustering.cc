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

#include "tensorflow/core/graph/graph.h"

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/ngraph_version_utils.h"

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

// Check if op is supported by backend using is_supported API
Status IsSupportedByBackend(
    const Node* node, const ng::runtime::Backend* op_backend,
    const std::map<std::string, std::set<shared_ptr<ng::Node>>>&
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
    auto ret = op_backend->is_supported(**it);
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
    set_attributes_map["AvgPoolGrad"] = SetStaticInputs({0});
    set_attributes_map["ConcatV2"] = SetStaticInputs({-1});
    set_attributes_map["CombinedNonMaxSuppression"] =
        SetStaticInputs({2, 3, 4, 5});
    set_attributes_map["Conv2DBackpropFilter"] = SetStaticInputs({1});
    set_attributes_map["Conv2DBackpropInput"] = SetStaticInputs({0});
    set_attributes_map["ExpandDims"] = SetStaticInputs({1});
    set_attributes_map["Fill"] = SetStaticInputs({0});
    set_attributes_map["GatherV2"] = SetStaticInputs({2});
    set_attributes_map["Max"] = SetStaticInputs({1});
    set_attributes_map["Mean"] = SetStaticInputs({1});
    set_attributes_map["Min"] = SetStaticInputs({1});
    set_attributes_map["NonMaxSuppressionV4"] = SetStaticInputs({2, 3, 4});
    set_attributes_map["OneHot"] = SetStaticInputs({1});
    set_attributes_map["Pad"] = SetStaticInputs({1});
    set_attributes_map["Prod"] = SetStaticInputs({1});

    set_attributes_map["QuantizeAndDequantizeV2"] = SetStaticInputs({1, 2});
    set_attributes_map["QuantizedConcat"] = [](Node* n) {
      SetStaticInputs(n, {0});  // the axis
      auto num_of_tensors_to_concat = (n->num_inputs() - 1) / 3;
      // mark all mins and maxes static
      for (int idx = num_of_tensors_to_concat + 1; idx < n->num_inputs();
           idx++) {
        SetStaticInputs(n, {idx});
      }
      return Status::OK();
    };
    set_attributes_map["QuantizedConcatV2"] = [](Node* n) {
      auto num_of_tensors_to_concat = (n->num_inputs() - 1) / 3;
      // mark axis, all mins and maxes static
      std::vector<int> static_input_vec;
      for (int idx = num_of_tensors_to_concat; idx < n->num_inputs(); idx++) {
        static_input_vec.push_back(idx);
      }
      SetStaticInputs(n, static_input_vec);
      return Status::OK();
    };
    set_attributes_map["RandomUniform"] = SetStaticInputs({0});
    set_attributes_map["Reshape"] = SetStaticInputs({1});
    set_attributes_map["ResizeBilinear"] = SetStaticInputs({1});
    set_attributes_map["ScatterNd"] = SetStaticInputs({2});
    set_attributes_map["Slice"] = SetStaticInputs({1, 2});
    set_attributes_map["Split"] = SetStaticInputs({0});
    set_attributes_map["SplitV"] = SetStaticInputs({1, 2});
    set_attributes_map["StridedSlice"] = SetStaticInputs({1, 2, 3});
    set_attributes_map["Sum"] = SetStaticInputs({1});
    set_attributes_map["TopKV2"] = SetStaticInputs({1});
    set_attributes_map["Tile"] = SetStaticInputs({1});
    set_attributes_map["Transpose"] = SetStaticInputs({1});
    set_attributes_map["UnsortedSegmentSum"] = SetStaticInputs({2});
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
    confirmation_function_map["Add"] = SimpleConfirmationFunction();
    confirmation_function_map["AddN"] = SimpleConfirmationFunction();
    confirmation_function_map["AddV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Any"] = SimpleConfirmationFunction();
    confirmation_function_map["All"] = SimpleConfirmationFunction();
    confirmation_function_map["ArgMax"] = SimpleConfirmationFunction();
    confirmation_function_map["ArgMin"] = SimpleConfirmationFunction();
    confirmation_function_map["Atan2"] = SimpleConfirmationFunction();
    confirmation_function_map["AvgPool"] = SimpleConfirmationFunction();
    confirmation_function_map["AvgPoolGrad"] = SimpleConfirmationFunction();
    confirmation_function_map["BatchMatMul"] = SimpleConfirmationFunction();
    confirmation_function_map["BatchMatMulV2"] = SimpleConfirmationFunction();
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
    confirmation_function_map["Conv3D"] = SimpleConfirmationFunction();
    confirmation_function_map["CropAndResize"] = SimpleConfirmationFunction();
    confirmation_function_map["Cos"] = SimpleConfirmationFunction();
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
    confirmation_function_map["Dequantize"] = [](Node* n, bool* result) {
      string mode;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "mode", &mode));
      *result = (mode.compare("SCALED") == 0);
      return Status::OK();
    };
    confirmation_function_map["Equal"] = SimpleConfirmationFunction();
    confirmation_function_map["Exp"] = SimpleConfirmationFunction();
    confirmation_function_map["ExpandDims"] = SimpleConfirmationFunction();
    confirmation_function_map["Fill"] = SimpleConfirmationFunction();
    confirmation_function_map["Floor"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorDiv"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorMod"] = SimpleConfirmationFunction();
    confirmation_function_map["FusedBatchNorm"] = SimpleConfirmationFunction();
    confirmation_function_map["FusedBatchNormV2"] =
        SimpleConfirmationFunction();
    confirmation_function_map["FusedBatchNormV3"] =
        SimpleConfirmationFunction();
    confirmation_function_map["FusedBatchNormGrad"] = [](Node* n,
                                                         bool* result) {
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "is_training", result));
      return Status::OK();
    };
    confirmation_function_map["FusedBatchNormGradV3"] = [](Node* n,
                                                           bool* result) {
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "is_training", result));
      return Status::OK();
    };
    confirmation_function_map["_FusedConv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["GatherNd"] = SimpleConfirmationFunction();
    confirmation_function_map["GatherV2"] = SimpleConfirmationFunction();
    confirmation_function_map["_FusedMatMul"] =
        SimpleConfirmationFunction();  // TODO accept under all conditions?
                                       // check?
    confirmation_function_map["Greater"] = SimpleConfirmationFunction();
    confirmation_function_map["GreaterEqual"] = SimpleConfirmationFunction();
#if defined NGRAPH_DISTRIBUTED
    confirmation_function_map["HorovodAllreduce"] =
        SimpleConfirmationFunction();
    confirmation_function_map["HorovodBroadcast"] =
        SimpleConfirmationFunction();
#endif
    confirmation_function_map["Identity"] = SimpleConfirmationFunction();
    confirmation_function_map["IsFinite"] = SimpleConfirmationFunction();
    confirmation_function_map["L2Loss"] = SimpleConfirmationFunction();
    confirmation_function_map["LogSoftmax"] = SimpleConfirmationFunction();
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
    confirmation_function_map["MaxPool3D"] = SimpleConfirmationFunction();
    confirmation_function_map["MaxPoolGrad"] = SimpleConfirmationFunction();
    confirmation_function_map["Mean"] = SimpleConfirmationFunction();
    confirmation_function_map["Min"] = SimpleConfirmationFunction();
    confirmation_function_map["Minimum"] = SimpleConfirmationFunction();
    confirmation_function_map["Mul"] = SimpleConfirmationFunction();
    confirmation_function_map["Neg"] = SimpleConfirmationFunction();
    confirmation_function_map["NoOp"] = SimpleConfirmationFunction();
    confirmation_function_map["OneHot"] = SimpleConfirmationFunction();
    confirmation_function_map["Pad"] = SimpleConfirmationFunction();
    confirmation_function_map["Pow"] = SimpleConfirmationFunction();
    confirmation_function_map["PreventGradient"] = SimpleConfirmationFunction();
    confirmation_function_map["Prod"] = SimpleConfirmationFunction();
    confirmation_function_map["Rank"] = SimpleConfirmationFunction();
    confirmation_function_map["RandomUniform"] = SimpleConfirmationFunction();
    confirmation_function_map["QuantizeAndDequantizeV2"] = [](Node* n,
                                                              bool* result) {
      // accept only when num_bits == 8 and range is given
      bool range_given;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "range_given", &range_given));
      int num_bits;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "num_bits", &num_bits));
      *result = (num_bits == 8) && range_given;
      return Status::OK();
    };
    confirmation_function_map["QuantizedAvgPool"] =
        SimpleConfirmationFunction();
    confirmation_function_map["QuantizedConcat"] = SimpleConfirmationFunction();
    confirmation_function_map["QuantizedConcatV2"] =
        SimpleConfirmationFunction();
    confirmation_function_map["QuantizedConv2DWithBiasAndReluAndRequantize"] =
        SimpleConfirmationFunction();
    confirmation_function_map["QuantizedConv2DWithBiasAndRequantize"] =
        SimpleConfirmationFunction();
    confirmation_function_map
        ["QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"] =
            SimpleConfirmationFunction();
    confirmation_function_map
        ["QuantizedConv2DWithBiasSumAndReluAndRequantize"] =
            SimpleConfirmationFunction();
    confirmation_function_map["QuantizedMaxPool"] =
        SimpleConfirmationFunction();
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
    confirmation_function_map["ResizeBilinear"] = SimpleConfirmationFunction();
    confirmation_function_map["Rsqrt"] = SimpleConfirmationFunction();
    confirmation_function_map["RsqrtGrad"] = SimpleConfirmationFunction();
    confirmation_function_map["ScatterNd"] = SimpleConfirmationFunction();
    confirmation_function_map["Select"] = SimpleConfirmationFunction();
    confirmation_function_map["Shape"] = SimpleConfirmationFunction();
    confirmation_function_map["Sigmoid"] = SimpleConfirmationFunction();
    confirmation_function_map["SigmoidGrad"] = SimpleConfirmationFunction();
    confirmation_function_map["Sign"] = SimpleConfirmationFunction();
    confirmation_function_map["Sin"] = SimpleConfirmationFunction();
    confirmation_function_map["Size"] = SimpleConfirmationFunction();
    confirmation_function_map["Slice"] = SimpleConfirmationFunction();
    confirmation_function_map["Snapshot"] = SimpleConfirmationFunction();
    confirmation_function_map["Softmax"] = SimpleConfirmationFunction();
    confirmation_function_map["SoftmaxCrossEntropyWithLogits"] =
        SimpleConfirmationFunction();
    confirmation_function_map["Softplus"] = SimpleConfirmationFunction();
    confirmation_function_map["SpaceToDepth"] =
        confirmation_function_map["DepthToSpace"];
    confirmation_function_map["SparseSoftmaxCrossEntropyWithLogits"] =
        SimpleConfirmationFunction();
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
    confirmation_function_map["Tanh"] = SimpleConfirmationFunction();
    confirmation_function_map["TanhGrad"] = SimpleConfirmationFunction();
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
    confirmation_function_map["UnsortedSegmentSum"] =
        SimpleConfirmationFunction();
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
    type_constraint_map["Add"]["T"] = NGraphNumericDTypes();
    type_constraint_map["AddN"]["T"] = NGraphNumericDTypes();
    type_constraint_map["AddV2"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Any"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["All"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["ArgMax"]["T"] = NGraphNumericDTypes();
    type_constraint_map["ArgMax"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["ArgMin"]["T"] = NGraphNumericDTypes();
    type_constraint_map["ArgMin"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Atan2"]["T"] = NGraphRealDTypes();
    type_constraint_map["AvgPool"]["T"] = NGraphNumericDTypes();
    type_constraint_map["AvgPoolGrad"]["T"] = NGraphNumericDTypes();
    type_constraint_map["BatchMatMul"]["T"] = NGraphNumericDTypes();
    type_constraint_map["BatchMatMulV2"]["T"] = NGraphNumericDTypes();
    type_constraint_map["BiasAdd"]["T"] = NGraphNumericDTypes();
    type_constraint_map["BiasAddGrad"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Cast"]["SrcT"] = NGraphDTypes();
    type_constraint_map["Cast"]["DstT"] = NGraphDTypes();
    type_constraint_map["ConcatV2"]["T"] = NGraphDTypes();
    type_constraint_map["ConcatV2"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Const"]["dtype"] = NGraphDTypes();
    type_constraint_map["Conv2D"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Conv2DBackpropInput"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Conv3D"]["T"] = NGraphNumericDTypes();
    type_constraint_map["CropAndResize"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Cos"]["T"] = NGraphRealDTypes();
    type_constraint_map["Cumsum"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Cumsum"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["DepthToSpace"]["T"] = NGraphDTypes();
    type_constraint_map["DepthwiseConv2dNative"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Dequantize"]["T"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["Equal"]["T"] = NGraphDTypes();
    type_constraint_map["Exp"]["T"] = NGraphNumericDTypes();
    type_constraint_map["ExpandDims"]["T"] = NGraphDTypes();
    type_constraint_map["Floor"]["T"] = NGraphNumericDTypes();
    type_constraint_map["FloorDiv"]["T"] = NGraphNumericDTypes();
    type_constraint_map["FloorMod"]["T"] = NGraphNumericDTypes();
    type_constraint_map["FusedBatchNorm"]["T"] = NGraphNumericDTypes();
    // TODO (mingshan): FusedBatchNormV2, V3 supports DT_HALF,DT_BFLOAT16,
    // DT_FLOAT
    type_constraint_map["FusedBatchNormV2"]["T"] = {DT_FLOAT};
    type_constraint_map["FusedBatchNormV3"]["T"] = {DT_FLOAT};
    type_constraint_map["FusedBatchNormGrad"]["T"] = NGraphNumericDTypes();
    type_constraint_map["GatherNd"]["Tparams"] = {DT_FLOAT};  // NGraphDTypes();
    type_constraint_map["GatherNd"]["Tindices"] = NGraphIndexDTypes();
    type_constraint_map["FusedBatchNormGradV3"]["T"] = NGraphNumericDTypes();
    type_constraint_map["GatherV2"]["Tparams"] = NGraphDTypes();
    type_constraint_map["GatherV2"]["Tindices"] = NGraphIndexDTypes();
    type_constraint_map["GatherV2"]["Taxis"] = NGraphIndexDTypes();
    type_constraint_map["_FusedConv2D"]["T"] = NGraphRealDTypes();
    type_constraint_map["_FusedMatMul"]["T"] = NGraphRealDTypes();
    type_constraint_map["Greater"]["T"] = NGraphDTypes();
    type_constraint_map["GreaterEqual"]["T"] = NGraphDTypes();
#if defined NGRAPH_DISTRIBUTED
    type_constraint_map["HorovodAllreduce"]["T"] = NGraphNumericDTypes();
    type_constraint_map["HorovodBroadcast"]["T"] = NGraphNumericDTypes();
#endif
    type_constraint_map["Identity"]["T"] = NGraphDTypes();
    type_constraint_map["IsFinite"]["T"] = NGraphRealDTypes();
    type_constraint_map["L2Loss"]["T"] = NGraphNumericDTypes();
    type_constraint_map["LogSoftmax"]["T"] = NGraphRealDTypes();
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
    type_constraint_map["MaxPool3D"]["T"] = NGraphNumericDTypes();
    type_constraint_map["MaxPoolGrad"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Mean"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Mean"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Min"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Min"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["Minimum"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Mul"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Neg"]["T"] = NGraphNumericDTypes();
    type_constraint_map["NonMaxSuppressionV4"]["T"] = {
        DT_FLOAT};  // TF allows half too
    type_constraint_map["OneHot"]["T"] = NGraphDTypes();
    type_constraint_map["Pack"]["T"] = NGraphDTypes();
    type_constraint_map["RandomUniform"]["T"] = NGraphDTypes();
    type_constraint_map["Pad"]["T"] = NGraphDTypes();
    type_constraint_map["Pad"]["Tpaddings"] = NGraphIndexDTypes();
    type_constraint_map["Pow"]["T"] = NGraphNumericDTypes();
    type_constraint_map["PreventGradient"]["T"] = NGraphDTypes();
    type_constraint_map["Prod"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Prod"]["Tidx"] = NGraphIndexDTypes();
    type_constraint_map["QuantizeAndDequantizeV2"]["T"] = NGraphRealDTypes();
    type_constraint_map["QuantizedAvgPool"]["T"] =
        NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConcat"]["T"] =
        NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConcatV2"]["T"] =
        NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasAndReluAndRequantize"]
                       ["Tinput"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasAndReluAndRequantize"]
                       ["Tfilter"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasAndReluAndRequantize"]
                       ["Tbias"] = NGraphBiasDTypes();
    // TODO: check if any other type constraint is required
    // https://github.com/tensorflow/tensorflow/blob/c95ca05536144451ef78ca6e2c15f0f65ebaaf95/tensorflow/core/ops/nn_ops.cc#L2780
    type_constraint_map["QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"]
                       ["Tinput"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"]
                       ["Tsummand"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"]
                       ["Tfilter"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"]
                       ["Tbias"] = NGraphBiasDTypes();
    type_constraint_map["QuantizedConv2DWithBiasSumAndReluAndRequantize"]
                       ["Tinput"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasSumAndReluAndRequantize"]
                       ["Tsummand"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasSumAndReluAndRequantize"]
                       ["Tfilter"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasSumAndReluAndRequantize"]
                       ["Tbias"] = NGraphBiasDTypes();
    type_constraint_map["QuantizedConv2DWithBiasAndRequantize"]["Tinput"] =
        NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasAndRequantize"]["Tfilter"] =
        NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizedConv2DWithBiasAndRequantize"]["Tbias"] =
        NGraphBiasDTypes();
    type_constraint_map["QuantizedMaxPool"]["T"] =
        NGraphSupportedQuantizedDTypes();
    type_constraint_map["QuantizeV2"]["T"] = NGraphSupportedQuantizedDTypes();
    type_constraint_map["Rank"]["T"] = NGraphNumericDTypes();
    type_constraint_map["RealDiv"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Reciprocal"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Relu"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Relu6"]["T"] = NGraphNumericDTypes();
    type_constraint_map["ReluGrad"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Reshape"]["T"] = NGraphDTypes();
    type_constraint_map["Reshape"]["Tshape"] = NGraphIndexDTypes();
    type_constraint_map["ResizeBilinear"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Rsqrt"]["T"] = NGraphDTypes();
    type_constraint_map["RsqrtGrad"]["T"] = NGraphRealDTypes();
    type_constraint_map["ScatterNd"]["T"] = NGraphDTypes();
    type_constraint_map["ScatterNd"]["Tindices"] = NGraphIndexDTypes();
    type_constraint_map["Select"]["T"] = NGraphDTypes();
    type_constraint_map["Shape"]["T"] = NGraphDTypes();
    type_constraint_map["Shape"]["out_type"] = NGraphIndexDTypes();
    type_constraint_map["Sigmoid"]["T"] = NGraphNumericDTypes();
    type_constraint_map["SigmoidGrad"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Sign"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Sin"]["T"] = NGraphRealDTypes();
    type_constraint_map["Size"]["T"] = NGraphDTypes();
    type_constraint_map["Size"]["out_type"] = NGraphIndexDTypes();
    type_constraint_map["Slice"]["T"] = NGraphDTypes();
    type_constraint_map["Slice"]["Index"] = NGraphIndexDTypes();
    type_constraint_map["Snapshot"]["T"] = NGraphDTypes();
    type_constraint_map["Softmax"]["T"] = NGraphNumericDTypes();
    // For SoftmaxCrossEntropyWithLogits, see
    // https://github.com/tensorflow/tensorflow/blob/c95ca05536144451ef78ca6e2c15f0f65ebaaf95/tensorflow/core/ops/nn_ops.cc#L1096
    type_constraint_map["SoftmaxCrossEntropyWithLogits"]["T"] =
        NGraphRealDTypes();
    type_constraint_map["Softplus"]["T"] = NGraphRealDTypes();
    type_constraint_map["SpaceToDepth"]["T"] = NGraphDTypes();
    type_constraint_map["SparseSoftmaxCrossEntropyWithLogits"]["T"] =
        NGraphRealDTypes();
    type_constraint_map["SparseSoftmaxCrossEntropyWithLogits"]["Tlabels"] =
        NGraphNumericDTypes();
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
    type_constraint_map["Tanh"]["T"] = NGraphNumericDTypes();
    type_constraint_map["TanhGrad"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Tile"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Tile"]["Tmultiples"] = NGraphIndexDTypes();
    type_constraint_map["TopKV2"]["T"] = NGraphNumericDTypes();
    type_constraint_map["Transpose"]["T"] = NGraphDTypes();
    type_constraint_map["Transpose"]["Tperm"] = NGraphIndexDTypes();
    type_constraint_map["Unpack"]["T"] = NGraphDTypes();
    type_constraint_map["UnsortedSegmentSum"]["T"] = NGraphNumericDTypes();
    type_constraint_map["UnsortedSegmentSum"]["Tindices"] = NGraphIndexDTypes();
    type_constraint_map["UnsortedSegmentSum"]["Tnumsegments"] =
        NGraphIndexDTypes();
    initialized = true;
  }
  return type_constraint_map;
}

const std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>&
GetTFToNgOpMap() {
  // Constant Op, ReluGrad Op do not have default Constructor
  // in ngraph, so passing a dummy node
  auto constant = ngraph::op::Constant::create(ngraph::element::f32,
                                               ngraph::Shape{}, {2.0f});
  auto shape_a = ngraph::Shape{2, 5};
  auto A = make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_a);
  auto delta_val =
      make_shared<ngraph::op::Parameter>(ngraph::element::f32, shape_a);
  auto relu = make_shared<ngraph::op::ReluBackprop>(A, delta_val);
  // Map:: TF ops to NG Ops to track if all the Ngraph ops
  // are supported by backend
  // Update this Map if a new TF Op translation is
  // implemented or a new Ngraph Op has been added
  static std::map<std::string, std::set<shared_ptr<ng::Node>>> TFtoNgraphOpMap {
    {"Abs", {std::make_shared<ngraph::op::Abs>()}},
        {"Add", {std::make_shared<ngraph::op::Add>()}},
        {"AddN", {std::make_shared<ngraph::op::Add>()}},
        {"AddV2",
         {std::make_shared<ngraph::op::Add>(),
          std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"Any",
         {std::make_shared<ngraph::op::Any>(),
          std::make_shared<ngraph::op::Broadcast>(), constant,
          std::make_shared<ngraph::op::Reshape>()}},
        {"All",
         {std::make_shared<ngraph::op::All>(),
          std::make_shared<ngraph::op::Broadcast>(), constant,
          std::make_shared<ngraph::op::Reshape>()}},
        {"ArgMax", {std::make_shared<ngraph::op::ArgMax>()}},
        {"ArgMin", {std::make_shared<ngraph::op::ArgMin>()}},
        {"Atan2", {std::make_shared<ngraph::op::Atan2>()}},
        {"AvgPool", {std::make_shared<ngraph::op::AvgPool>()}},
        {"AvgPoolGrad", {std::make_shared<ngraph::op::AvgPoolBackprop>()}},
        {"BatchMatMul",
         {std::make_shared<ngraph::op::BatchMatMulTranspose>(),
          std::make_shared<ngraph::op::Dot>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"BatchMatMulV2",
         {std::make_shared<ngraph::op::BatchMatMulTranspose>(),
          std::make_shared<ngraph::op::Dot>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"BiasAdd",
         {std::make_shared<ngraph::op::Add>(),
          std::make_shared<ngraph::op::Broadcast>()}},
        {"BiasAddGrad", {std::make_shared<ngraph::op::Sum>(), constant}},
        {"Cast", {std::make_shared<ngraph::op::Convert>()}},
        {"ConcatV2", {std::make_shared<ngraph::op::Concat>()}},
        {"Const", {constant}}, {"Conv2D",
                                {std::make_shared<ngraph::op::Reshape>(),
                                 std::make_shared<ngraph::op::Convolution>()}},
        {"Conv2DBackpropFilter",
         {std::make_shared<ngraph::op::ConvolutionBackpropFilters>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"Conv2DBackpropInput",
         {std::make_shared<ngraph::op::ConvolutionBackpropData>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"Conv3D",
         {std::make_shared<ngraph::op::Convolution>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"Cos", {std::make_shared<ngraph::op::Cos>()}},
        {"CropAndResize", {std::make_shared<ngraph::op::CropAndResize>()}},
        {"Cumsum", {std::make_shared<ngraph::op::CumSum>()}},
        {"DepthToSpace", {std::make_shared<ngraph::op::Reshape>()}},
        {"DepthwiseConv2dNative",
         {std::make_shared<ngraph::op::Slice>(),
          std::make_shared<ngraph::op::Convolution>(),
          std::make_shared<ngraph::op::Concat>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"Dequantize",
         {std::make_shared<ngraph::op::Dequantize>(), constant,
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Maximum>(),
          std::make_shared<ngraph::op::Abs>()}},
        {"Equal", {std::make_shared<ngraph::op::Equal>()}},
        {"Exp", {std::make_shared<ngraph::op::Exp>()}},
        {"ExpandDims", {std::make_shared<ngraph::op::Reshape>()}},
        {"Fill", {std::make_shared<ngraph::op::Broadcast>()}},
        {"Floor", {std::make_shared<ngraph::op::Floor>()}},
        {"FloorDiv",
         {std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Floor>(),
          std::make_shared<ngraph::op::Broadcast>()}},
        {"FloorMod",
         {std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Subtract>(),
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Floor>(),
          std::make_shared<ngraph::op::Broadcast>()}},
        {"FusedBatchNorm",
         {std::make_shared<ngraph::op::BatchNormTraining>(),
          std::make_shared<ngraph::op::GetOutputElement>(), constant,
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::BatchNormInference>()}},
        {"FusedBatchNormV2",
         {std::make_shared<ngraph::op::BatchNormTraining>(),
          std::make_shared<ngraph::op::GetOutputElement>(), constant,
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::BatchNormInference>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"FusedBatchNormV3",
         {std::make_shared<ngraph::op::BatchNormTraining>(),
          std::make_shared<ngraph::op::GetOutputElement>(), constant,
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::BatchNormInference>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"FusedBatchNormGrad",
         {constant, std::make_shared<ngraph::op::GetOutputElement>(),
          std::make_shared<ngraph::op::BatchNormTrainingBackprop>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"FusedBatchNormGradV3",
         {constant, std::make_shared<ngraph::op::GetOutputElement>(),
          std::make_shared<ngraph::op::BatchNormTrainingBackprop>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"GatherNd", {std::make_shared<ngraph::op::GatherND>()}},
        {"GatherV2", {std::make_shared<ngraph::op::Gather>()}},
        {"_FusedConv2D",
         {std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::Convolution>(), constant,
          std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Relu>(),
          std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Add>(),
          std::make_shared<ngraph::op::BatchNormInference>()}},
        {"_FusedMatMul",
         {std::make_shared<ngraph::op::Dot>(),
          std::make_shared<ngraph::op::Relu>(),
          std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Add>(), constant,
          std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"Greater", {std::make_shared<ngraph::op::Greater>()}},
        {"GreaterEqual", {std::make_shared<ngraph::op::GreaterEq>()}},
        {"Identity", {}}, {"IsFinite",
                           {constant, std::make_shared<ngraph::op::NotEqual>(),
                            std::make_shared<ngraph::op::Equal>(),
                            std::make_shared<ngraph::op::And>()}},
        {"L2Loss",
         {constant, std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Sum>(),
          std::make_shared<ngraph::op::Divide>()}},
        {"LogSoftmax",
         {std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Max>(),
          std::make_shared<ngraph::op::Subtract>(),
          std::make_shared<ngraph::op::Exp>(),
          std::make_shared<ngraph::op::Log>(),
          std::make_shared<ngraph::op::Sum>(), constant}},
        {"Less", {std::make_shared<ngraph::op::Less>()}},
        {"LessEqual", {std::make_shared<ngraph::op::LessEq>()}},
        {"Log", {std::make_shared<ngraph::op::Log>()}},
        {"LogicalAnd", {std::make_shared<ngraph::op::And>()}},
        {"LogicalNot", {std::make_shared<ngraph::op::Not>()}},
        {"LogicalOr", {std::make_shared<ngraph::op::Or>()}},
        {"MatMul",
         {std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::Dot>()}},
        {"Max", {std::make_shared<ngraph::op::Max>(), constant}},
        {"Maximum",
         {std::make_shared<ngraph::op::Maximum>(),
          std::make_shared<ngraph::op::Broadcast>()}},
        {"MaxPool",
         {std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::MaxPool>()}},
        {"MaxPool3D",
         {std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::MaxPool>()}},
        {"MaxPoolGrad",
         {std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::MaxPoolBackprop>()}},
        {"Mean",
         {std::make_shared<ngraph::op::Reshape>(), constant,
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Sum>()}},
        {"Min", {std::make_shared<ngraph::op::Min>(), constant}},
        {"Minimum",
         {std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Broadcast>()}},
        {"Mul", {std::make_shared<ngraph::op::Multiply>()}},
        {"Neg", {std::make_shared<ngraph::op::Negative>()}},
        {"OneHot",
         {std::make_shared<ngraph::op::OneHot>(),
          std::make_shared<ngraph::op::Convert>(),
          std::make_shared<ngraph::op::Select>(),
          std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::Broadcast>()}},
        {"Pack",
         {std::make_shared<ngraph::op::Concat>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"Pad", {constant, std::make_shared<ngraph::op::Pad>()}},
        {"Pow", {std::make_shared<ngraph::op::Power>()}},
        {"PreventGradient", {}},
        {"Prod",
         {std::make_shared<ngraph::op::Product>(), constant,
          std::make_shared<ngraph::op::Reshape>()}},
        {"QuantizeAndDequantizeV2",
         {constant, std::make_shared<ngraph::op::Quantize>(),
          std::make_shared<ngraph::op::Dequantize>()}},
        // Next few are CPU only ops
        {"QuantizedAvgPool",
         {std::make_shared<ngraph::op::AvgPool>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"QuantizedConcat",
         {constant, std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::Min>(),
          std::make_shared<ngraph::op::Max>(),
          std::make_shared<ngraph::op::Abs>(),
          std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Maximum>(),
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Dequantize>(),
          std::make_shared<ngraph::op::Quantize>(),
          std::make_shared<ngraph::op::Concat>(),
          std::make_shared<ngraph::op::Broadcast>()}},
        {"QuantizedConcatV2",
         {constant, std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::Min>(),
          std::make_shared<ngraph::op::Max>(),
          std::make_shared<ngraph::op::Abs>(),
          std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Maximum>(),
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Dequantize>(),
          std::make_shared<ngraph::op::Quantize>(),
          std::make_shared<ngraph::op::Concat>()}},
        {"QuantizedConv2DWithBiasAndReluAndRequantize",
         {constant, std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Abs>(),
          std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Maximum>(),
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Quantize>(),
          std::make_shared<ngraph::op::QuantizedConvolutionBias>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"QuantizedConv2DWithBiasAndRequantize",
         {constant, std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Abs>(),
          std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Maximum>(),
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Quantize>(),
          std::make_shared<ngraph::op::QuantizedConvolutionBias>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
         {constant, std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Abs>(),
          std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Maximum>(),
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Quantize>(),
          std::make_shared<ngraph::op::QuantizedConvolutionBiasSignedAdd>(),
          std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::Convert>()}},
        {"QuantizedConv2DWithBiasSumAndReluAndRequantize",
         {constant, std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Abs>(),
          std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Maximum>(),
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Quantize>(),
          std::make_shared<ngraph::op::QuantizedConvolutionBiasAdd>(),
          std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::Convert>()}},
        {"QuantizedMaxPool",
         {std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::MaxPool>()}},
        // End of CPU only ops
        {"QuantizeV2",
         {constant, std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Abs>(),
          std::make_shared<ngraph::op::Maximum>(),
          std::make_shared<ngraph::op::Quantize>(),
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Add>()}},
        {
            "RandomUniform",
            {constant, std::make_shared<ngraph::op::RandomUniform>()},
        },
        {"Rank", {constant}}, {"RealDiv",
                               {std::make_shared<ngraph::op::Divide>(),
                                std::make_shared<ngraph::op::Broadcast>()}},
        {"Reciprocal", {constant, std::make_shared<ngraph::op::Power>()}},
        {"Relu", {std::make_shared<ngraph::op::Relu>()}},
        {"Relu6",
         {constant, std::make_shared<ngraph::op::Minimum>(),
          std::make_shared<ngraph::op::Relu>()}},
        {"ReluGrad", {relu}},
        // TODO: remove Convert later
        {"ResizeBilinear",
         {std::make_shared<ngraph::op::Convert>(),
          std::make_shared<ngraph::op::Interpolate>()}},
        {"Rsqrt", {constant, std::make_shared<ngraph::op::Power>()}},
        {"RsqrtGrad",
         {constant, std::make_shared<ngraph::op::Power>(),
          std::make_shared<ngraph::op::Multiply>()}},
        {"Select",
         {std::make_shared<ngraph::op::Reshape>(),
          std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Select>()}},
        {"Reshape", {std::make_shared<ngraph::op::Reshape>()}},
        {"ScatterNd", {constant, std::make_shared<ngraph::op::ScatterNDAdd>()}},
        {"Shape", {constant}}, {"Sigmoid",
                                {constant, std::make_shared<ngraph::op::Exp>(),
                                 std::make_shared<ngraph::op::Negative>(),
                                 std::make_shared<ngraph::op::Add>(),
                                 std::make_shared<ngraph::op::Divide>()}},
        {"SigmoidGrad",
         {constant, std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Subtract>()}},
        {"Sin", {std::make_shared<ngraph::op::Sin>()}}, {"Size", {constant}},
        {"Sign", {std::make_shared<ngraph::op::Sign>()}},
        {"Slice", {std::make_shared<ngraph::op::Slice>()}}, {"Snapshot", {}},
        {"Softmax", {std::make_shared<ngraph::op::Softmax>(), constant}},
        {"SoftmaxCrossEntropyWithLogits",
         {std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Max>(),
          std::make_shared<ngraph::op::Subtract>(),
          std::make_shared<ngraph::op::Exp>(),
          std::make_shared<ngraph::op::Sum>(),
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::Convert>(),
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Log>(), constant}},
        {"Softplus",
         {constant, std::make_shared<ngraph::op::Exp>(),
          std::make_shared<ngraph::op::Log>(),
          std::make_shared<ngraph::op::Add>()}},
        {"SpaceToDepth",
         {std::make_shared<ngraph::op::Slice>(),
          std::make_shared<ngraph::op::Concat>()}},
        {"SparseSoftmaxCrossEntropyWithLogits",
         {std::make_shared<ngraph::op::Broadcast>(),
          std::make_shared<ngraph::op::Max>(),
          std::make_shared<ngraph::op::Subtract>(),
          std::make_shared<ngraph::op::Exp>(),
          std::make_shared<ngraph::op::Sum>(),
          std::make_shared<ngraph::op::Divide>(),
          std::make_shared<ngraph::op::OneHot>(),
          std::make_shared<ngraph::op::Convert>(),
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Log>(), constant}},
        {"Split", {std::make_shared<ngraph::op::Slice>()}},
        {"SplitV", {std::make_shared<ngraph::op::Slice>()}},
        {"Sqrt", {std::make_shared<ngraph::op::Sqrt>()}},
        {"Square", {std::make_shared<ngraph::op::Multiply>()}},
        {"SquaredDifference",
         {std::make_shared<ngraph::op::Subtract>(),
          std::make_shared<ngraph::op::Multiply>(),
          std::make_shared<ngraph::op::Broadcast>()}},
        {"Squeeze", {std::make_shared<ngraph::op::Reshape>()}},
        {"StridedSlice",
         {std::make_shared<ngraph::op::Reverse>(),
          std::make_shared<ngraph::op::Slice>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"Sub", {std::make_shared<ngraph::op::Subtract>()}},
        {"Sum",
         {std::make_shared<ngraph::op::Sum>(),
          std::make_shared<ngraph::op::Reshape>(), constant}},
        {"Tanh", {std::make_shared<ngraph::op::Tanh>()}},
        {"TanhGrad",
         {constant, std::make_shared<ngraph::op::Subtract>(),
          std::make_shared<ngraph::op::Multiply>()}},
        {"Tile", {constant, std::make_shared<ngraph::op::Concat>()}},
        {"TopKV2",
         {std::make_shared<ngraph::op::TopK>(),
          std::make_shared<ngraph::op::GetOutputElement>()}},
        {"Transpose", {constant, std::make_shared<ngraph::op::Reshape>()}},
        {"UnsortedSegmentSum",
         {constant, std::make_shared<ngraph::op::ScatterAdd>()}},
        {"Unpack",
         {std::make_shared<ngraph::op::Slice>(),
          std::make_shared<ngraph::op::Reshape>()}},
        {"ZerosLike", {constant}},
#if defined NGRAPH_DISTRIBUTED
        {"HorovodAllreduce", {std::make_shared<ngraph::op::AllReduce>()}},
        {"HorovodBroadcast",
         {std::make_shared<ngraph::op::BroadcastDistributed>()}},
#endif
        {"NoOp", {}},
  };

  return TFtoNgraphOpMap;
}

//
// Main entry point for the marking pass.
//
Status MarkForClustering(Graph* graph, const std::set<string> skip_these_nodes,
                         const string& current_backend) {
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

  std::set<string> disabled_ops_set_current = config::GetDisabledOps();

  bool op_set_support_has_changed =
      disabled_ops_set_current != disabled_ops_set;

  if (!initialized || op_set_support_has_changed) {
    confirmation_function_map = GetConfirmationMap();
    initialized = true;
  }

  // Right now it cannot be inside the if(!initialized) block, because it is
  // backend dependent, which might change with different sess.run()s
  confirmation_function_map["NonMaxSuppressionV4"] = [&current_backend](
      Node*, bool* result) {
    auto config_map =
        BackendManager::GetBackendAttributeValues(current_backend);
    *result = (config_map.at("ngraph_backend") == "NNPI");
    return Status::OK();
  };

  confirmation_function_map["CombinedNonMaxSuppression"] = [&current_backend](
      Node*, bool* result) {
    auto config_map =
        BackendManager::GetBackendAttributeValues(current_backend);
    *result = (config_map.at("ngraph_backend") == "NNPI");
    return Status::OK();
  };

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
  vector<Node*> variable_type_nodes;
  string ng_backend_type;
  // Create nGraph backend
  BackendManager::GetCurrentlySetBackendName(&ng_backend_type);
  // Create backend to query is_supported
  TF_RETURN_IF_ERROR(BackendManager::CreateBackend(ng_backend_type));
  ng::runtime::Backend* op_backend =
      BackendManager::GetBackend(ng_backend_type);

  for (auto node : graph->op_nodes()) {
    bool mark_for_clustering = false;

    if (IsNGVariableType(node->type_string())) {
      variable_type_nodes.push_back(node);
      continue;
    }

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
        NGRAPH_VLOG(5) << "TF Op " << node->name() << " of type "
                       << node->type_string()
                       << " is not supported by backend: " << ng_backend_type;
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

  // Release backend created to query is_supported
  BackendManager::ReleaseBackend(ng_backend_type);

  if (config::IsLoggingPlacement()) {
    std::cout << "\n=============New sub-graph logs=============\n";
    // print summary for nodes failed to be marked
    std::cout << "NGTF_SUMMARY: Op_not_supported: ";
    print_node_histogram(no_support_histogram);
    std::cout << "\n";
    std::cout << "NGTF_SUMMARY: Op_failed_confirmation: ";
    print_node_histogram(fail_confirmation_histogram);
    std::cout << "\n";
    std::cout << "NGTF_SUMMARY: Op_failed_type_constraint: ";
    print_node_histogram(fail_constraint_histogram);
    std::cout << "\n";
  }

  for (auto node : nodes_marked_for_clustering) {
    // TODO(amprocte): move attr name to a constant
    node->AddAttr("_ngraph_marked_for_clustering", true);
    SetNodeBackend(node, current_backend);
    auto it = set_attributes_map.find(node->type_string());
    if (it != set_attributes_map.end()) {
      TF_RETURN_IF_ERROR(it->second(node));
    }
  }

  for (auto node : variable_type_nodes) {
    SetNodeBackend(node, current_backend);
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

Status GetNodeBackend(const Node* node, string* backend_name) {
  // TODO(amprocte): move attr name to a constant
  NGRAPH_VLOG(5) << "Getting backend " << node->name();
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), "_ngraph_backend", backend_name));
  return Status::OK();
}

// Can be extended to check the TF Device placement and/or user specified
// backend
// and accordingly assign backend
void SetNodeBackend(Node* node, const string& backend_name) {
  NGRAPH_VLOG(5) << "Setting backend " << node->name() << " " << backend_name;
  node->AddAttr("_ngraph_backend", backend_name);
}

void ResetMarkForClustering(Graph* graph) {
  ClearAttribute(graph, {"_ngraph_marked_for_clustering", "_ngraph_backend",
                         "_ngraph_static_inputs"});
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
