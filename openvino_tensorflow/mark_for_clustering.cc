/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "tensorflow/core/graph/graph.h"

#include "api.h"
#include "backend_manager.h"
#include "openvino_tensorflow/default_opset.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_utils.h"
#include "openvino_tensorflow/ovtf_version_utils.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

//
// The "marking" pass checks every node with requested placement on nGraph,
// and either rejects the placement request, or tags it with suitable metadata.
//
// For now we assume that every node has nGraph placement requested, unless the
// environment variable OPENVINO_TF_DISABLE is set. (TODO(amprocte): implement
// something better.)
//
// Each TensorFlow op supported by nGraph has a "confirmation function"
// associated with it. When the confirmation pass encounters a node of op "Op",
// the confirmation function for "Op" first checks if this particular instance
// of the op can be placed on nGraph, and returns "true" if placement is
// allowed. This is followed by checks for input datatype of the
// op.

// Each op that passes all the checks, has the attribute
// "_ovtf_marked_for_clustering" set to "true". Additional metadata (Static
// Inputs) for the op is also set.

// Different Checks before we mark for clustering
//
// Utility function to check if placement on the NGRAPH device has been
// requested.
//
// FIXME(amprocte): stubbed out for now because NGRAPH device is gone.
//

// Marks the input indices in "inputs" as static
static inline void SetStaticInputs(Node* n, std::vector<int32> inputs) {
#ifdef _WIN32
  if (!inputs.empty()) {
    n->AddAttr("_ovtf_static_inputs", inputs);
  }
#else
  n->AddAttr("_ovtf_static_inputs", inputs);
#endif
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

const std::map<std::string, SetAttributesFunction>& GetAttributeSetters() {
  //
  // A map of op types (e.g. "Add") to set_attribute functions. These can be
  // used to set any additional attributes. For example:
  //
  //    confirmation_function_map["MyOp"] = [](Node* n) {
  //     if(n->condition()){
  //        int dummy=5;
  //        n->AddAttr("_ovtf_dummy_attr", dummy);
  //      }
  //
  //      vector<int32> static_input_index =5;
  //      n->AddAttr("_ovtf_static_inputs", static_input_index);
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
    set_attributes_map["BatchToSpaceND"] = SetStaticInputs({1});
    set_attributes_map["ConcatV2"] = SetStaticInputs({-1});
    set_attributes_map["Conv2DBackpropInput"] = SetStaticInputs({0});
    set_attributes_map["Conv3DBackpropInputV2"] = SetStaticInputs({0});
    set_attributes_map["CropAndResize"] = SetStaticInputs({1, 2, 3});
    set_attributes_map["ExpandDims"] = SetStaticInputs({1});
    set_attributes_map["GatherV2"] = SetStaticInputs({2});
    set_attributes_map["Max"] = SetStaticInputs({1});
    set_attributes_map["Mean"] = SetStaticInputs({1});
    set_attributes_map["Min"] = SetStaticInputs({1});
    set_attributes_map["MirrorPad"] = SetStaticInputs({1});
    set_attributes_map["Pad"] = SetStaticInputs({1});
    set_attributes_map["PadV2"] = SetStaticInputs({1});
    set_attributes_map["Prod"] = SetStaticInputs({1});
    set_attributes_map["Reshape"] = SetStaticInputs({1});
    set_attributes_map["ScatterNd"] = SetStaticInputs({2});
    set_attributes_map["Slice"] = SetStaticInputs({1, 2});
    set_attributes_map["SpaceToBatchND"] = SetStaticInputs({1});
    set_attributes_map["Split"] = SetStaticInputs({0});
    set_attributes_map["SplitV"] = SetStaticInputs({1, 2});
    set_attributes_map["StridedSlice"] = SetStaticInputs({1, 2, 3});
    set_attributes_map["Sum"] = SetStaticInputs({1});
    set_attributes_map["Tile"] = SetStaticInputs({1});
    initialized = true;
  }
  return set_attributes_map;
}

const std::map<std::string, std::set<std::shared_ptr<ov::Node>>>&
GetTFToNgOpMap() {
  // Constant Op does not have default Constructor
  // in ngraph, so passing a dummy node
  auto constant =
      opset::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
  // Map:: TF ops to NG Ops to track if all the Ngraph ops
  // are supported by backend
  // Update this Map if a new TF Op translation is
  // implemented or a new Ngraph Op has been added
  static std::map<std::string, std::set<shared_ptr<ov::Node>>> TFtoNgraphOpMap{
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
        std::make_shared<opset::Subtract>(), std::make_shared<opset::Log>()}},
      {"LeakyRelu", {std::make_shared<opset::PRelu>()}},
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
        std::make_shared<opset::Equal>(), std::make_shared<opset::Select>()}},
      {"Unpack", {constant, std::make_shared<opset::StridedSlice>()}},
      {"ZerosLike", {constant}},
      {"NoOp", {}},
  };

  return TFtoNgraphOpMap;
}

bool NodeIsMarkedForClustering(const Node* node) {
  bool is_marked;
  // TODO(amprocte): move attr name to a constant
  return (GetNodeAttr(node->attrs(), "_ovtf_marked_for_clustering",
                      &is_marked) == Status::OK() &&
          is_marked);
}

void GetStaticInputs(const Node* node, std::vector<int32>* inputs) {
  if (GetNodeAttr(node->attrs(), "_ovtf_static_inputs", inputs) !=
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

        OVTF_VLOG(5) << "For arg " << index << " checking edge "
                     << edge->DebugString();

        if (InputIsStatic(edge->dst(), edge->dst_input())) {
          OVTF_VLOG(5) << "Marking edge static: " << edge->DebugString();
          static_input_indexes->push_back(index);
          break;
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
