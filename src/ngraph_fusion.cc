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

#include "ngraph_fusion.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

// The functions here provide support for fusion of XLA graphs to
// coarser-grained nGraph primitives for ops like pooling, convolution,
// and others. The general name of the game here is to recognize patterns in
// the graph that match those created from high-level TF ops by the kernels in
// tensorflow/compiler/tf2xla/kernels. These subgraphs are then fused to a
// "kFusion::kCustom" HLO op, with a mapping maintained from our newly created
// fusion ops to emitter objects that will later be called by the nGraph
// generation pass.
//
// It is VERY IMPORTANT TO NOTE that the fusion functions here are tailored to
// recognize *exactly* the graphs produced by the tf2xla kernels, and to ignore
// anything not matching those patterns. Even a small change in the HLO graph,
// like reversing the order of operands to the "plus" in the embedded
// computation for "sum", will result in a fusion opportunity being missed. For
// this reason, it is intended that the NGraphFusion pass be run at the very
// beginning of the HLO compilation pipeline, BEFORE any other HLO-level
// optimizations.

namespace xla {
namespace ngraph_plugin {

class NGraphFusionVisitor : public DfsHloVisitorWithDefault {
 public:
  // Action that we will run on all nodes in the graph. (Because most of the
  // fusion cases are concerned with complex subgraphs where it's not always
  // obvious from the outside what the root node should be, we run everything
  // through one default action rather than using the opcode-specific handlers
  // supported by DfsHloVisitor).
  Status DefaultAction(HloInstruction* hlo_instruction) override;

  // Returns whether nGraph fusion has occurred.
  const bool changed() const { return m_changed; }

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation,
                  NGraphEmitter::FusedOpMap* fusion_map);

 private:
  // Fuses the supplied instructions and adds the supplied emitter to the
  // fusion map. The instructions must be in reverse topological order,
  // starting at the root of the subgraph to be fused.
  void FuseInstructions(
      const NGraphEmitter::FusedOpEmitter* emitter,
      tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse);

  // Each of these functions attempts to apply a particular fusion rule on the
  // subgraph rooted at the given instruction. If successful, it creates an
  // NGraphEmitter for the matching subgraph, calls FuseInstructions on that
  // emitter and the matched subgraph (thus adding the emitter to the fused op
  // map), and returns true. If matching is not successful, returns false.
  bool FuseReduce(HloInstruction* instruction);
  bool FuseMaxPoolFprop(HloInstruction* instruction);
  bool FuseAvgPoolFprop(HloInstruction* instruction);
  bool FuseMaxPoolBprop(HloInstruction* instruction);
  bool FuseAvgPoolBprop(HloInstruction* instruction);
  bool FuseConvolutionBpropData(HloInstruction* instruction);
  bool FuseConvolutionBpropFilters(HloInstruction* instruction);
  bool FuseRelu(HloInstruction* instruction);
  bool FuseReluBprop(HloInstruction* instruction);

  explicit NGraphFusionVisitor(HloComputation* computation,
                               NGraphEmitter::FusedOpMap* fusion_map)
      : m_computation(computation), m_fusion_map(fusion_map) {}

  // Current HloComputation instance the NGraphFusionVisitor is
  // traversing.
  HloComputation* m_computation;

  // Whether nGraph fusion has occurred.
  bool m_changed = false;

  // The fusion map we are adding to.
  NGraphEmitter::FusedOpMap* m_fusion_map;
};

Status NGraphFusionVisitor::DefaultAction(HloInstruction* hlo_instruction) {
  // Try all the fusion rules, returning as soon as the first one succeeds
  // (we are exploiting short-circuit evaluation of || here).
  m_changed |=
      (FuseReduce(hlo_instruction) || FuseMaxPoolFprop(hlo_instruction) ||
       FuseAvgPoolFprop(hlo_instruction) || FuseMaxPoolBprop(hlo_instruction) ||
       FuseAvgPoolBprop(hlo_instruction) ||
       FuseConvolutionBpropData(hlo_instruction) ||
       FuseConvolutionBpropFilters(hlo_instruction) ||
       FuseRelu(hlo_instruction) || FuseReluBprop(hlo_instruction));
  return Status::OK();
}

void NGraphFusionVisitor::FuseInstructions(
    const NGraphEmitter::FusedOpEmitter* emitter,
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse) {
  const HloInstruction* fused = m_computation->CreateFusionInstruction(
      instructions_to_fuse, HloInstruction::FusionKind::kCustom);
  (*m_fusion_map)[fused] =
      std::unique_ptr<const NGraphEmitter::FusedOpEmitter>(emitter);
}

bool NGraphFusionVisitor::Run(HloComputation* computation,
                              NGraphEmitter::FusedOpMap* fusion_map) {
  NGraphFusionVisitor visitor(computation, fusion_map);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.m_changed;
}

StatusOr<bool> NGraphFusion::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "NGraphFusion::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (NGraphFusionVisitor::Run(comp, m_fusion_map)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "NGraphFusion::Run(), after:\n" + module->ToString());
  return changed;
}

// A common helper function to detect whether a function just trivially applies
// a single op to its two arguments (as in, for example, f(x) = x+y).
static inline bool ComputationIsTrivialBinop(
    const HloComputation* computation) {
  const HloInstruction* instr = computation->root_instruction();
  const std::vector<HloInstruction*>& params =
      computation->parameter_instructions();

  return (computation->parameter_instructions().size() == 2 &&
          instr->operand(0) == params[0] && instr->operand(1) == params[1] &&
          params[0] != params[1]);
}

// ============================
//
// Reduction fusion
//
// ============================
//
// Match the following tree:
//
//      const(x)  arg  computation
//              \  |  /
//              reduce
//
// where computation is:
//
//   param0     param1 [distinct]
//         \   /
//          op
//
// Where "op" is a supported reduction op, and x is the identity element
// for "op".
bool NGraphFusionVisitor::FuseReduce(HloInstruction* reduce) {
  if (reduce->opcode() != HloOpcode::kReduce) {
    return false;
  }

  HloInstruction* init_value = reduce->mutable_operand(1);
  HloComputation* function = reduce->to_apply();

  if (init_value->opcode() != HloOpcode::kConstant ||
      !ComputationIsTrivialBinop(function)) {
    return false;
  }

  switch (function->root_instruction()->opcode()) {
    case HloOpcode::kAdd:
      if (!init_value->literal().IsAll(0)) {
        return false;
      }
      break;
    case HloOpcode::kMultiply:
      if (!init_value->literal().IsAll(1)) {
        return false;
      }
      break;
    case HloOpcode::kMaximum:
      if (init_value->literal() !=
          Literal::MinValue(init_value->literal().shape().element_type())) {
        return false;
      }
      break;
    default:
      return false;
  }

  auto emitter = new NGraphEmitter::FusedReductionEmitter(
      function->root_instruction()->opcode(), reduce->dimensions());
  FuseInstructions(emitter, {reduce, init_value});
  return true;
}

// Infers which of the supported data formats (NCHW or NHWC) is implied by
// the window parameters; returns true if this can be inferred, else
// returns false.
static inline bool ReconstructDataFormatFromPoolingWindow(
    const Window& window, tensorflow::TensorFormat& data_format) {
  // If we don't have enough dimensions for a batched pooling op, return false.
  if (window.dimensions_size() < 3) {
    return false;
  }

  // Both the batch and channel dimension will have to have size, stride, and
  // dilations of 1, and padding of 0.
  auto check_dimension = [](const WindowDimension& dim) -> bool {
    return (dim.size() == 1 && dim.stride() == 1 && dim.padding_low() == 0 &&
            dim.padding_high() == 0 && dim.window_dilation() == 1 &&
            dim.base_dilation() == 1);
  };

  // The batch dimension will be at position 0 either way.
  if (!check_dimension(window.dimensions(0))) {
    return false;
  }

  // The channel dimension will be either the last position (NHWC) or the
  // second (NCHW).
  if (check_dimension(window.dimensions(window.dimensions_size() - 1))) {
    data_format = tensorflow::FORMAT_NHWC;
    return true;
  } else if (check_dimension(window.dimensions(1))) {
    data_format = tensorflow::FORMAT_NCHW;
    return true;
  } else {
    return false;
  }
}

// ============================
//
// Max-pool fusion
//
// ============================
//
// Try to match max-pool fprop
//
//    operand   init_val computation
//           \     |    /
//            \    |   /
//           reduce_window (has ksize, stride, padding)
//
// where computation is:
//
//   param0     param1 [distinct]
//         \   /
//          max
bool NGraphFusionVisitor::FuseMaxPoolFprop(HloInstruction* root) {
  if (root->opcode() != HloOpcode::kReduceWindow) {
    return false;
  }

  HloInstruction* init_value = root->mutable_operand(1);
  HloComputation* function = root->to_apply();

  if (init_value->opcode() != HloOpcode::kConstant ||
      !ComputationIsTrivialBinop(function) ||
      function->root_instruction()->opcode() != HloOpcode::kMaximum) {
    return false;
  }

  const Literal& literal = init_value->literal();

  if (literal != Literal::MinValue(literal.shape().element_type())) {
    return false;
  }

  // See if we can determine from the window shape whether the data batch
  // shape is NCHW or NHWC, and bail if neither one makes sense.
  const Window& window = root->window();

  tensorflow::TensorFormat data_format;
  if (!ReconstructDataFormatFromPoolingWindow(window, data_format)) {
    return false;
  }

  auto emitter = new NGraphEmitter::FusedMaxPoolEmitter(
      window, data_format == tensorflow::FORMAT_NCHW);
  FuseInstructions(emitter, {root, init_value});
  return true;
}

// Match the following tree:
//
//      input deltas const(0) comput0 comput1
//           \     |    |   __/  _____/
//            \    |    |  /    /
//            select_and_scatter
//
// where comput0 (select) is trivial ">=" and comput1 (scatter) is trivial "add"
bool NGraphFusionVisitor::FuseMaxPoolBprop(HloInstruction* root) {
  if (root->opcode() != HloOpcode::kSelectAndScatter) {
    return false;
  }

  HloInstruction* init_value = root->mutable_operand(2);

  HloComputation* comput_select = root->select();
  HloComputation* comput_scatter = root->scatter();

  if (init_value->opcode() != HloOpcode::kConstant ||
      !init_value->literal().IsAll(0)) {
    return false;
  }

  if (!ComputationIsTrivialBinop(comput_select) ||
      comput_select->root_instruction()->opcode() != HloOpcode::kGe) {
    return false;
  }

  if (!ComputationIsTrivialBinop(comput_scatter) ||
      comput_scatter->root_instruction()->opcode() != HloOpcode::kAdd) {
    return false;
  }

  // See if we can determine from the window shape whether the data batch
  // shape is NCHW or NHWC, and bail if neither one makes sense.
  const Window& window = root->window();

  tensorflow::TensorFormat data_format;
  if (!ReconstructDataFormatFromPoolingWindow(window, data_format)) {
    return false;
  }

  // TODO(amprocte): find matching forward-prop op
  auto emitter = new NGraphEmitter::FusedMaxPoolBackpropEmitter(
      window, data_format == tensorflow::FORMAT_NCHW);
  FuseInstructions(emitter, {root, init_value});
  return true;
}

// ============================
//
// Avg-pool fusion
//
// ============================
//
// The patterns for average pool fusion are considerably more complex than
// max-pool, so we will use a number of helper functions here. TODO: add more
// detail on the matched patterns.

// Infers which of the supported data formats (NCHW or NHWC) is implied by
// the padding config from the "Pad" op embedded inside an AvgPool backprop.
// Returns true if this can be inferred, else returns false.
//
// TODO(amprocte): Is this actually unambiguous? Is it possible that both
// conditions (the NCHW condition and the NHWC condition) will apply?
static inline bool ReconstructDataFormatFromAvgPoolBackpropPadding(
    const PaddingConfig& padding, tensorflow::TensorFormat& data_format) {
  // If we don't have enough dimensions for a batched pooling op, return false.
  if (padding.dimensions_size() < 3) {
    return false;
  }

  // Both the batch and channel dimension will have to have zero padding
  // everywhere.
  auto check_dimension =
      [](const PaddingConfig::PaddingConfigDimension& dim) -> bool {
    return (dim.edge_padding_low() == 0 && dim.edge_padding_high() == 0 &&
            dim.interior_padding() == 0);
  };

  // The batch dimension will be at position 0 either way.
  if (!check_dimension(padding.dimensions(0))) {
    return false;
  }

  // The channel dimension will be either the last position (NHWC) or the
  // second (NCHW).
  if (check_dimension(padding.dimensions(padding.dimensions_size() - 1))) {
    data_format = tensorflow::FORMAT_NHWC;
    return true;
  } else if (check_dimension(padding.dimensions(1))) {
    data_format = tensorflow::FORMAT_NCHW;
    return true;
  } else {
    return false;
  }
}

static bool MatchAvgPoolDivideByCountRHSOnesPostBroadcast(
    HloInstruction* root, std::vector<HloInstruction*>& matched_instructions) {
  if (root->opcode() == HloOpcode::kConstant && root->literal().IsAll(1)) {
    matched_instructions.push_back(root);
    return true;
  }

  return false;
}

static bool MatchAvgPoolDivideByCountRHSOnes(
    HloInstruction* root, std::vector<HloInstruction*>& matched_instructions) {
  if (root->opcode() == HloOpcode::kBroadcast) {
    matched_instructions.push_back(root);
    return MatchAvgPoolDivideByCountRHSOnesPostBroadcast(
        root->mutable_operand(0), matched_instructions);
  }

  return MatchAvgPoolDivideByCountRHSOnesPostBroadcast(root,
                                                       matched_instructions);
}

static bool MatchAvgPoolDivideByCountRHSPostBroadcast(
    HloInstruction* root, std::vector<HloInstruction*>& matched_instructions) {
  if (root->opcode() != HloOpcode::kReduceWindow) return false;

  HloComputation* comput = root->to_apply();
  if (!ComputationIsTrivialBinop(comput) ||
      comput->root_instruction()->opcode() != HloOpcode::kAdd)
    return false;

  HloInstruction* init_value = root->mutable_operand(1);
  if (init_value->opcode() != HloOpcode::kConstant ||
      !(init_value->literal().IsAll(0)))
    return false;

  matched_instructions.push_back(root);
  matched_instructions.push_back(init_value);

  return MatchAvgPoolDivideByCountRHSOnes(root->mutable_operand(0),
                                          matched_instructions);
}

static bool MatchBroadcastedConstant(
    HloInstruction* root, std::vector<HloInstruction*>& matched_instructions) {
  if (root->opcode() == HloOpcode::kConstant) {
    matched_instructions.push_back(root);
    return true;
  } else if (root->opcode() == HloOpcode::kBroadcast &&
             root->mutable_operand(0)->opcode() == HloOpcode::kConstant) {
    matched_instructions.push_back(root);
    matched_instructions.push_back(root->mutable_operand(0));
    return true;
  }

  return false;
}

static bool MatchAvgPoolDivideByCountRHS(
    HloInstruction* root, std::vector<HloInstruction*>& matched_instructions,
    bool& is_valid_padding) {
  if (MatchBroadcastedConstant(root, matched_instructions)) {
    is_valid_padding = true;
    return true;
  }

  is_valid_padding = false;

  if (root->opcode() == HloOpcode::kBroadcast) {
    matched_instructions.push_back(root);
    return MatchAvgPoolDivideByCountRHSPostBroadcast(root->mutable_operand(0),
                                                     matched_instructions);
  }

  return MatchAvgPoolDivideByCountRHSPostBroadcast(root, matched_instructions);
}

static bool MatchAvgPoolFpropLHS(
    HloInstruction* root, std::vector<HloInstruction*>& matched_instructions) {
  if (root->opcode() != HloOpcode::kReduceWindow) return false;

  HloComputation* comput = root->to_apply();
  HloInstruction* init_value = root->mutable_operand(1);

  if (ComputationIsTrivialBinop(comput) &&
      comput->root_instruction()->opcode() == HloOpcode::kAdd &&
      init_value->opcode() == HloOpcode::kConstant &&
      init_value->literal().IsAll(0)) {
    matched_instructions.push_back(root);
    matched_instructions.push_back(init_value);
    return true;
  }

  return false;
}

static bool MatchAvgPoolBpropReduceWindowOperand(
    HloInstruction* root, std::vector<HloInstruction*>& matched_instructions,
    bool& is_valid_padding) {
  if (root->opcode() != HloOpcode::kPad) return false;

  matched_instructions.push_back(root);

  HloInstruction* operand = root->mutable_operand(0);
  if (operand->opcode() != HloOpcode::kDivide) return false;

  matched_instructions.push_back(operand);

  HloInstruction* pad_value = root->mutable_operand(1);
  if (pad_value->opcode() != HloOpcode::kConstant ||
      !(pad_value->literal().IsAll(0)))
    return false;

  matched_instructions.push_back(pad_value);

  return MatchAvgPoolDivideByCountRHS(operand->mutable_operand(1),
                                      matched_instructions, is_valid_padding);
}

static bool MatchAvgPoolBprop(
    HloInstruction* root, std::vector<HloInstruction*>& matched_instructions,
    bool& is_valid_padding) {
  if (root->opcode() != HloOpcode::kReduceWindow) return false;

  HloComputation* comput = root->to_apply();
  if (!ComputationIsTrivialBinop(comput) ||
      comput->root_instruction()->opcode() != HloOpcode::kAdd)
    return false;

  HloInstruction* init_value = root->mutable_operand(1);
  if (init_value->opcode() != HloOpcode::kConstant ||
      !(init_value->literal().IsAll(0)))
    return false;

  matched_instructions.push_back(root);
  matched_instructions.push_back(init_value);

  return MatchAvgPoolBpropReduceWindowOperand(
      root->mutable_operand(0), matched_instructions, is_valid_padding);
}

bool NGraphFusionVisitor::FuseAvgPoolFprop(HloInstruction* root) {
  // First check: the root is always an "AvgPoolDivideByCount".
  //
  //  lhs   rhs
  //    \   /
  //     div
  if (root->opcode() != HloOpcode::kDivide) {
    return false;
  }

  // Now check if the LHS and the RHS match the expected pattern for fprop.
  // Ops will be added by the callees to matched_instructions.
  std::vector<HloInstruction*> matched_instructions{root};

  HloInstruction* lhs = root->mutable_operand(0);
  if (!MatchAvgPoolFpropLHS(lhs, matched_instructions)) {
    return false;
  }

  HloInstruction* rhs = root->mutable_operand(1);
  bool is_valid_padding;  // ignored for fprop, but returned by ref from matcher

  if (!MatchAvgPoolDivideByCountRHS(rhs, matched_instructions,
                                    is_valid_padding)) {
    return false;
  }

  // See if we can determine from the window shape whether the data batch
  // shape is NCHW or NHWC, and bail if neither one makes sense.
  const Window& window = lhs->window();

  tensorflow::TensorFormat data_format;
  if (!ReconstructDataFormatFromPoolingWindow(window, data_format)) {
    return false;
  }

  auto emitter = new NGraphEmitter::FusedAvgPoolEmitter(
      window, data_format == tensorflow::FORMAT_NCHW);
  FuseInstructions(emitter, matched_instructions);
  return true;
}

bool NGraphFusionVisitor::FuseAvgPoolBprop(HloInstruction* root) {
  if (root->opcode() != HloOpcode::kReduceWindow) {
    return false;
  }

  // Instructions will be added by the callee to matched_instructions.
  std::vector<HloInstruction*> matched_instructions;
  bool is_valid_padding;  // returned by reference from MatchAvgPoolBprop

  if (!MatchAvgPoolBprop(root, matched_instructions, is_valid_padding)) {
    return false;
  }

  // If we are here, we know that there is a "pad" op coming into the
  // ReduceWindow, and this holds the info we need, along with
  // is_valid_padding, to reconstruct the forward window parameters.
  const HloInstruction* pad = root->mutable_operand(0);

  // See if we can determine from the padding config whether the data batch
  // shape is NCHW or NHWC, and bail if neither one makes sense.
  const PaddingConfig& padding = pad->padding_config();

  tensorflow::TensorFormat data_format;
  if (!ReconstructDataFormatFromAvgPoolBackpropPadding(padding, data_format)) {
    return false;
  }

  // Unlike, e.g., max-pool, we have to reconstruct the window here, since
  // this info is not represented unambiguously anywhere in the graph.
  size_t spatial_dims_start = (data_format == tensorflow::FORMAT_NCHW ? 2 : 1);
  size_t spatial_dim_count = root->shape().dimensions_size() - 2;
  size_t batch_axis = 0;
  size_t channel_axis = (data_format == tensorflow::FORMAT_NCHW
                             ? 1
                             : spatial_dims_start + spatial_dim_count);

  // Start by reserving spatial_dim_count + 2 dimensions (+2 is for batch and
  // channel dims)
  Window window;

  for (size_t i = 0; i < spatial_dim_count + 2; i++) {
    window.add_dimensions();
  }

  // Populate batch and channel dimensions with unit size/stride and no
  // padding.
  window.mutable_dimensions(batch_axis)->set_size(1);
  window.mutable_dimensions(batch_axis)->set_stride(1);
  window.mutable_dimensions(batch_axis)->set_padding_low(0);
  window.mutable_dimensions(batch_axis)->set_padding_high(0);
  window.mutable_dimensions(batch_axis)->set_window_dilation(1);
  window.mutable_dimensions(batch_axis)->set_base_dilation(1);

  window.mutable_dimensions(channel_axis)->set_size(1);
  window.mutable_dimensions(channel_axis)->set_stride(1);
  window.mutable_dimensions(channel_axis)->set_padding_low(0);
  window.mutable_dimensions(channel_axis)->set_padding_high(0);
  window.mutable_dimensions(channel_axis)->set_window_dilation(1);
  window.mutable_dimensions(channel_axis)->set_base_dilation(1);

  // Set the parameters for each spatial dimension.
  for (size_t i = 0; i < spatial_dim_count; i++) {
    auto dim = window.mutable_dimensions(i + spatial_dims_start);
    dim->set_size(root->window().dimensions(i + spatial_dims_start).size());
    dim->set_stride(
        padding.dimensions(i + spatial_dims_start).interior_padding() + 1);

    if (is_valid_padding) {
      dim->set_padding_low(0);
      dim->set_padding_high(0);
    } else {
      size_t padding_needed = dim->size() - 1;
      dim->set_padding_low(padding_needed / 2);
      dim->set_padding_high(padding_needed - dim->padding_low());
    }

    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }

  auto emitter = new NGraphEmitter::FusedAvgPoolBackpropEmitter(
      root->shape(), window, data_format == tensorflow::FORMAT_NCHW);
  FuseInstructions(emitter, matched_instructions);
  return true;
}

// ============================
//
// Convolution backprop fusion
//
// ============================

// Given the TF convolution dimensions, attempts to reconstruct the data
// format (NCHW or NHWC). Returns false if the dimensions are not consistent
// with NCHW; otherwise, updates "data_format" by reference.
static bool ReconstructConvolutionDataFormatForBpropData(
    const ConvolutionDimensionNumbers& dnums,
    tensorflow::TensorFormat& data_format) {
  if (dnums.input_batch_dimension() == 0 &&
      dnums.input_feature_dimension() == 1) {
    data_format = tensorflow::FORMAT_NCHW;
    return true;
  } else if (dnums.input_batch_dimension() == 0 &&
             dnums.input_feature_dimension() ==
                 dnums.input_spatial_dimensions_size() + 1) {
    data_format = tensorflow::FORMAT_NHWC;
    return true;
  } else {
    return false;
  }
}

// Given the known input, filter, and output shapes, attempts to reconstruct
// what the padding type was (either SAME or VALID). Returns false if the
// shapes are consistent with neither SAME nor VALID, otherwise updates
// "padding_type" by reference.
static bool ReconstructConvolutionPaddingType(
    tensorflow::TensorFormat data_format,
    const std::vector<int64>& forward_strides,
    const std::vector<int64>& forward_input_dims,
    const std::vector<int64>& forward_output_dims,
    const std::vector<int64>& forward_filter_dims,
    tensorflow::Padding& padding_type) {
  if (forward_input_dims.size() < 3 ||
      forward_input_dims.size() != forward_output_dims.size() ||
      forward_input_dims.size() != forward_filter_dims.size() ||
      forward_strides.size() != forward_input_dims.size() - 2) {
    return false;
  }

  size_t num_spatial_dims = forward_input_dims.size() - 2;

  // We only support NCHW and NHWC. Other formats will be left untouched.
  if (data_format == tensorflow::FORMAT_NCHW ||
      data_format == tensorflow::FORMAT_NHWC) {
    size_t batch_n_axis = 0;
    size_t batch_c_axis =
        (data_format == tensorflow::FORMAT_NCHW) ? 1 : num_spatial_dims + 1;
    size_t batch_first_spatial_axis =
        (data_format == tensorflow::FORMAT_NCHW) ? 2 : 1;
    size_t filter_ci_axis = num_spatial_dims;
    size_t filter_co_axis = num_spatial_dims + 1;

    // If the filter and input have an inconsistent shape, give up.
    if (forward_input_dims[batch_c_axis] !=
        forward_filter_dims[filter_ci_axis]) {
      return false;
    }

    // Test whether the sizes are consistent with "same" padding.
    std::vector<int64> same_expected_output_dims(num_spatial_dims + 2);

    same_expected_output_dims[batch_n_axis] = forward_input_dims[batch_n_axis];
    same_expected_output_dims[batch_c_axis] =
        forward_filter_dims[filter_co_axis];

    for (int64 i = 0; i < int64(num_spatial_dims); i++) {
      int64 batch_pos = i + batch_first_spatial_axis;
      same_expected_output_dims[batch_pos] = tensorflow::MathUtil::CeilOfRatio(
          forward_input_dims[batch_pos], forward_strides[i]);
    }

    if (forward_output_dims == same_expected_output_dims) {
      padding_type = tensorflow::Padding::SAME;
      return true;
    }

    // Test whether the sizes are consistent with "valid" padding.
    std::vector<int64> valid_expected_output_dims(num_spatial_dims + 2);

    valid_expected_output_dims[batch_n_axis] = forward_input_dims[batch_n_axis];
    valid_expected_output_dims[batch_c_axis] =
        forward_filter_dims[filter_co_axis];

    for (size_t i = 0; i < num_spatial_dims; i++) {
      size_t batch_pos = i + batch_first_spatial_axis;
      valid_expected_output_dims[batch_pos] = tensorflow::MathUtil::CeilOfRatio(
          forward_input_dims[batch_pos] - forward_filter_dims[i] + 1,
          forward_strides[i]);
    }

    if (forward_output_dims == valid_expected_output_dims) {
      padding_type = tensorflow::Padding::VALID;
      return true;
    }

    // Neither "same" nor "valid" matches, so we will give up.
    return false;
  }

  else {
    return false;
  }
}

// Given an checks that the op is a reverse op that applies to the kernel
// spatial dims, according to the TF convention that kernel shape is "HWio".
static bool RevAppliesToFilterSpatialDims(HloInstruction* rev) {
  if (rev->opcode() != HloOpcode::kReverse) {
    return false;
  }

  size_t num_spatial_dims = rev->shape().dimensions_size() - 2;

  const std::vector<int64>& dimensions = rev->dimensions();

  if (dimensions.size() != num_spatial_dims) {
    return false;
  }

  std::vector<bool> included(num_spatial_dims, false);

  for (int64 d : dimensions) {
    if (d < 0 || d >= int64(num_spatial_dims)) {
      return false;
    }
    included[d] = true;
  }

  for (bool b : included) {
    if (!b) {
      return false;
    }
  }

  return true;
}

// Returns true if and only if every element of the container is equal to 1,
// else returns false.
template <typename T>
bool AllOnes(const T& container) {
  for (const auto& x : container) {
    if (x != 1) {
      return false;
    }
  }

  return true;
}

// Check for convolution backprop to data batch:
//
//            filter
//              |
//  out_delta  Rev (of all spatial dims)
//     \      /
//   Convolution (where in and out channels are flipped from usual)
//        |
//   input_delta
//
// Where for the convolution op: window strides and rhs_dilation are all
// ones, and the shapes of the various inputs and outputs have either
// the expected values from kSame or kValid that would be produced
// with strides equal to the lhs_dilation of the convolution op.
bool NGraphFusionVisitor::FuseConvolutionBpropData(HloInstruction* root) {
  if (root->opcode() != HloOpcode::kConvolution) {
    return false;
  }

  HloInstruction* lhs = root->mutable_operand(0);
  HloInstruction* rhs = root->mutable_operand(1);

  // The rhs must be a reverse applied to the spatial dimensions of the filter.
  if (rhs->opcode() != HloOpcode::kReverse ||
      !RevAppliesToFilterSpatialDims(rhs)) {
    return false;
  }

  // The input and output feature dimensions for the (spatially reversed)
  // filter must be switched from the usual interpretation: "HWoi" instead
  // of TF's usual HWio.
  const ConvolutionDimensionNumbers& dims =
      root->convolution_dimension_numbers();
  size_t num_spatial_dims = dims.input_spatial_dimensions().size();

  if (dims.kernel_output_feature_dimension() != int64(num_spatial_dims) ||
      dims.kernel_input_feature_dimension() != int64(num_spatial_dims) + 1) {
    return false;
  }

  // Attempt to reconstruct the data format from the dims, give up on fusion if
  // we can't.
  tensorflow::TensorFormat data_format;  // returned by reference

  if (!ReconstructConvolutionDataFormatForBpropData(dims, data_format)) {
    return false;
  }

  // Reconstruct the forward strides/dilation, and the backward dilation.
  std::vector<int64> forward_window_strides;
  std::vector<int64> backward_window_strides;
  std::vector<int64> backward_window_dilation;

  for (auto dim : root->window().dimensions()) {
    forward_window_strides.push_back(dim.base_dilation());
    backward_window_strides.push_back(dim.stride());
    backward_window_dilation.push_back(dim.window_dilation());
  }

  // The backward window dilation and backward window strides must be all ones.
  if (!AllOnes(backward_window_dilation) || !AllOnes(backward_window_strides)) {
    return false;
  }

  // Extract the input/output and filter shapes to vectors.
  std::vector<int64> forward_input_shape(root->shape().dimensions().begin(),
                                         root->shape().dimensions().end());
  std::vector<int64> forward_output_shape(lhs->shape().dimensions().begin(),
                                          lhs->shape().dimensions().end());
  std::vector<int64> forward_filter_shape(rhs->shape().dimensions().begin(),
                                          rhs->shape().dimensions().end());

  // Attempt to reconstruct the padding type (either SAME or VALID), and give
  // up on fusion if we can't.
  tensorflow::Padding padding_type;  // returned by reference

  if (!ReconstructConvolutionPaddingType(
          data_format, forward_window_strides, forward_input_shape,
          forward_output_shape, forward_filter_shape, padding_type)) {
    return false;
  }

  // Construct the emitter parameters for nGraph.
  ngraph::Strides ng_forward_window_movement_strides;
  for (auto s : forward_window_strides) {
    ng_forward_window_movement_strides.push_back(size_t(s));
  }

  size_t input_first_spatial_axis =
      (data_format == tensorflow::FORMAT_NCHW ? 2 : 1);

  ngraph::CoordinateDiff ng_forward_padding_below;
  ngraph::CoordinateDiff ng_forward_padding_above;

  for (size_t i = 0; i < num_spatial_dims; i++) {
    if (padding_type == tensorflow::Padding::VALID) {
      ng_forward_padding_below.push_back(0);
      ng_forward_padding_above.push_back(0);
    } else {
      size_t image_size = forward_input_shape[i + input_first_spatial_axis];
      size_t filter_shape = forward_filter_shape[i];
      size_t filter_stride = forward_window_strides[i];

      ssize_t padding_needed;
      if (image_size % filter_stride == 0) {
        padding_needed = filter_shape - filter_stride;
      } else {
        padding_needed = filter_shape - (image_size % filter_stride);
      }
      if (padding_needed < 0) {
        padding_needed = 0;
      }

      size_t padding_lhs = padding_needed / 2;
      size_t padding_rhs = padding_needed - padding_lhs;
      ng_forward_padding_below.push_back(padding_lhs);
      ng_forward_padding_above.push_back(padding_rhs);
    }
  }

  ngraph::Shape ng_forward_input_shape(forward_input_shape.begin(),
                                       forward_input_shape.end());

  // Build the emitter and put it into the map.
  auto emitter = new NGraphEmitter::FusedConvBackpropInputEmitter(
      ng_forward_input_shape, ng_forward_window_movement_strides,
      ng_forward_padding_below, ng_forward_padding_above,
      (data_format == tensorflow::FORMAT_NCHW));

  FuseInstructions(emitter, {root, rhs});
  return true;
}

// Given the TF convolution dimensions, attempts to reconstruct the data
// format (NCHW or NHWC). Returns false if the dimensions are not consistent
// with NCHW; otherwise, updates "data_format" by reference.
//
// This is identical to ReconstructConvolutionDataFormat, except that the
// roles (and therefore expected positions) of the batch and channel axes
// are reversed.
static bool ReconstructConvolutionDataFormatForBpropFilters(
    const ConvolutionDimensionNumbers& dnums,
    tensorflow::TensorFormat& data_format) {
  if (dnums.input_feature_dimension() == 0 &&
      dnums.input_batch_dimension() == 1) {
    data_format = tensorflow::FORMAT_NCHW;
    return true;
  } else if (dnums.input_feature_dimension() == 0 &&
             dnums.input_batch_dimension() ==
                 dnums.input_spatial_dimensions_size() + 1) {
    data_format = tensorflow::FORMAT_NHWC;
    return true;
  } else {
    return false;
  }
}

// Given a set of convolution dimension numbers, check whether the output
// format matches the expected, i.e. "HWNC". (Here the N really corresponds
// to the input-channel dimension and C to the output-channel dimension of
// the filters.
static bool CheckOutputDimensionsForBpropFilters(
    const ConvolutionDimensionNumbers& dnums) {
  return (dnums.output_batch_dimension() ==
              dnums.output_spatial_dimensions_size() &&
          dnums.output_feature_dimension() ==
              dnums.output_spatial_dimensions_size() + 1);
}

// Check for convolution backprop to filters:
//
//  input   out_delta
//     \      /
//   Convolution
//        |
//    Transpose
//        |
//   filter_delta
//
// Where for the convolution op: window strides and lhs_dilation are all
// ones, and the shapes of the various inputs and outputs have either
// the expected values from kSame or kValid that would be produced
// with strides equal to the rhs_dilation of the convolution op.
//
// And where the "N" and "C" axes are swapped from their usual sense for
// input, and the "N" and "C" axes are respectively treated as the "Ci"
// and "Co" dimensions for out_delta.
//
// And where the transpose op is reshaping oHWi to HWio.
bool NGraphFusionVisitor::FuseConvolutionBpropFilters(
    HloInstruction* convolution) {
  if (convolution->opcode() != HloOpcode::kConvolution) {
    return false;
  }

  HloInstruction* input = convolution->mutable_operand(0);
  HloInstruction* out_delta = convolution->mutable_operand(1);

  const ConvolutionDimensionNumbers& dims =
      convolution->convolution_dimension_numbers();

  // Make sure the output format matches what's expected.
  if (!CheckOutputDimensionsForBpropFilters(dims)) {
    return false;
  }

  // Figure out the data format (either NCHW or NHWC), and give up on fusion if
  // we can't.
  tensorflow::TensorFormat data_format;  // returned by reference

  if (!ReconstructConvolutionDataFormatForBpropFilters(dims, data_format)) {
    return false;
  }

  // Reconstruct forward/backward strides, and backward image dilation.
  std::vector<int64> forward_window_strides;
  std::vector<int64> backward_window_strides;
  std::vector<int64> backward_image_dilation;

  for (auto dim : convolution->window().dimensions()) {
    forward_window_strides.push_back(dim.window_dilation());
    backward_window_strides.push_back(dim.stride());
    backward_image_dilation.push_back(dim.base_dilation());
  }

  // The backward window strides and dilation should be all 1.
  if (!AllOnes(backward_window_strides) || !AllOnes(backward_image_dilation)) {
    return false;
  }

  // Extract forward input/output and filter shapes.
  std::vector<int64> forward_input_shape(input->shape().dimensions().begin(),
                                         input->shape().dimensions().end());
  std::vector<int64> forward_output_shape(
      out_delta->shape().dimensions().begin(),
      out_delta->shape().dimensions().end());
  std::vector<int64> forward_filter_shape(
      convolution->shape().dimensions().begin(),
      convolution->shape().dimensions().end());

  // Try to reconstruct the padding type (SAME or VALID), give up on fusion if
  // we can't.
  tensorflow::Padding padding_type;  // returned by reference

  if (!ReconstructConvolutionPaddingType(
          data_format, forward_window_strides, forward_input_shape,
          forward_output_shape, forward_filter_shape, padding_type)) {
    return false;
  }

  // Construct the emitter parameters for nGraph.
  ngraph::Strides ng_forward_window_movement_strides;
  for (auto s : forward_window_strides) {
    ng_forward_window_movement_strides.push_back(size_t(s));
  }

  ngraph::CoordinateDiff ng_forward_padding_below;
  ngraph::CoordinateDiff ng_forward_padding_above;

  size_t input_first_spatial_axis =
      (data_format == tensorflow::FORMAT_NCHW ? 2 : 1);
  size_t num_spatial_dims = dims.input_spatial_dimensions().size();

  for (size_t i = 0; i < num_spatial_dims; i++) {
    if (padding_type == tensorflow::Padding::VALID) {
      ng_forward_padding_below.push_back(0);
      ng_forward_padding_above.push_back(0);
    } else {
      size_t image_size = forward_input_shape[i + input_first_spatial_axis];
      size_t filter_shape = forward_filter_shape[i];
      size_t filter_stride = forward_window_strides[i];

      ssize_t padding_needed;
      if (image_size % filter_stride == 0) {
        padding_needed = filter_shape - filter_stride;
      } else {
        padding_needed = filter_shape - (image_size % filter_stride);
      }
      if (padding_needed < 0) {
        padding_needed = 0;
      }
      size_t padding_lhs = padding_needed / 2;
      size_t padding_rhs = padding_needed - padding_lhs;
      ng_forward_padding_below.push_back(padding_lhs);
      ng_forward_padding_above.push_back(padding_rhs);
    }
  }

  ngraph::Shape ng_forward_filters_shape(forward_filter_shape.begin(),
                                         forward_filter_shape.end());

  // Build the emitter and put it into the map.
  auto emitter = new NGraphEmitter::FusedConvBackpropFiltersEmitter(
      ng_forward_filters_shape, ng_forward_window_movement_strides,
      ng_forward_padding_below, ng_forward_padding_above,
      (data_format == tensorflow::FORMAT_NCHW));

  FuseInstructions(emitter, {convolution});
  return true;
}

// ============================
//
// ReLU fusion
//
// ============================

static bool MatchBroadcastedZero(
    HloInstruction* root, std::vector<HloInstruction*>& matched_instructions) {
  if (root->opcode() == HloOpcode::kBroadcast) {
    matched_instructions.push_back(root);
    root = root->mutable_operand(0);
  }

  if (root->opcode() == HloOpcode::kConstant && root->literal().IsAll(0)) {
    matched_instructions.push_back(root);
    return true;
  }

  return false;
}

// Look for graphs of the form:
//
// const(0)
//    |
// broadcast    arg
//         \   /
//          max
//
// where the "broadcast" is optional.
bool NGraphFusionVisitor::FuseRelu(HloInstruction* root) {
  if (root->opcode() != HloOpcode::kMaximum) {
    return false;
  }

  std::vector<HloInstruction*> matched_instructions;
  matched_instructions.push_back(root);

  HloInstruction* lhs = root->mutable_operand(0);

  if (!MatchBroadcastedZero(lhs, matched_instructions)) {
    return false;
  }

  // Build the emitter and put it into the map.
  auto emitter = new NGraphEmitter::FusedReluEmitter();

  FuseInstructions(emitter, matched_instructions);
  return true;
}

// Look for graphs of the form:
//
//        const(0)
//           |
// arg1   broadcast
//     \ /       |
//     gt  arg0 /
//       \  |  /
//        select
bool NGraphFusionVisitor::FuseReluBprop(HloInstruction* root) {
  if (root->opcode() != HloOpcode::kSelect) {
    return false;
  }

  HloInstruction* pred = root->mutable_operand(0);

  if (pred->opcode() != HloOpcode::kGt) {
    return false;
  }

  HloInstruction* broadcasted_zeros = pred->mutable_operand(1);

  if (broadcasted_zeros->opcode() != HloOpcode::kBroadcast) {
    return false;
  }

  HloInstruction* zero = broadcasted_zeros->mutable_operand(0);

  if (zero->opcode() != HloOpcode::kConstant || !zero->literal().IsAll(0)) {
    return false;
  }

  // This is a bit unusual: the kernel uses aliasing here, and we will match
  // that pattern exactly.
  if (root->operand(2) != broadcasted_zeros) {
    return false;
  }

  // Build the emitter and put it into the map.
  auto emitter = new NGraphEmitter::FusedReluBackpropEmitter();

  FuseInstructions(emitter, {root, pred, broadcasted_zeros, zero});
  return true;
}

}  // namespace ngraph_plugin
}  // namespace xla
