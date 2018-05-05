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

#include "ngraph_emitter.h"

#include <cinttypes>
#include <memory>
#include <sstream>

#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/util.hpp"
#include "ngraph_autobroadcast.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"
#include "ngraph_xla_compat.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace ngraph_plugin {

//---------------------------------------------------------------------------
// NGraphEmitter:Initialize()
//---------------------------------------------------------------------------
Status NGraphEmitter::Initialize() {
  // Populate with unused parameters
  for (const auto& next_hlo_param : m_hlo_parameter_list) {
    auto parameter_number = next_hlo_param->parameter_number();
    const Shape& xla_shape = next_hlo_param->shape();
    auto ng_shape = ngraph::Shape(xla_shape.dimensions().begin(),
                                  xla_shape.dimensions().end());
    auto ng_op = std::shared_ptr<ngraph::Node>();
    switch (xla_shape.element_type()) {
      case F32:
        ng_op = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32,
                                                        ng_shape);
        break;
      case S32:
        ng_op = std::make_shared<ngraph::op::Parameter>(ngraph::element::i32,
                                                        ng_shape);
        break;
      case S64:
        ng_op = std::make_shared<ngraph::op::Parameter>(ngraph::element::i64,
                                                        ng_shape);
        break;
      case PRED:
        ng_op = std::make_shared<ngraph::op::Parameter>(
            ngraph::element::boolean, ng_shape);
        break;
      default:
        return InternalError(
            "Parameter data type: '%s' shape: '%s'",
            PrimitiveType_Name(xla_shape.element_type()).c_str(),
            ShapeUtil::HumanString(xla_shape).c_str());
        break;
    }
    m_parameter_number_map[parameter_number] = next_hlo_param;
    m_op_map[next_hlo_param] = ng_op;
  }
  return Status::OK();
}

//---------------------------------------------------------------------------
// NGraphEmitter:ProcessElementwiseUnary()
//---------------------------------------------------------------------------
Status NGraphEmitter::ProcessElementwiseUnary(HloInstruction* hlo,
                                              HloOpcode opcode) {
  NGRAPH_VLOG(2) << "NGraphEmitter::" << __func__ << ": "
                 << "HloInstruction " << hlo->ToString() << ", "
                 << "HloOpcode " << HloOpcodeString(opcode);

  // Look up the operands
  const auto& operands = hlo->operands();
  TF_RET_CHECK(operands.size() == 1);
  auto ng_operand = m_op_map.find(operands[0])->second;

  // Create ng_op
  auto ng_op = std::shared_ptr<ngraph::Node>();
  switch (opcode) {
    case HloOpcode::kAbs:
      ng_op = std::make_shared<ngraph::op::Abs>(ng_operand);
      break;
    case HloOpcode::kCeil:
      ng_op = std::make_shared<ngraph::op::Ceiling>(ng_operand);
      break;
    case HloOpcode::kExp:
      ng_op = std::make_shared<ngraph::op::Exp>(ng_operand);
      break;
    case HloOpcode::kFloor:
      ng_op = std::make_shared<ngraph::op::Floor>(ng_operand);
      break;
    case HloOpcode::kLog:
      ng_op = std::make_shared<ngraph::op::Log>(ng_operand);
      break;
    case HloOpcode::kNegate:
      ng_op = std::make_shared<ngraph::op::Negative>(ng_operand);
      break;
    case HloOpcode::kSign:
      ng_op = std::make_shared<ngraph::op::Sign>(ng_operand);
      break;
    case HloOpcode::kSin:
      ng_op = std::make_shared<ngraph::op::Sin>(ng_operand);
      break;
    case HloOpcode::kCos:
      ng_op = std::make_shared<ngraph::op::Cos>(ng_operand);
      break;
    case HloOpcode::kTanh:
      ng_op = std::make_shared<ngraph::op::Tanh>(ng_operand);
      break;
    case HloOpcode::kNot:
    case HloOpcode::kIsFinite:
    case HloOpcode::kReducePrecision:
    default:
      return Unimplemented("unary op '%s'", HloOpcodeString(opcode).c_str());
  }

  // Save ng_op in op_map
  m_op_map[hlo] = ng_op;
  m_instruction_list.push_back({hlo->ToString(), ng_op->get_name()});

  return tensorflow::Status::OK();
}

//---------------------------------------------------------------------------
// NGraphEmitter:ProcessElementwiseBinary()
//---------------------------------------------------------------------------
Status NGraphEmitter::ProcessElementwiseBinary(HloInstruction* hlo,
                                               HloOpcode opcode) {
  NGRAPH_VLOG(3) << "NGraphEmitter::" << __func__ << ": "
                 << "HloInstruction " << hlo->ToString();

  const auto& operands = hlo->operands();
  TF_RET_CHECK(operands.size() == 2);

  NGRAPH_VLOG(2) << "LHS operand: " << operands[0]->ToString() << ", "
                 << "RHS operand: " << operands[1]->ToString();

  // Look up the operands
  auto ng_lhs = m_op_map.find(operands[0])->second;
  auto ng_rhs = m_op_map.find(operands[1])->second;
  auto ng_op = std::shared_ptr<ngraph::Node>(nullptr);

  // Check and fix the shape compatibility i.e., the following:
  // 1. Ensure that both have the same shape
  // 2. If not, then see if one is a scalar. If so:
  //    - create a new  broadcase instruction to expand the scalar to equal to
  //    the other shape.
  //    - now replace the scalar operand with the new broadcast operand
  if (!ShapeUtil::Compatible(operands[0]->shape(), operands[1]->shape())) {
    auto ng_shape_lhs = ngraph::Shape(operands[0]->shape().dimensions().begin(),
                                      operands[0]->shape().dimensions().end());

    auto ng_shape_rhs = ngraph::Shape(operands[1]->shape().dimensions().begin(),
                                      operands[1]->shape().dimensions().end());

    try {
      // TODO: CLEANUP More graceful exit without try/catch
      auto auto_broadcaster =
          AutoBroadcast(ng_lhs, ng_shape_lhs, ng_rhs, ng_shape_rhs);
      ng_lhs = auto_broadcaster.lhs();
      ng_rhs = auto_broadcaster.rhs();
    } catch (...) {
      NGRAPH_VLOG(1)
          << "Exception creating AutoBroadcast class: HLO instruction: "
          << hlo->ToString();
      return InternalError("Exception creating AutoBroadcast");
    }
  }

  switch (opcode) {
    case HloOpcode::kAdd:
      ng_op = std::make_shared<ngraph::op::Add>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kSubtract:
      ng_op = std::make_shared<ngraph::op::Subtract>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kMultiply:
      ng_op = std::make_shared<ngraph::op::Multiply>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kDivide:
      ng_op = std::make_shared<ngraph::op::Divide>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kPower:
      ng_op = std::make_shared<ngraph::op::Power>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kMaximum:
      ng_op = std::make_shared<ngraph::op::Maximum>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kMinimum:
      ng_op = std::make_shared<ngraph::op::Minimum>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kRemainder:
      ng_op = std::make_shared<ngraph::op::Remainder>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kEq:
      ng_op = std::make_shared<ngraph::op::Equal>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kGt:
      ng_op = std::make_shared<ngraph::op::Greater>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kLt:
      ng_op = std::make_shared<ngraph::op::Less>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kNe:
      ng_op = std::make_shared<ngraph::op::NotEqual>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kGe:
      ng_op = std::make_shared<ngraph::op::GreaterEq>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kLe:
      ng_op = std::make_shared<ngraph::op::LessEq>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kAnd:
      ng_op = std::make_shared<ngraph::op::And>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kOr:
      ng_op = std::make_shared<ngraph::op::Or>(ng_lhs, ng_rhs);
      break;
    default:
      return Unimplemented("ProcessElementwiseBinary OPCODE: '%s'",
                           HloOpcodeString(opcode).c_str());
      break;
  }

  // Store this in the map
  m_op_map[hlo] = ng_op;
  m_instruction_list.push_back({hlo->ToString(), ng_op->get_name()});
  return Status::OK();
}

//---------------------------------------------------------------------------
// NGraphEmitter::ProcessConcatenate()
//---------------------------------------------------------------------------
Status NGraphEmitter::ProcessConcatenate(
    HloInstruction* concatenate,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  std::vector<std::shared_ptr<ngraph::Node>> ng_nodes;
  for (auto& next_operand : operands) {
    ng_nodes.push_back(m_op_map[next_operand]);
  }

  // Create the nGraph Concatename operator
  auto ng_op = std::make_shared<ngraph::op::Concat>(
      ng_nodes, concatenate->concatenate_dimension());

  // Save it
  m_op_map[concatenate] = ng_op;
  m_instruction_list.push_back({concatenate->ToString(), ng_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessConvert()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessConvert(HloInstruction* convert) {
  // Create the ngraph literal
  auto ng_operand = m_op_map.find(convert->operand(0))->second;
  auto ng_op = std::shared_ptr<ngraph::Node>();
  switch (convert->shape().element_type()) {
    case F32: {
      ng_op = std::make_shared<ngraph::op::Convert>(ng_operand,
                                                    ngraph::element::f32);
    } break;
    case S32: {
      ng_op = std::make_shared<ngraph::op::Convert>(ng_operand,
                                                    ngraph::element::i32);
    } break;
    case S64: {
      ng_op = std::make_shared<ngraph::op::Convert>(ng_operand,
                                                    ngraph::element::i64);
    } break;
    case PRED: {
      ng_op = std::make_shared<ngraph::op::Convert>(ng_operand,
                                                    ngraph::element::boolean);
    } break;
    default:
      return Unimplemented(
          "HandleConvert: data type '%s'",
          PrimitiveType_Name(convert->shape().element_type()).c_str());
      break;
  }

  m_op_map[convert] = ng_op;
  m_instruction_list.push_back({convert->ToString(), ng_op->get_name()});
  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter:ProcessReverse()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessReverse(HloInstruction* reverse,
                                     const HloInstruction* operand) {
  ngraph::AxisSet ng_axis_set;
  for (auto dim : reverse->dimensions()) {
    ng_axis_set.insert(dim);
  }

  // Create the ngraph literal
  auto ng_operand = m_op_map.find(operand)->second;
  auto ng_op = std::make_shared<ngraph::op::Reverse>(ng_operand, ng_axis_set);
  m_op_map[reverse] = ng_op;
  m_instruction_list.push_back({reverse->ToString(), ng_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::GetConvolutionImagebatchShuffleVector()
// Given a convolution operation \p convolution, determine what reordering, if
// any, must be applied
// to its image-batch tensor operand in order to comply with the axis-order
// requirements of nGraph's
// Convolution operator.
// If no reordering is required, the returned vector will have the value [0, 1,
// ..., (rank-1)].
//-----------------------------------------------------------------------------
StatusOr<ngraph::AxisVector>
NGraphEmitter::GetConvolutionImagebatchShuffleVector(
    HloInstruction* convolution) {
  // nGraph's convolution op requires the image-batches tensor columns to be in
  // order
  // (N, C_in, d_1, ..., d_m).

  const xla::Shape& input_xla_shape = convolution->shape();
  const size_t rank = input_xla_shape.dimensions_size();
  const xla::ConvolutionDimensionNumbers& cdn =
      convolution->convolution_dimension_numbers();

  ngraph::AxisVector return_vect(rank);

  // TODO(amprocte): input_?
  return_vect[0] = cdn.input_batch_dimension();
  return_vect[1] = cdn.input_feature_dimension();
  for (int i = 0; i < cdn.input_spatial_dimensions_size(); ++i) {
    return_vect[2 + i] = cdn.input_spatial_dimensions()[i];
  }
  TF_RET_CHECK(IsPermutationOfZeroBasedVector(return_vect));

  return return_vect;
}

//-----------------------------------------------------------------------------
// NGraphEmitter::GetConvolutionFiltersShuffleVector()
// Given a convolution operation \p convolution, determine what reordering, if
// any, must be applied
// to its filter tensor operand in order to comply with the axis-order
// requirements of nGraph's
// Convolution operator.
// If no reordering is required, the returned vector will have the value [0, 1,
// ..., (rank-1)].
//-----------------------------------------------------------------------------
StatusOr<ngraph::AxisVector> NGraphEmitter::GetConvolutionFiltersShuffleVector(
    HloInstruction* convolution) {
  // nGraph's convolution op requires the filters tensor columns to be in order
  // (C_out, C_in, d_1, ..., d_m).

  const xla::Shape& input_xla_shape = convolution->shape();
  const size_t rank = input_xla_shape.dimensions_size();
  const xla::ConvolutionDimensionNumbers& cdn =
      convolution->convolution_dimension_numbers();

  ngraph::AxisVector return_vect(rank);

  return_vect[0] = cdn.kernel_output_feature_dimension();
  return_vect[1] = cdn.kernel_input_feature_dimension();
  for (int i = 0; i < cdn.kernel_spatial_dimensions_size(); ++i) {
    return_vect[2 + i] = cdn.kernel_spatial_dimensions()[i];
  }
  TF_RET_CHECK(IsPermutationOfZeroBasedVector(return_vect));

  return return_vect;
}

//-----------------------------------------------------------------------------
// NGraphEmitter::GetConvolutionOutputShuffleVector()
// Given a convolution operation \p convolution, determine what reordering, if
// any, must be applied
// to its result tensor, to shuffle its indices from the order promised by
// nGraph's Convolution
// operator to the order promised by HLO's Convolution operator.
// If no reordering is required, the returned vector will have the value [0, 1,
// ..., (rank-1)].
//-----------------------------------------------------------------------------
StatusOr<ngraph::AxisVector> NGraphEmitter::GetConvolutionOutputShuffleVector(
    HloInstruction* convolution) {
  // nGraph's Convolution op promises this output-tensor axis ordering:
  // (N, C_out, d_1, ..., d_m).
  //
  // HLO's Convolution op also promises this output-tensor axis ordering,
  // according to
  // documentation:
  // (N, C_out, d_1, ..., d_m).
  //
  // However, in practice that promise seem to not be kept.  Instead, we need to
  // query the
  // 'convolution_dimension_numbers' data structure to discover required the
  // axis-ordering of
  // the HLO Convolution operator's output-tensor.

  const xla::Shape& input_xla_shape = convolution->shape();
  const size_t rank = input_xla_shape.dimensions_size();
  const xla::ConvolutionDimensionNumbers& cdn =
      convolution->convolution_dimension_numbers();

  ngraph::AxisVector return_vect(rank);
  // TODO(amprocte): input_?
  return_vect[cdn.input_batch_dimension()] = 0;
  return_vect[cdn.input_feature_dimension()] = 1;
  for (int i = 0; i < cdn.input_spatial_dimensions_size(); ++i) {
    return_vect[cdn.input_spatial_dimensions()[i]] = i + 2;
  }

  TF_RET_CHECK(IsPermutationOfZeroBasedVector(return_vect));
  return return_vect;
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessConvolution
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessConvolution(HloInstruction* convolution,
                                         const HloInstruction* image_batch,
                                         const HloInstruction* filters,
                                         const Window& window) {
  // If necessary, add a reshape operation to make the image-batch tensor have
  // the required axis ordering.
  ngraph::AxisVector imgbatch_shuffle_vector;
  TF_ASSIGN_OR_RETURN(imgbatch_shuffle_vector,
                      GetConvolutionImagebatchShuffleVector(convolution));

  std::shared_ptr<ngraph::Node> properly_ordered_imgbatch_node;

  TF_CHECK_OK(MaybeAddAxesShuffle(m_op_map[image_batch],
                                  imgbatch_shuffle_vector,
                                  properly_ordered_imgbatch_node));

  // If necessary, add a reshape operation to make the filter-kernels tensor
  // have the required axis ordering.
  ngraph::AxisVector filters_shuffle_vector;
  TF_ASSIGN_OR_RETURN(filters_shuffle_vector,
                      GetConvolutionFiltersShuffleVector(convolution));

  std::shared_ptr<ngraph::Node> properly_ordered_filters_node;

  TF_CHECK_OK(MaybeAddAxesShuffle(m_op_map[filters], filters_shuffle_vector,
                                  properly_ordered_filters_node));

  ngraph::Strides window_movement_strides;
  ngraph::Strides window_dilation_factors;
  ngraph::Strides image_dilation_strides;
  ngraph::CoordinateDiff padding_below;
  ngraph::CoordinateDiff padding_above;

  for (int i = 0; i < window.dimensions_size(); ++i) {
    const ::xla::WindowDimension& wd = window.dimensions(i);
    window_movement_strides.push_back(wd.stride());
    window_dilation_factors.push_back(wd.window_dilation());
    image_dilation_strides.push_back(wd.base_dilation());
    padding_below.push_back(wd.padding_low());
    padding_above.push_back(wd.padding_high());
  }

  auto ng_op_conv = std::make_shared<ngraph::op::Convolution>(
      properly_ordered_imgbatch_node, properly_ordered_filters_node,
      window_movement_strides, window_dilation_factors, padding_below,
      padding_above, image_dilation_strides);

  // If necessary, add a reshape operation to make the Convoution output tensor
  // have the required axis ordering.
  ngraph::AxisVector convolution_output_shuffle_vector;
  TF_ASSIGN_OR_RETURN(convolution_output_shuffle_vector,
                      GetConvolutionOutputShuffleVector(convolution));

  std::shared_ptr<ngraph::Node> properly_ordered_convolution_output_node;

  TF_CHECK_OK(MaybeAddAxesShuffle(ng_op_conv, convolution_output_shuffle_vector,
                                  properly_ordered_convolution_output_node));

  m_op_map[convolution] = properly_ordered_convolution_output_node;
  m_instruction_list.push_back(
      {convolution->ToString(),
       properly_ordered_convolution_output_node->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter:ProcessSelect
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessSelect(HloInstruction* select,
                                    const HloInstruction* pred,
                                    const HloInstruction* on_true,
                                    const HloInstruction* on_false) {
  auto ng_pred_arg = m_op_map.find(pred)->second;
  auto ng_on_true_arg = m_op_map.find(on_true)->second;
  auto ng_on_false_arg = m_op_map.find(on_false)->second;

  auto ng_op_select = std::make_shared<ngraph::op::Select>(
      ng_pred_arg, ng_on_true_arg, ng_on_false_arg);

  m_op_map[select] = ng_op_select;
  m_instruction_list.push_back({select->ToString(), ng_op_select->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessDot
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessDot(HloInstruction* dot, const HloInstruction* lhs,
                                 const HloInstruction* rhs) {
  // Look up ngraph operands
  auto ng_lhs = m_op_map.find(lhs)->second;
  auto ng_rhs = m_op_map.find(rhs)->second;

  // Ensure that the parameters are of compatible shape
  if (!ShapeUtil::Compatible(lhs->shape(), lhs->shape())) {
    auto ng_shape_lhs = ngraph::Shape(lhs->shape().dimensions().begin(),
                                      lhs->shape().dimensions().end());

    auto ng_shape_rhs = ngraph::Shape(rhs->shape().dimensions().begin(),
                                      rhs->shape().dimensions().end());

    try {
      // TODO: CLEANUP More graceful exit without try/catch
      auto auto_broadcaster =
          AutoBroadcast(ng_lhs, ng_shape_lhs, ng_rhs, ng_shape_rhs);
      ng_lhs = auto_broadcaster.lhs();
      ng_rhs = auto_broadcaster.rhs();
    } catch (...) {
      NGRAPH_VLOG(1)
          << "Exception creating AutoBroadcast class: HLO instruction: "
          << dot->ToString();
      return InternalError("Exception creating AutoBroadcast");
    }
  }

  // Create ngraph node
  auto ng_op = std::make_shared<ngraph::op::Dot>(ng_lhs, ng_rhs);
  m_op_map[dot] = ng_op;
  m_instruction_list.push_back({dot->ToString(), ng_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessCompare()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessCompare(HloInstruction* compare, HloOpcode opcode,
                                     const HloInstruction* lhs,
                                     const HloInstruction* rhs) {
  const auto& operands = compare->operands();
  TF_RET_CHECK(operands.size() == 2);

  // Look up the operands
  auto ng_lhs = m_op_map.find(operands[0])->second;
  auto ng_rhs = m_op_map.find(operands[1])->second;
  auto ng_op = std::shared_ptr<ngraph::Node>();

  switch (opcode) {
    case HloOpcode::kEq:
      ng_op = std::make_shared<ngraph::op::Equal>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kGt:
      ng_op = std::make_shared<ngraph::op::Greater>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kLt:
      ng_op = std::make_shared<ngraph::op::Less>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kNe:
      ng_op = std::make_shared<ngraph::op::NotEqual>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kGe:
      ng_op = std::make_shared<ngraph::op::GreaterEq>(ng_lhs, ng_rhs);
      break;
    case HloOpcode::kLe:
      ng_op = std::make_shared<ngraph::op::LessEq>(ng_lhs, ng_rhs);
      break;
    default:
      return Unimplemented("HandleCompare OPCODE: '%s'",
                           HloOpcodeString(opcode).c_str());
  }

  // Store this in the map
  m_op_map[compare] = ng_op;
  m_instruction_list.push_back({compare->ToString(), ng_op->get_name()});
  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::MakeNGraphConstant()
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>> NGraphEmitter::MakeNGraphConstant(
    const xla::Shape& xla_shape, const Literal& literal) {
  auto ng_shape = ngraph::Shape(xla_shape.dimensions().begin(),
                                xla_shape.dimensions().end());

  std::shared_ptr<ngraph::Node> ng_node;
  switch (xla_shape.element_type()) {
    case F32: {
      auto float_vector = std::vector<float>(literal.data<float>().begin(),
                                             literal.data<float>().end());
      ng_node = ngraph::op::Constant::create(ngraph::element::f32, ng_shape,
                                             float_vector);
    } break;
    case S32: {
      auto int_vector = std::vector<int32>(literal.data<int32>().begin(),
                                           literal.data<int32>().end());
      ng_node = ngraph::op::Constant::create(ngraph::element::i32, ng_shape,
                                             int_vector);
    } break;
    case S64: {
      auto int_vector = std::vector<int64>(literal.data<int64>().begin(),
                                           literal.data<int64>().end());
      ng_node = ngraph::op::Constant::create(ngraph::element::i64, ng_shape,
                                             int_vector);
    } break;
    case PRED: {
      // Note: In ngraph a Bool is a traited type<char> - so we need to
      // get a std::vector<char> instead of std::vector<bool> in the line
      // below
      auto bool_vector = std::vector<char>(literal.data<bool>().begin(),
                                           literal.data<bool>().end());
      ng_node = ngraph::op::Constant::create(ngraph::element::boolean, ng_shape,
                                             bool_vector);
    } break;
    default:
      return Unimplemented(
          "MakeNGraphConstant: data type '%s'",
          PrimitiveType_Name(xla_shape.element_type()).c_str());
      break;
  }

  return ng_node;
}
//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessConstant()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessConstant(HloInstruction* constant,
                                      const Literal& literal) {
  const Shape& xla_shape = constant->shape();
  auto ng_shape = ngraph::Shape(xla_shape.dimensions().begin(),
                                xla_shape.dimensions().end());

  // Create the ngraph literal
  std::shared_ptr<ngraph::Node> ng_op;

  /*
    if (xla_shape.element_type() == TUPLE) {
      std::vector<std::shared_ptr<ngraph::Node>> tuple_vector;
      int num_elements = int(ShapeUtil::TupleElementCount(xla_shape));
      for (int i = 0; i < num_elements; ++i) {
        auto tuple_element_shape = ShapeUtil::GetTupleElementShape(xla_shape,
    i);
        std::shared_ptr<ngraph::Node> ng_tuple_constant;

        TF_ASSIGN_OR_RETURN(
            ng_tuple_constant,
            MakeNGraphConstant(tuple_element_shape,
    literal.data<Literal>().at(i)));

        tuple_vector.push_back(ng_tuple_constant);
      }
      ng_op = std::make_shared<ngraph::xla::op::Tuple>(tuple_vector);
    } else {
      TF_ASSIGN_OR_RETURN(ng_op, MakeNGraphConstant(xla_shape, literal));
    }
    */
  TF_ASSIGN_OR_RETURN(ng_op, MakeNGraphConstant(xla_shape, literal));

  // Store this in the map
  m_op_map[constant] = ng_op;
  m_instruction_list.push_back({constant->ToString(), ng_op->get_name()});
  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessGetTupleElement()
//
// A tuple is an array of pointers, one for each operand. Each pointer points
// to the output buffer of its corresponding operand. A GetTupleElement
// instruction forwards a pointer to the tuple element buffer at the given
// index.
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessGetTupleElement(HloInstruction* get_tuple_element,
                                             const HloInstruction* operand) {
  auto ng_op = m_tuple_op_map_vector[operand][get_tuple_element->tuple_index()];
  m_op_map[get_tuple_element] = ng_op;
  m_instruction_list.push_back(
      {get_tuple_element->ToString(), ng_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessParameter()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessParameter(HloInstruction* parameter) {
  auto ng_param_entry = m_op_map.find(parameter);
  TF_RET_CHECK(ng_param_entry != m_op_map.end());

  m_instruction_list.push_back(
      {parameter->ToString(), ng_param_entry->second->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessSlice()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessSlice(HloInstruction* slice,
                                   const HloInstruction* operand) {
  // Validate the input
  if (slice->slice_starts().size() != slice->slice_limits().size()) {
    return InvalidArgument(
        "HandleSlice: slice start and limit sizes differ: %zu vs %zu",
        slice->slice_starts().size(), slice->slice_limits().size());
  }

  if (slice->slice_starts().size() != slice->slice_strides().size()) {
    return InvalidArgument(
        "HandleSlice: slice start and strides sizes differ: %zu vs %zu",
        slice->slice_starts().size(), slice->slice_strides().size());
  }

  if (static_cast<int64>(slice->slice_starts().size()) !=
      ShapeUtil::Rank(operand->shape())) {
    return InvalidArgument(
        "HandleSlice: slice index count does not match argument rank: %zu vs "
        "%lld",
        slice->slice_starts().size(), ShapeUtil::Rank(operand->shape()));
  }

  if (ShapeUtil::IsScalar(slice->shape())) {
    return Unimplemented("HandleSlice: Instruction: '%s'",
                         slice->ToString().c_str());
  }

  // Create the vectors that contain the slice dimensions
  std::vector<size_t> lower_bounds(slice->slice_starts().begin(),
                                   slice->slice_starts().end());
  std::vector<size_t> upper_bounds(slice->slice_limits().begin(),
                                   slice->slice_limits().end());

  // Get the Slice input parameter that is saved in the op map already
  auto slice_input = m_op_map.find(operand)->second;

  // Createthe ngraph Slice operator
  auto ng_op = std::make_shared<ngraph::op::Slice>(slice_input, lower_bounds,
                                                   upper_bounds);

  // Update the map
  m_op_map[slice] = ng_op;
  m_instruction_list.push_back({slice->ToString(), ng_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessTuple()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessTuple(
    HloInstruction* tuple,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  std::vector<std::shared_ptr<ngraph::Node>> ng_ops_vector;
  for (auto operand : operands) {
    auto& ng_op = m_op_map.at(operand);
    ng_ops_vector.push_back(ng_op);
  }
  auto ng_op =
      std::make_shared<compat::op::Tuple>(ngraph::NodeVector{ng_ops_vector});

  m_op_map[tuple] = ng_op;
  m_tuple_op_map_vector[tuple] = ng_ops_vector;
  m_instruction_list.push_back({tuple->ToString(), "LIST-OF-TUPLES"});

  return Status::OK();
}

//-----------------------------------------------------------------------------
//  NGraphEmitter::NGraphFunction
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<compat::XLAFunction>> NGraphEmitter::NGraphFunction(
    const HloInstruction* root_instruction) {
  // Get the parameters from the graph
  std::vector<std::shared_ptr<ngraph::Node>> ng_parameter_list;

  // TODO: We are using std::map and assuming that the parameters are stored
  // in the order of their number. Probably better to revisit this and replace
  // with a suitable data structure.
  for (const auto& next_item : m_parameter_number_map) {
    auto& hlo_param = next_item.second;
    std::shared_ptr<ngraph::op::Parameter> ng_parameter =
        std::static_pointer_cast<ngraph::op::Parameter>(m_op_map.at(hlo_param));
    ng_parameter_list.push_back(ng_parameter);
  }

  // Create the ngraph XLAFunction
  const auto& ng_root_instruction = m_op_map.find(root_instruction)->second;

  std::vector<std::shared_ptr<ngraph::Node>> ng_roots_list{ng_root_instruction};

  std::shared_ptr<compat::XLAFunction> ng_function =
      std::make_shared<compat::XLAFunction>(ng_roots_list, ng_parameter_list,
                                            "");

  return ng_function;
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessReduce()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessReduce(HloInstruction* reduce,
                                    const HloInstruction* arg,
                                    const HloInstruction* init_value,
                                    const std::vector<int64>& dimensions,
                                    const HloComputation* function) {
  // Get the function instruction from the computation
  HloInstruction* function_instruction = function->root_instruction();

  // Embedded function - parse the sub-graph from the `function` instruction
  NGraphEmitter embedded_emitter(function->parameter_instructions(),
                                 m_fusion_map);
  TF_CHECK_OK(embedded_emitter.Initialize());

  TF_CHECK_OK(function_instruction->Accept(embedded_emitter.GetVisitor()));

  // Get the ngraph::function from the visited sub-graph
  std::shared_ptr<compat::XLAFunction> ng_function;
  TF_ASSIGN_OR_RETURN(ng_function,
                      embedded_emitter.NGraphFunction(function_instruction));

  // Get the tensor parameter that will be reduced
  auto ng_input_param = m_op_map.at(arg);

  // Get the tensor parameter that represents the init_value
  auto ng_init_value_param = m_op_map.at(init_value);

  // Create the AxisSet from the dimenstions array
  ngraph::AxisSet ng_axis_set;
  for (auto dim : dimensions) {
    ng_axis_set.insert(dim);
  }

  // Get the Reduce operator
  auto ng_reduce_op = std::make_shared<ngraph::op::Reduce>(
      ng_input_param, ng_init_value_param, ng_function, ng_axis_set);

  // Save ng_op in op_map
  m_op_map[reduce] = ng_reduce_op;
  m_instruction_list.push_back({reduce->ToString(), ng_reduce_op->get_name()});

  return tensorflow::Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessReduceWindow()
// \param arg_reductee The tensor view to be reduced.
// \param arg_init The initial value for reduction.
// \param reduction_function The reduction function to use.
// \param window_shape The window shape.
// \param window_movement_strides The window movement strides.
// ReduceWindow(const std::shared_ptr<Node>& arg_reductee,
//              const std::shared_ptr<Node>& arg_init,
//              const std::shared_ptr<Function>& reduction_function,
//              const Shape& window_shape,
//              const Strides& window_movement_strides);
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessReduceWindow(HloInstruction* reduce_window,
                                          const HloInstruction* operand,
                                          const Window& window,
                                          const HloComputation* function) {
  // TODO: Implement dilation for reduce-window.
  if (window_util::HasDilation(window)) {
    return Unimplemented(
        "Dilation for reduce-window not implemented on nGraph.");
  }

  // TODO: Implement padding for reduce-window
  if (window_util::HasPadding(window)) {
    return Unimplemented(
        "HasPadding for reduce-window not implemented on nGraph.");
  }

  // Get the function instruction from the computation
  HloInstruction* function_instruction = function->root_instruction();

  // Embedded function - parse the sub-graph from the `function` instruction
  NGraphEmitter embedded_emitter(function->parameter_instructions(),
                                 m_fusion_map);
  TF_CHECK_OK(embedded_emitter.Initialize());
  TF_CHECK_OK(function_instruction->Accept(embedded_emitter.GetVisitor()));

  // Get the ngraph::function from the visited sub-graph
  std::shared_ptr<compat::XLAFunction> ng_reduction_function;
  TF_ASSIGN_OR_RETURN(ng_reduction_function,
                      embedded_emitter.NGraphFunction(function_instruction));

  // Get ngraph operand
  auto ng_arg_reductee = m_op_map.at(operand);

  // Get operands, the first operand should be the same as operand passed in
  TF_RET_CHECK(reduce_window->operand(0) == operand);
  // The second operand is the initial value, should be in map already
  auto ng_arg_init = m_op_map.at(reduce_window->operand(1));

  // Get window size and strides
  std::vector<int64> window_size;
  std::vector<int64> window_stride;
  for (const auto& dim : window.dimensions()) {
    window_size.push_back(dim.size());
    window_stride.push_back(dim.stride());
  }
  auto ng_window_shape = ngraph::Shape(window_size.begin(), window_size.end());
  auto ng_window_movement_strides =
      ngraph::Strides(window_stride.begin(), window_stride.end());

  auto ng_op = std::make_shared<ngraph::op::ReduceWindow>(
      ng_arg_reductee, ng_arg_init, ng_reduction_function, ng_window_shape,
      ng_window_movement_strides);

  // Save ng_op to op_map
  m_op_map[reduce_window] = ng_op;
  m_instruction_list.push_back({reduce_window->ToString(), ng_op->get_name()});

  return tensorflow::Status::OK();
}

//-----------------------------------------------------------------------------
// MGraphEmitter::ProcessSelectAndScatter()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessSelectAndScatter(
    HloInstruction* select_and_scatter) {
  // TODO: Implement padding for select-and-scatter
  const Window& window = select_and_scatter->window();
  if (window_util::HasPadding(window)) {
    return Unimplemented(
        "HasPadding for Select-And-Scatter not implemented on nGraph.");
  }

  // Get operands for operand, source, and initial value
  auto operands = select_and_scatter->operands();
  TF_RET_CHECK(operands.size() == 3);

  HloInstruction* operand = operands[0];
  auto ng_operand = m_op_map.find(operand)->second;

  HloInstruction* source = operands[1];
  auto ng_source = m_op_map.find(source)->second;

  HloInstruction* init_value = operands[2];
  auto ng_init_value = m_op_map.find(init_value)->second;

  // Convert the TF select function to nGraph representation
  HloComputation* select_function = select_and_scatter->select();
  // Get the function instruction from the computation
  HloInstruction* select_function_instruction =
      select_function->root_instruction();

  // Embedded function - parse the sub-graph from the `function` instruction
  NGraphEmitter select_embedded_emitter(
      select_function->parameter_instructions(), m_fusion_map);
  TF_CHECK_OK(select_embedded_emitter.Initialize());
  TF_CHECK_OK(select_function_instruction->Accept(
      select_embedded_emitter.GetVisitor()));

  // Get the XLAFunction from the visited sub-graph
  std::shared_ptr<compat::XLAFunction> ng_select_function;
  TF_ASSIGN_OR_RETURN(
      ng_select_function,
      select_embedded_emitter.NGraphFunction(select_function_instruction));

  // Process the scatter function to an nGraph function like the select function
  HloComputation* scatter_function = select_and_scatter->scatter();
  HloInstruction* scatter_function_instruction =
      scatter_function->root_instruction();
  NGraphEmitter scatter_embedded_emitter(
      scatter_function->parameter_instructions(), m_fusion_map);
  TF_CHECK_OK(scatter_embedded_emitter.Initialize());
  TF_CHECK_OK(scatter_function_instruction->Accept(
      scatter_embedded_emitter.GetVisitor()));
  std::shared_ptr<compat::XLAFunction> ng_scatter_function;
  TF_ASSIGN_OR_RETURN(
      ng_scatter_function,
      scatter_embedded_emitter.NGraphFunction(scatter_function_instruction));

  // Grab the window dimensions and stride
  std::vector<int64> window_size;
  std::vector<int64> window_stride;
  for (const auto& dim : window.dimensions()) {
    window_size.push_back(dim.size());
    window_stride.push_back(dim.stride());
  }
  auto ng_window_shape = ngraph::Shape(window_size.begin(), window_size.end());
  auto ng_window_strides =
      ngraph::Strides(window_stride.begin(), window_stride.end());

  // Put it all together as an ngraph op
  auto ng_op = std::make_shared<ngraph::op::SelectAndScatter>(
      ng_operand, ng_source, ng_init_value, ng_select_function,
      ng_scatter_function, ng_window_shape, ng_window_strides);

  // Save ng_op to op_map
  m_op_map[select_and_scatter] = ng_op;
  m_instruction_list.push_back(
      {select_and_scatter->ToString(), ng_op->get_name()});

  return tensorflow::Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessBroadcast()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessBroadcast(HloInstruction* broadcast) {
  auto ng_op = std::shared_ptr<ngraph::Node>(nullptr);
  auto operands = broadcast->operands();
  TF_RET_CHECK(operands.size() == 1);

  HloInstruction* operand = operands[0];
  auto ng_operand = m_op_map.find(operand)->second;

  TF_ASSIGN_OR_RETURN(
      ng_op, MakeNGBroadcastOp(operand, ng_operand, broadcast->dimensions(),
                               broadcast->shape()));

  // Save ng_op in op_map
  m_op_map[broadcast] = ng_op;
  m_instruction_list.push_back({broadcast->ToString(), ng_op->get_name()});

  return tensorflow::Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::MakeNGBroadcastOP()
// In HloInstruction, for each axis in the original axis there is a
// corresponding axis in the broadcasted tensor (see example below).
// HloInstruction::dimensions() gives us such mapping. As defined in XLA
// specs, HloInstruction::dimensions() must be strictly increasing integers.
// Please see this link for more info:
// https://www.tensorflow.org/performance/xla/broadcasting#formal_definition.
//
// nGraph requires AxisSet, which denotes the set of new axes after broadcast.
// AxisSet is the set difference of {all axis in the resulting tensor} and
// {axis specified in HloInstruction::dimensions()}.
//
// Example:
//   - operand shape: [x, y]
//   - broadcasted shape: [a, b, x, c, y, d]
//                        [0][1][2][3][4][5]
//   - broadcast->dimensions(): [2, 4], since 'x' maps to the index 2 and
//                              'y' maps to index 4 after broadcasting.
//                              This needs to be strictly increasing.
//   - ngraph's AxisSet: {0, 1, 3, 5}, which is the newly added axes
//
// Notes: at ComputationBuilder, XLA always add new dimensions to the front of
// a tensor for broadcast. "If broadcast_sizes has values {a0, ..., aN} and
// the operand shape has dimensions {b0, ..., bM} then the shape of the output
// has dimensions {a0, ..., aN, b0, ..., bM}":
// https://www.tensorflow.org/performance/xla/operation_semantics#broadcast.
// However, HloInstruction supports more general broadcasting. The
// documentation for ComputationBuilder is here just for reference.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::op::Broadcast>>
NGraphEmitter::MakeNGBroadcastOp(
    const HloInstruction* operand,
    const std::shared_ptr<ngraph::Node>& ng_operand,
    const std::vector<int64>& broadcast_dimensions,
    const Shape& broadcast_shape) const {
  // Number of dimensions after broadcast
  int64 xla_rank = ShapeUtil::Rank(broadcast_shape);

  const Shape& operand_shape = operand->shape();
  int64 operand_rank = ShapeUtil::Rank(operand_shape);

  // Sanity check: operand_rank must match vector length of
  // broadcast_dimensions
  if (operand_rank != int64(broadcast_dimensions.size())) {
    return InternalError(
        "Broadcast operand_rank(%d) != broadcast_dimensions.size()(%d)",
        int(operand_rank), int(broadcast_dimensions.size()));
  }
  // Sanity check: dimension_difference >= 0
  if (xla_rank - int64(broadcast_dimensions.size()) < 0) {
    return InternalError("dimension_difference %d < 0",
                         int(xla_rank - int64(broadcast_dimensions.size())));
  }
  // Sanity check: broadcasting dimension must be strictly increasing
  if (broadcast_dimensions.size() > 1) {
    for (size_t i = 0; i < broadcast_dimensions.size() - 1; ++i) {
      if (broadcast_dimensions[i] >= broadcast_dimensions[i + 1]) {
        return InternalError(
            "broadcast_dimensions must be strictly increase, expecting %d < "
            "%d",
            int(broadcast_dimensions[i]), int(broadcast_dimensions[i + 1]));
      }
    }
  }

  // ng_axis_set is simply the set difference of:
  // {0, 1, ..., xla_rank - 1} \ set(broadcast_dimensions)
  std::set<size_t> all_axis_set;
  for (size_t i = 0; i < size_t(xla_rank); ++i) {
    all_axis_set.insert(i);
  }
  std::set<size_t> broadcast_dimensions_set(broadcast_dimensions.begin(),
                                            broadcast_dimensions.end());
  ngraph::AxisSet ng_axis_set;
  std::set_difference(all_axis_set.begin(), all_axis_set.end(),
                      broadcast_dimensions.begin(), broadcast_dimensions.end(),
                      std::inserter(ng_axis_set, ng_axis_set.begin()));

  // Create ngraph op
  auto ng_shape = ngraph::Shape(broadcast_shape.dimensions().begin(),
                                broadcast_shape.dimensions().end());
  auto ng_op = std::make_shared<ngraph::op::Broadcast>(ng_operand, ng_shape,
                                                       ng_axis_set);

  NGRAPH_VLOG(2) << "broadcast_dimensions: "
                 << container2string(broadcast_dimensions.begin(),
                                     broadcast_dimensions.end())
                 << "; broadcast_shape: "
                 << ShapeUtil::HumanString(broadcast_shape)
                 << "; operand_shape: " << ShapeUtil::HumanString(operand_shape)
                 << "; ng_axis_set: "
                 << container2string(ng_axis_set.begin(), ng_axis_set.end());
  return ng_op;
}

//-----------------------------------------------------------------------------
//  NGraphEmitter::ProcessReshape()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessReshape(HloInstruction* reshape) {
  // Shape of the Reshape HloInstruction
  const Shape& xla_shape = reshape->shape();

  // Operand info
  auto operands = reshape->operands();

  TF_RET_CHECK(operands.size() == 1);
  HloInstruction* operand = operands[0];
  const Shape& operand_shape = operand->shape();

  // The dimensions are natural i.e., {0 ..Rank-1}
  // However, if the user requests a different ordering then we will get a
  // transpose followed by a reshape.
  std::vector<size_t> reshape_dimensions(operand_shape.dimensions().size());
  std::iota(reshape_dimensions.begin(), reshape_dimensions.end(), 0);

  // nGraph Shape of the resulting output
  auto ng_resulting_shape = ngraph::Shape(xla_shape.dimensions().begin(),
                                          xla_shape.dimensions().end());
  // nGraph object representing the opeand - that was passed to us earlier
  auto ng_arg = m_op_map.find(operand)->second;

  // Create the nGraph reshape object
  auto ng_reshape_op = std::make_shared<ngraph::op::Reshape>(
      ng_arg, reshape_dimensions, ng_resulting_shape);

  // Save ng_op in op_map
  m_op_map[reshape] = ng_reshape_op;
  m_instruction_list.push_back(
      {reshape->ToString(), ng_reshape_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessTranspose()
// Exmple of how this works
// Consider the Transpose of 0x2x3 to 2x3x0
// Transpose Operand: s32[0,2,3] Layout: {2,1,0}
// Transpose "dimensions" i.e., permutation: {1,2,0}
// Resulting Shape: s32[2,3,0] Layout: {1,0,2}
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessTranspose(HloInstruction* transpose) {
  const HloInstruction* operand = transpose->operand(0);
  const Shape& operand_shape = operand->shape();

  // Transpose permutation for nGraph AxisVector
  auto ng_input_order = ngraph::AxisVector(transpose->dimensions().begin(),
                                           transpose->dimensions().end());

  auto operand_dims = operand_shape.dimensions();

  // nGraph Shape of the resulting output
  auto ng_resulting_shape = ngraph::Shape{};
  for (size_t i = 0; i < ng_input_order.size(); i++) {
    ng_resulting_shape.push_back(operand_dims[ng_input_order[i]]);
  }

  // nGraph object representing the opeand - that was passed to us earlier
  // const auto ng_op_entry = m_op_map.find(operand);
  const auto& ng_arg = m_op_map.find(operand)->second;

  // Create the nGraph reshape object
  auto ng_transpose_op = std::make_shared<ngraph::op::Reshape>(
      ng_arg, ng_input_order, ng_resulting_shape);

  // Save ng_op in op_map
  m_op_map[transpose] = ng_transpose_op;
  m_instruction_list.push_back(
      {transpose->ToString(), ng_transpose_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessPad()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessPad(HloInstruction* pad) {
  const HloInstruction* operand_input = pad->operand(0);
  const HloInstruction* operand_pad_value = pad->operand(1);

  const auto& ng_arg_input = m_op_map.find(operand_input)->second;
  const auto& ng_arg_pad_value = m_op_map.find(operand_pad_value)->second;

  auto ng_padding_below = ngraph::Shape{};
  auto ng_padding_above = ngraph::Shape{};
  auto ng_padding_interior = ngraph::Shape{};

  for (auto& padding_dimension : pad->padding_config().dimensions()) {
    // Check for negative padding, which we do not support yet.
    // Note that XLA itself does not allow negative
    // interior padding, only edge padding.
    if (padding_dimension.edge_padding_low() < 0 ||
        padding_dimension.edge_padding_high() < 0) {
      return Unimplemented(
          "nGraph's pad op doesn't yet support negative padding: op=%s",
          pad->ToString().c_str());
    }

    ng_padding_below.push_back(padding_dimension.edge_padding_low());
    ng_padding_above.push_back(padding_dimension.edge_padding_high());
    ng_padding_interior.push_back(padding_dimension.interior_padding());
  }

  // Create the nGraph pad object
  auto ng_pad_op = std::make_shared<ngraph::op::Pad>(
      ng_arg_input, ng_arg_pad_value, ng_padding_below, ng_padding_above,
      ng_padding_interior);

  // Save ng_op in op_map
  m_op_map[pad] = ng_pad_op;
  m_instruction_list.push_back({pad->ToString(), ng_pad_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessBatchNormTraining()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessBatchNormTraining(HloInstruction* bnt) {
  const HloInstruction* operand_input = bnt->operand(0);
  const HloInstruction* operand_gamma = bnt->operand(1);
  const HloInstruction* operand_beta = bnt->operand(2);

  double epsilon = bnt->epsilon();

  std::shared_ptr<ngraph::Node> ng_input_op =
      m_op_map.find(operand_input)->second;
  std::shared_ptr<ngraph::Node> ng_gamma_op =
      m_op_map.find(operand_gamma)->second;
  std::shared_ptr<ngraph::Node> ng_beta_op =
      m_op_map.find(operand_beta)->second;

  // If the normed axis is not 1, we will need to move the normed axis into
  // that position.
  if (bnt->feature_index() != 1) {
    const ngraph::Shape& input_shape = ng_input_op->get_shape();
    ngraph::AxisVector axis_order;
    ngraph::Shape reshaped_shape;

    for (size_t i = 0; i < input_shape.size(); i++) {
      if (i != bnt->feature_index()) {
        axis_order.push_back(i);
        reshaped_shape.push_back(input_shape[i]);
      }
    }

    axis_order.insert(axis_order.begin() + 1, bnt->feature_index());
    reshaped_shape.insert(reshaped_shape.begin() + 1,
                          input_shape[bnt->feature_index()]);

    ng_input_op = std::make_shared<ngraph::op::Reshape>(ng_input_op, axis_order,
                                                        reshaped_shape);
  }

  std::shared_ptr<ngraph::Node> ng_result_op =
      std::make_shared<ngraph::op::BatchNorm>(epsilon, ng_gamma_op, ng_beta_op,
                                              ng_input_op);

  std::shared_ptr<ngraph::Node> ng_output_op =
      std::make_shared<ngraph::op::GetOutputElement>(ng_result_op, 0);
  std::shared_ptr<ngraph::Node> ng_mean_op =
      std::make_shared<ngraph::op::GetOutputElement>(ng_result_op, 1);
  std::shared_ptr<ngraph::Node> ng_variance_op =
      std::make_shared<ngraph::op::GetOutputElement>(ng_result_op, 2);

  // If the normed axis is not 1, we will need to undo the reshaping we did
  // above, on output 0 of the batch-norm op. We then have to glue the results
  // back together with xla::op::Tuple.
  if (bnt->feature_index() != 1) {
    const ngraph::Shape& output_shape = ng_result_op->get_output_shape(0);
    ngraph::AxisVector axis_order;
    ngraph::Shape reshaped_shape;

    for (size_t i = 0; i < output_shape.size(); i++) {
      if (i != 1) {
        axis_order.push_back(i);
        reshaped_shape.push_back(output_shape[i]);
      }
    }

    axis_order.insert(axis_order.begin() + bnt->feature_index(), 1);
    reshaped_shape.insert(reshaped_shape.begin() + bnt->feature_index(),
                          output_shape[1]);

    ng_output_op = std::make_shared<ngraph::op::Reshape>(
        ng_output_op, axis_order, reshaped_shape);

    ng_result_op = std::make_shared<compat::op::Tuple>(
        ngraph::NodeVector{ng_output_op, ng_mean_op, ng_variance_op});
  }

  // Because this is a multi-output (tuple-typed) op, we need to put it in the
  // tuple op map.
  m_tuple_op_map_vector[bnt] =
      ngraph::NodeVector{ng_output_op, ng_mean_op, ng_variance_op};
  m_instruction_list.push_back({bnt->ToString(), ng_result_op->get_name()});
  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessBatchNormInference()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessBatchNormInference(HloInstruction* bni) {
  const HloInstruction* operand_input = bni->operand(0);
  const HloInstruction* operand_gamma = bni->operand(1);
  const HloInstruction* operand_beta = bni->operand(2);
  const HloInstruction* operand_mean = bni->operand(3);
  const HloInstruction* operand_variance = bni->operand(4);

  double epsilon = bni->epsilon();

  std::shared_ptr<ngraph::Node> ng_input_op =
      m_op_map.find(operand_input)->second;
  std::shared_ptr<ngraph::Node> ng_gamma_op =
      m_op_map.find(operand_gamma)->second;
  std::shared_ptr<ngraph::Node> ng_beta_op =
      m_op_map.find(operand_beta)->second;
  std::shared_ptr<ngraph::Node> ng_mean_op =
      m_op_map.find(operand_mean)->second;
  std::shared_ptr<ngraph::Node> ng_variance_op =
      m_op_map.find(operand_variance)->second;

  // If the normed axis is not 1, we will need to move the normed axis into
  // that position.
  if (bni->feature_index() != 1) {
    const ngraph::Shape& input_shape = ng_input_op->get_shape();
    ngraph::AxisVector axis_order;
    ngraph::Shape reshaped_shape;

    for (size_t i = 0; i < input_shape.size(); i++) {
      if (i != bni->feature_index()) {
        axis_order.push_back(i);
        reshaped_shape.push_back(input_shape[i]);
      }
    }

    axis_order.insert(axis_order.begin() + 1, bni->feature_index());
    reshaped_shape.insert(reshaped_shape.begin() + 1,
                          input_shape[bni->feature_index()]);

    ng_input_op = std::make_shared<ngraph::op::Reshape>(ng_input_op, axis_order,
                                                        reshaped_shape);
  }

  std::shared_ptr<ngraph::Node> ng_result_op =
      std::make_shared<ngraph::op::BatchNorm>(epsilon, ng_gamma_op, ng_beta_op,
                                              ng_input_op, ng_mean_op,
                                              ng_variance_op);

  // If the normed axis is not 1, we will need to undo the reshaping we did
  // above.
  if (bni->feature_index() != 1) {
    const ngraph::Shape& output_shape = ng_result_op->get_output_shape(0);
    ngraph::AxisVector axis_order;
    ngraph::Shape reshaped_shape;

    for (size_t i = 0; i < output_shape.size(); i++) {
      if (i != 1) {
        axis_order.push_back(i);
        reshaped_shape.push_back(output_shape[i]);
      }
    }

    axis_order.insert(axis_order.begin() + bni->feature_index(), 1);
    reshaped_shape.insert(reshaped_shape.begin() + bni->feature_index(),
                          output_shape[1]);

    ng_result_op = std::make_shared<ngraph::op::Reshape>(
        ng_result_op, axis_order, reshaped_shape);
  }

  m_instruction_list.push_back({bni->ToString(), ng_result_op->get_name()});
  m_op_map[bni] = ng_result_op;

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessBatchNormGrad()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessBatchNormGrad(HloInstruction* bng) {
  const HloInstruction* operand_input = bng->operand(0);
  const HloInstruction* operand_gamma = bng->operand(1);
  const HloInstruction* operand_mean = bng->operand(2);
  const HloInstruction* operand_variance = bng->operand(3);
  const HloInstruction* operand_delta = bng->operand(4);

  double epsilon = bng->epsilon();

  std::shared_ptr<ngraph::Node> ng_input_op =
      m_op_map.find(operand_input)->second;
  std::shared_ptr<ngraph::Node> ng_gamma_op =
      m_op_map.find(operand_gamma)->second;
  std::shared_ptr<ngraph::Node> ng_mean_op =
      m_op_map.find(operand_mean)->second;
  std::shared_ptr<ngraph::Node> ng_variance_op =
      m_op_map.find(operand_variance)->second;
  std::shared_ptr<ngraph::Node> ng_delta_op =
      m_op_map.find(operand_delta)->second;

  // If the normed axis is not 1, we will need to move the normed axis into
  // that position, both for the input and for the output delta.
  if (bng->feature_index() != 1) {
    const ngraph::Shape& input_shape = ng_input_op->get_shape();
    ngraph::AxisVector axis_order;
    ngraph::Shape reshaped_shape;

    for (size_t i = 0; i < input_shape.size(); i++) {
      if (i != bng->feature_index()) {
        axis_order.push_back(i);
        reshaped_shape.push_back(input_shape[i]);
      }
    }

    axis_order.insert(axis_order.begin() + 1, bng->feature_index());
    reshaped_shape.insert(reshaped_shape.begin() + 1,
                          input_shape[bng->feature_index()]);

    ng_input_op = std::make_shared<ngraph::op::Reshape>(ng_input_op, axis_order,
                                                        reshaped_shape);
    ng_delta_op = std::make_shared<ngraph::op::Reshape>(ng_delta_op, axis_order,
                                                        reshaped_shape);
  }

  // TODO(amprocte): We are temporarily supplying a fake value for beta here
  // (all zero, same shape/et as gamma), because XLA does not give beta to us.
  // This should work because nGraph should not actually use beta. The nGraph
  // op may change to discard this parameter. Update this when nGraph does.
  std::shared_ptr<ngraph::Node> ng_zero_scalar_op =
      std::make_shared<ngraph::op::Constant>(ng_gamma_op->get_element_type(),
                                             ngraph::Shape{},
                                             std::vector<std::string>{"0"});
  ngraph::AxisSet ng_broadcast_axes;
  for (size_t i = 0; i < ng_gamma_op->get_shape().size(); i++) {
    ng_broadcast_axes.insert(i);
  }
  std::shared_ptr<ngraph::Node> ng_zero_tensor_op =
      std::make_shared<ngraph::op::Broadcast>(
          ng_zero_scalar_op, ng_gamma_op->get_shape(), ng_broadcast_axes);

  std::shared_ptr<ngraph::Node> ng_result_op =
      std::make_shared<ngraph::op::BatchNormBackprop>(
          epsilon, ng_gamma_op, /*beta=*/ng_zero_tensor_op, ng_input_op,
          ng_mean_op, ng_variance_op, ng_delta_op);

  std::shared_ptr<ngraph::Node> ng_input_delta_op =
      std::make_shared<ngraph::op::GetOutputElement>(ng_result_op, 0);
  std::shared_ptr<ngraph::Node> ng_gamma_delta_op =
      std::make_shared<ngraph::op::GetOutputElement>(ng_result_op, 1);
  std::shared_ptr<ngraph::Node> ng_beta_delta_op =
      std::make_shared<ngraph::op::GetOutputElement>(ng_result_op, 2);

  // If the normed axis is not 1, we will need to undo the reshaping we did
  // above, on output 0 of the batch-norm op. We then have to glue the results
  // back together with xla::op::Tuple.
  if (bng->feature_index() != 1) {
    const ngraph::Shape& output_shape = ng_result_op->get_output_shape(0);
    ngraph::AxisVector axis_order;
    ngraph::Shape reshaped_shape;

    for (size_t i = 0; i < output_shape.size(); i++) {
      if (i != 1) {
        axis_order.push_back(i);
        reshaped_shape.push_back(output_shape[i]);
      }
    }

    axis_order.insert(axis_order.begin() + bng->feature_index(), 1);
    reshaped_shape.insert(reshaped_shape.begin() + bng->feature_index(),
                          output_shape[1]);

    ng_input_delta_op = std::make_shared<ngraph::op::Reshape>(
        ng_input_delta_op, axis_order, reshaped_shape);

    ng_result_op = std::make_shared<compat::op::Tuple>(ngraph::NodeVector{
        ng_input_delta_op, ng_gamma_delta_op, ng_beta_delta_op});
  }

  // Because this is a multi-output (tuple-typed) op, we need to put it in the
  // tuple op map.
  m_tuple_op_map_vector[bng] = ngraph::NodeVector{
      ng_input_delta_op, ng_gamma_delta_op, ng_beta_delta_op};
  m_instruction_list.push_back({bng->ToString(), ng_result_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::ProcessFusion()
//-----------------------------------------------------------------------------
Status NGraphEmitter::ProcessFusion(HloInstruction* fusion) {
  if (fusion->fusion_kind() != HloInstruction::FusionKind::kCustom) {
    return Unimplemented(
        "nGraph emitter does not support fused ops other than kCustom");
  }

  auto fusion_emitter_it = m_fusion_map->find(fusion);

  if (fusion_emitter_it == m_fusion_map->end()) {
    return Unimplemented(
        "nGraph emitter does not support kCustom ops with no associated fusion"
        " emitter (this is likely an internal error in the nGraph bridge)");
  }

  auto& fusion_emitter = fusion_emitter_it->second;

  std::shared_ptr<ngraph::Node> ng_root_op;
  TF_ASSIGN_OR_RETURN(ng_root_op, fusion_emitter->Emit(fusion, m_op_map));

  m_op_map[fusion] = ng_root_op;
  m_instruction_list.push_back({fusion->ToString(), ng_root_op->get_name()});

  return Status::OK();
}

//-----------------------------------------------------------------------------
// NGraphEmitter::FusedReductionEmitter::Emit()
// Handler for fused binop reductions.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>>
NGraphEmitter::FusedReductionEmitter::Emit(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*,
                             std::shared_ptr<ngraph::Node>>& op_map) const {
  const HloInstruction* operand = instruction->operand(0);

  // Get the tensor parameter that will be reduced
  auto ng_operand = op_map.find(operand)->second;

  // Create the AxisSet from the dimensions array
  ngraph::AxisSet ng_axis_set;
  for (auto dim : m_dimensions) {
    ng_axis_set.insert(dim);
  }

  std::shared_ptr<ngraph::Node> result_node;

  switch (m_opcode) {
    case HloOpcode::kAdd:
      result_node = std::make_shared<ngraph::op::Sum>(ng_operand, ng_axis_set);
      break;
    case HloOpcode::kMultiply:
      result_node =
          std::make_shared<ngraph::op::Product>(ng_operand, ng_axis_set);
      break;
    case HloOpcode::kMaximum:
      result_node = std::make_shared<ngraph::op::Max>(ng_operand, ng_axis_set);
      break;
    default:
      return Unimplemented(
          "nGraph fused reduction requested for unsupported opcode %s",
          HloOpcodeString(m_opcode).c_str());
  }

  return result_node;
}

// Converts the shape of an NHWC batch to the corresponding NCHW shape.
static StatusOr<ngraph::Shape> NhwcToNchw(const ngraph::Shape& input_shape) {
  if (input_shape.size() < 3) {
    return InvalidArgument(
        "Argument for NhwcToNchw must have at least three dimensions");
  }

  // Subtract out the batch and channel dimensions to obtain number of spatial
  // dimensions.
  size_t num_spatial_dims = input_shape.size() - 2;

  // To map NHWC to NCHW, input order will be 0,n,1,2,3,...,n-1
  ngraph::Shape result_shape;

  result_shape.push_back(input_shape[0]);
  result_shape.push_back(input_shape[input_shape.size() - 1]);

  for (size_t i = 0; i < num_spatial_dims; i++) {
    result_shape.push_back(input_shape[i + 1]);
  }

  return result_shape;
}

// Builds a reshape op to convert an NHWC tensor into an NCHW tensor.
static StatusOr<std::shared_ptr<ngraph::Node>> NhwcToNchw(
    std::shared_ptr<ngraph::Node> node) {
  const ngraph::Shape& input_shape = node->get_shape();

  if (input_shape.size() < 3) {
    return InvalidArgument(
        "Argument for NhwcToNchw must have at least three dimensions");
  }

  // Subtract out the batch and channel dimensions to obtain number of spatial
  // dimensions.
  size_t num_spatial_dims = input_shape.size() - 2;

  // To map NHWC to NCHW, input order will be 0,n,1,2,3,...,n-1
  ngraph::AxisVector input_order;
  ngraph::Shape result_shape;

  input_order.push_back(0);
  input_order.push_back(input_shape.size() - 1);

  result_shape.push_back(input_shape[0]);
  result_shape.push_back(input_shape[input_shape.size() - 1]);

  for (size_t i = 0; i < num_spatial_dims; i++) {
    input_order.push_back(i + 1);
    result_shape.push_back(input_shape[i + 1]);
  }

  return StatusOr<std::shared_ptr<ngraph::Node>>(
      std::make_shared<ngraph::op::Reshape>(node, input_order, result_shape));
}

// Builds a reshape op to convert an NHWC tensor into an NCHW tensor.
static StatusOr<std::shared_ptr<ngraph::Node>> NchwToNhwc(
    std::shared_ptr<ngraph::Node> node) {
  const ngraph::Shape& input_shape = node->get_shape();

  if (input_shape.size() < 3) {
    return InvalidArgument(
        "Argument for NchwToNhwc must have at least three dimensions");
  }

  // Subtract out the batch and channel dimensions to obtain number of spatial
  // dimensions.
  size_t num_spatial_dims = input_shape.size() - 2;

  // To map NHWC to NCHW, input order will be 0,2,3,...,n-1,n,1
  ngraph::AxisVector input_order;
  ngraph::Shape result_shape;

  input_order.push_back(0);
  result_shape.push_back(input_shape[0]);

  for (size_t i = 0; i < num_spatial_dims; i++) {
    input_order.push_back(i + 2);
    result_shape.push_back(input_shape[i + 2]);
  }

  input_order.push_back(1);
  result_shape.push_back(input_shape[1]);

  return StatusOr<std::shared_ptr<ngraph::Node>>(
      std::make_shared<ngraph::op::Reshape>(node, input_order, result_shape));
}

// Builds a reshape op to convert an HWio to an oiHW tensor.
static StatusOr<std::shared_ptr<ngraph::Node>> HwioToOihw(
    std::shared_ptr<ngraph::Node> node) {
  const ngraph::Shape& input_shape = node->get_shape();

  if (input_shape.size() < 3) {
    return InvalidArgument(
        "Argument for HwioToOihw must have at least three dimensions");
  }

  // Subtract out the batch and channel dimensions to obtain number of spatial
  // dimensions.
  size_t num_spatial_dims = input_shape.size() - 2;

  // To map HWio to oiHW, input order will be n,n-1,0,1,2,3,...,n-2
  ngraph::AxisVector input_order;
  ngraph::Shape result_shape;

  input_order.push_back(input_shape.size() - 1);
  input_order.push_back(input_shape.size() - 2);

  result_shape.push_back(input_shape[input_shape.size() - 1]);
  result_shape.push_back(input_shape[input_shape.size() - 2]);

  for (size_t i = 0; i < num_spatial_dims; i++) {
    input_order.push_back(i);
    result_shape.push_back(input_shape[i]);
  }

  return StatusOr<std::shared_ptr<ngraph::Node>>(
      std::make_shared<ngraph::op::Reshape>(node, input_order, result_shape));
}

// Converts the shape of an HWio filter bank to the corresponding oiHW shape.
static StatusOr<ngraph::Shape> HwioToOihw(const ngraph::Shape& input_shape) {
  if (input_shape.size() < 3) {
    return InvalidArgument(
        "Argument for HwioToOihw must have at least three dimensions");
  }

  // Subtract out the batch and channel dimensions to obtain number of spatial
  // dimensions.
  size_t num_spatial_dims = input_shape.size() - 2;

  // To map HWio to oiHW, input order will be n,n-1,0,1,2,3,...,n-2
  ngraph::Shape result_shape;

  result_shape.push_back(input_shape[input_shape.size() - 1]);
  result_shape.push_back(input_shape[input_shape.size() - 2]);

  for (size_t i = 0; i < num_spatial_dims; i++) {
    result_shape.push_back(input_shape[i]);
  }

  return result_shape;
}

// Builds a reshape op to convert an oiHW to an HWio tensor.
static StatusOr<std::shared_ptr<ngraph::Node>> OihwToHwio(
    std::shared_ptr<ngraph::Node> node) {
  const ngraph::Shape& input_shape = node->get_shape();

  if (input_shape.size() < 3) {
    return InvalidArgument(
        "Argument for OihwToHwio must have at least three dimensions");
  }

  // Subtract out the batch and channel dimensions to obtain number of spatial
  // dimensions.
  size_t num_spatial_dims = input_shape.size() - 2;

  // To map oiHW to HWio, input order will be 2,3,...,n,1,0
  ngraph::AxisVector input_order;
  ngraph::Shape result_shape;

  for (size_t i = 0; i < num_spatial_dims; i++) {
    input_order.push_back(i + 2);
    result_shape.push_back(input_shape[i + 2]);
  }

  input_order.push_back(1);
  input_order.push_back(0);

  result_shape.push_back(input_shape[1]);
  result_shape.push_back(input_shape[0]);

  return StatusOr<std::shared_ptr<ngraph::Node>>(
      std::make_shared<ngraph::op::Reshape>(node, input_order, result_shape));
}

// Extract window shape (in nGraph format, not including batch and channel
// dims) from an XLA window.
static ngraph::Shape XlaWindowShapeToNGraph(const xla::Window& window,
                                            bool is_nchw) {
  ngraph::Shape result;

  size_t first_spatial_dim = (is_nchw ? 2 : 1);
  size_t num_spatial_dims = window.dimensions().size() - 2;

  for (size_t i = first_spatial_dim; i < num_spatial_dims + first_spatial_dim;
       i++) {
    result.push_back(window.dimensions(i).size());
  }

  return result;
}

// Extract window strides (in nGraph format, not including batch and channel
// dims) from an XLA window.
static ngraph::Strides XlaWindowStridesToNGraph(const xla::Window& window,
                                                bool is_nchw) {
  ngraph::Strides result;

  size_t first_spatial_dim = (is_nchw ? 2 : 1);
  size_t num_spatial_dims = window.dimensions().size() - 2;

  for (size_t i = first_spatial_dim; i < num_spatial_dims + first_spatial_dim;
       i++) {
    result.push_back(window.dimensions(i).stride());
  }

  return result;
}

// Extract padding below (in nGraph format, not including batch and channel
// dims) from an XLA window.
static ngraph::Shape XlaWindowPaddingLowToNGraph(const xla::Window& window,
                                                 bool is_nchw) {
  ngraph::Shape result;

  size_t first_spatial_dim = (is_nchw ? 2 : 1);
  size_t num_spatial_dims = window.dimensions().size() - 2;

  for (size_t i = first_spatial_dim; i < num_spatial_dims + first_spatial_dim;
       i++) {
    result.push_back(window.dimensions(i).padding_low());
  }

  return result;
}

// Extract padding above (in nGraph format, not including batch and channel
// dims) from an XLA window.
static ngraph::Shape XlaWindowPaddingHighToNGraph(const xla::Window& window,
                                                  bool is_nchw) {
  ngraph::Shape result;

  size_t first_spatial_dim = (is_nchw ? 2 : 1);
  size_t num_spatial_dims = window.dimensions().size() - 2;

  for (size_t i = first_spatial_dim; i < num_spatial_dims + first_spatial_dim;
       i++) {
    result.push_back(window.dimensions(i).padding_high());
  }

  return result;
}

//-----------------------------------------------------------------------------
// NGraphEmitter::FusedMaxPoolEmitter::Emit()
// Handler for fused max-pool.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>>
NGraphEmitter::FusedMaxPoolEmitter::Emit(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*,
                             std::shared_ptr<ngraph::Node>>& op_map) const {
  const HloInstruction* operand = instruction->operand(0);

  // Get the tensor parameter that will be pooled
  auto ng_operand = op_map.find(operand)->second;

  if (!m_is_nchw) {
    // Input to nGraph MaxPool op is expected to be NCHW, so we must reshape.
    TF_ASSIGN_OR_RETURN(ng_operand, NhwcToNchw(ng_operand));
  }

  // Extract the window parameters.
  ngraph::Shape window_shape = XlaWindowShapeToNGraph(m_window, m_is_nchw);
  ngraph::Strides window_movement_strides =
      XlaWindowStridesToNGraph(m_window, m_is_nchw);
  ngraph::Shape padding_below =
      XlaWindowPaddingLowToNGraph(m_window, m_is_nchw);
  ngraph::Shape padding_above =
      XlaWindowPaddingHighToNGraph(m_window, m_is_nchw);

  std::shared_ptr<ngraph::Node> ng_max_pool_op =
      std::make_shared<ngraph::op::MaxPool>(ng_operand, window_shape,
                                            window_movement_strides,
                                            padding_below, padding_above);

  if (!m_is_nchw) {
    // Our output is expected to be in NHWC, so we must reshape.
    TF_ASSIGN_OR_RETURN(ng_max_pool_op, NchwToNhwc(ng_max_pool_op));
  }

  return StatusOr<std::shared_ptr<ngraph::Node>>(ng_max_pool_op);
}

//-----------------------------------------------------------------------------
// NGraphEmitter::FusedMaxPoolBackpropEmitter::Emit()
// Handler for fused max-pool backprop.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>>
NGraphEmitter::FusedMaxPoolBackpropEmitter::Emit(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*,
                             std::shared_ptr<ngraph::Node>>& op_map) const {
  const HloInstruction* operand = instruction->operand(0);
  const HloInstruction* deltas = instruction->operand(1);

  // Get the tensor parameter that will be pooled
  auto ng_operand = op_map.find(operand)->second;
  auto ng_deltas = op_map.find(deltas)->second;

  if (!m_is_nchw) {
    // Inputs to nGraph MaxPool backprop op are expected to be NCHW, so we must
    // reshape.
    TF_ASSIGN_OR_RETURN(ng_operand, NhwcToNchw(ng_operand));
    TF_ASSIGN_OR_RETURN(ng_deltas, NhwcToNchw(ng_deltas));
  }

  // Extract the window parameters.
  ngraph::Shape window_shape = XlaWindowShapeToNGraph(m_window, m_is_nchw);
  ngraph::Strides window_movement_strides =
      XlaWindowStridesToNGraph(m_window, m_is_nchw);
  ngraph::Shape padding_below =
      XlaWindowPaddingLowToNGraph(m_window, m_is_nchw);
  ngraph::Shape padding_above =
      XlaWindowPaddingHighToNGraph(m_window, m_is_nchw);

  std::shared_ptr<ngraph::Node> ng_max_pool_backprop_op =
      std::make_shared<ngraph::op::MaxPoolBackprop>(
          ng_operand, ng_deltas, window_shape, window_movement_strides,
          padding_below, padding_above);

  if (!m_is_nchw) {
    // Our output is expected to be in NHWC, so we have to reshape it back.
    TF_ASSIGN_OR_RETURN(ng_max_pool_backprop_op,
                        NchwToNhwc(ng_max_pool_backprop_op));
  }

  return StatusOr<std::shared_ptr<ngraph::Node>>(ng_max_pool_backprop_op);
}

//-----------------------------------------------------------------------------
// NGraphEmitter::FusedAvgPoolEmitter::Emit()
// Handler for fused avg-pool.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>>
NGraphEmitter::FusedAvgPoolEmitter::Emit(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*,
                             std::shared_ptr<ngraph::Node>>& op_map) const {
  const HloInstruction* operand = instruction->operand(0);

  // Get the tensor parameter that will be pooled
  auto ng_operand = op_map.find(operand)->second;

  if (!m_is_nchw) {
    // Input to nGraph AvgPool op is expected to be NCHW, so we must reshape.
    TF_ASSIGN_OR_RETURN(ng_operand, NhwcToNchw(ng_operand));
  }

  // Extract the window parameters.
  ngraph::Shape window_shape = XlaWindowShapeToNGraph(m_window, m_is_nchw);
  ngraph::Strides window_movement_strides =
      XlaWindowStridesToNGraph(m_window, m_is_nchw);
  ngraph::Shape padding_below =
      XlaWindowPaddingLowToNGraph(m_window, m_is_nchw);
  ngraph::Shape padding_above =
      XlaWindowPaddingHighToNGraph(m_window, m_is_nchw);

  std::shared_ptr<ngraph::Node> ng_avg_pool_op =
      std::make_shared<ngraph::op::AvgPool>(
          ng_operand, window_shape, window_movement_strides, padding_below,
          padding_above, false);

  if (!m_is_nchw) {
    // Our output is expected to be in NHWC, so we have to reshape it back.
    TF_ASSIGN_OR_RETURN(ng_avg_pool_op, NchwToNhwc(ng_avg_pool_op));
  }

  return StatusOr<std::shared_ptr<ngraph::Node>>(ng_avg_pool_op);
}

//-----------------------------------------------------------------------------
// NGraphEmitter::FusedAvgPoolBackpropEmitter::Emit()
// Handler for fused avg-pool backprop.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>>
NGraphEmitter::FusedAvgPoolBackpropEmitter::Emit(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*,
                             std::shared_ptr<ngraph::Node>>& op_map) const {
  const HloInstruction* operand = instruction->operand(0);

  // Get the tensor parameter that will be pooled
  auto ng_operand = op_map.find(operand)->second;

  ngraph::Shape ng_forward_arg_shape =
      XLAShapeToNgraphShape(m_forward_arg_shape);
  ngraph::Shape forward_arg_shape_reshaped;

  if (!m_is_nchw) {
    // Input to nGraph AvgPool backprop op is expected to be NCHW, so we must
    // reshape.
    TF_ASSIGN_OR_RETURN(ng_operand, NhwcToNchw(ng_operand));
    TF_ASSIGN_OR_RETURN(forward_arg_shape_reshaped,
                        NhwcToNchw(ng_forward_arg_shape));
  } else {
    forward_arg_shape_reshaped = ng_forward_arg_shape;
  }

  // Extract the window parameters.
  ngraph::Shape window_shape = XlaWindowShapeToNGraph(m_window, m_is_nchw);
  ngraph::Strides window_movement_strides =
      XlaWindowStridesToNGraph(m_window, m_is_nchw);
  ngraph::Shape padding_below =
      XlaWindowPaddingLowToNGraph(m_window, m_is_nchw);
  ngraph::Shape padding_above =
      XlaWindowPaddingHighToNGraph(m_window, m_is_nchw);

  std::shared_ptr<ngraph::Node> ng_avg_pool_backprop_op =
      std::make_shared<ngraph::op::AvgPoolBackprop>(
          forward_arg_shape_reshaped, ng_operand, window_shape,
          window_movement_strides, padding_below, padding_above, false);

  if (!m_is_nchw) {
    // Our output is expected to be in NHWC, so we have to reshape it back.
    TF_ASSIGN_OR_RETURN(ng_avg_pool_backprop_op,
                        NchwToNhwc(ng_avg_pool_backprop_op));
  }

  return StatusOr<std::shared_ptr<ngraph::Node>>(ng_avg_pool_backprop_op);
}

//-----------------------------------------------------------------------------
// NGraphEmitter::FusedConvBackpropInputEmitter::Emit()
// Handler for fused convolution backprop to input.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>>
NGraphEmitter::FusedConvBackpropInputEmitter::Emit(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*,
                             std::shared_ptr<ngraph::Node>>& op_map) const {
  std::shared_ptr<ngraph::Node> ng_output_delta_node =
      op_map.find(instruction->operand(0))->second;
  std::shared_ptr<ngraph::Node> ng_filter_node =
      op_map.find(instruction->operand(1))->second;

  ngraph::Shape forward_input_shape_reshaped;

  // If the output_delta tensor is not NCHW then it must be NHWC. We must
  // reshape it to NCHW.
  if (!m_is_nchw) {
    // Input to nGraph AvgPool backprop op is expected to be NCHW, so we must
    // reshape.
    //
    // Also, change the "forward arg shape" stored during fusion to NCHW.
    TF_ASSIGN_OR_RETURN(ng_output_delta_node, NhwcToNchw(ng_output_delta_node));
    TF_ASSIGN_OR_RETURN(forward_input_shape_reshaped,
                        NhwcToNchw(m_forward_input_shape));
  } else {
    forward_input_shape_reshaped = m_forward_input_shape;
  }

  // We will always have to reshape the filter from HWCiCo to CoCiHW.
  TF_ASSIGN_OR_RETURN(ng_filter_node, HwioToOihw(ng_filter_node));

  // Subtract 2 because batch, channel dims don't count.
  size_t num_spatial_dims = ng_output_delta_node->get_shape().size() - 2;

  // Construct the convolution backprop op. Dilation is always 1, because
  // tf2xla does not support other cases. (We can generalize in the future
  // if that situation changes.)
  ngraph::Strides forward_window_dilation_strides(num_spatial_dims, 1);
  ngraph::Strides forward_data_dilation_strides(num_spatial_dims, 1);

  std::shared_ptr<ngraph::Node> ng_conv_op =
      std::make_shared<ngraph::op::ConvolutionBackpropData>(
          forward_input_shape_reshaped, ng_filter_node, ng_output_delta_node,
          m_forward_window_movement_strides, forward_window_dilation_strides,
          m_forward_padding_below, m_forward_padding_above,
          forward_data_dilation_strides);

  // If our is expected to be in NHWC, we have to reshape it back.
  if (!m_is_nchw) {
    TF_ASSIGN_OR_RETURN(ng_conv_op, NchwToNhwc(ng_conv_op));
  }

  return StatusOr<std::shared_ptr<ngraph::Node>>(ng_conv_op);
}

//-----------------------------------------------------------------------------
// NGraphEmitter::FusedConvBackpropFiltersEmitter::Emit()
// Handler for fused convolution backprop to filters.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>>
NGraphEmitter::FusedConvBackpropFiltersEmitter::Emit(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*,
                             std::shared_ptr<ngraph::Node>>& op_map) const {
  std::shared_ptr<ngraph::Node> ng_input_node =
      op_map.find(instruction->operand(0))->second;
  std::shared_ptr<ngraph::Node> ng_output_delta_node =
      op_map.find(instruction->operand(1))->second;

  // If the input and output_delta tensors are not NCHW then they must be NHWC.
  // We must reshape them to NCHW.
  if (!m_is_nchw) {
    TF_ASSIGN_OR_RETURN(ng_input_node, NhwcToNchw(ng_input_node));
    TF_ASSIGN_OR_RETURN(ng_output_delta_node, NhwcToNchw(ng_output_delta_node));
  }

  // Subtract 2 because batch, channel dims don't count.
  size_t num_spatial_dims = ng_output_delta_node->get_shape().size() - 2;

  // Construct the convolution backprop op. Dilation is always 1, because
  // tf2xla does not support other cases. (We can generalize in the future
  // if that situation changes.)
  ngraph::Strides forward_window_dilation_strides(num_spatial_dims, 1);
  ngraph::Strides forward_data_dilation_strides(num_spatial_dims, 1);

  // The TF convention for the filters is HWio. nGraph's convention is oiHW.
  TF_ASSIGN_OR_RETURN(ngraph::Shape forward_filters_shape_reshaped,
                      HwioToOihw(ngraph::Shape(m_forward_filters_shape.begin(),
                                               m_forward_filters_shape.end())));

  std::shared_ptr<ngraph::Node> ng_conv_op =
      std::make_shared<ngraph::op::ConvolutionBackpropFilters>(
          ng_input_node, forward_filters_shape_reshaped, ng_output_delta_node,
          m_forward_window_movement_strides, forward_window_dilation_strides,
          m_forward_padding_below, m_forward_padding_above,
          forward_data_dilation_strides);

  // nGraph gives us the output in oiHW format, so we must reshape it to the
  // HWio that TensorFlow expects.
  TF_ASSIGN_OR_RETURN(ng_conv_op, OihwToHwio(ng_conv_op));

  return StatusOr<std::shared_ptr<ngraph::Node>>(ng_conv_op);
}

//-----------------------------------------------------------------------------
// NGraphEmitter::FusedReluEmitter::Emit()
// Handler for fused ReLU fprop.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>> NGraphEmitter::FusedReluEmitter::Emit(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*,
                             std::shared_ptr<ngraph::Node>>& op_map) const {
  std::shared_ptr<ngraph::Node> ng_input_node =
      op_map.find(instruction->operand(0))->second;

  std::shared_ptr<ngraph::Node> ng_relu_op =
      std::make_shared<ngraph::op::Relu>(ng_input_node);

  return StatusOr<std::shared_ptr<ngraph::Node>>(ng_relu_op);
}

//-----------------------------------------------------------------------------
// NGraphEmitter::FusedReluBackpropEmitter::Emit()
// Handler for fused ReLU bprop.
//-----------------------------------------------------------------------------
StatusOr<std::shared_ptr<ngraph::Node>>
NGraphEmitter::FusedReluBackpropEmitter::Emit(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*,
                             std::shared_ptr<ngraph::Node>>& op_map) const {
  std::shared_ptr<ngraph::Node> ng_arg0_node =
      op_map.find(instruction->operand(0))->second;
  std::shared_ptr<ngraph::Node> ng_arg1_node =
      op_map.find(instruction->operand(1))->second;

  // Note that the order of arguments comes out reversed here, due to the way
  // XLA fusion works.
  std::shared_ptr<ngraph::Node> ng_relu_op =
      std::make_shared<ngraph::op::ReluBackprop>(ng_arg1_node, ng_arg0_node);

  return StatusOr<std::shared_ptr<ngraph::Node>>(ng_relu_op);
}

}  // namespace ngraph_plugin
}  // namespace xla
