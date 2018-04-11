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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef NGRAPH_OP_HANDLER_H_
#define NGRAPH_OP_HANDLER_H_

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph_log.h"
#include "ngraph_utils.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"

// | HloOpcode           | Handler                                   |Supported|
// |---------------------|-------------------------------------------|---------|
// | kAbs                | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kAdd                | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kBatchNormGrad      | NGraphOpHandler::HandleBatchNormGrad      |YES
// | kBatchNormInference | NGraphOpHandler::HandleBatchNormInference |YES
// | kBatchNormTraining  | NGraphOpHandler::HandleBatchNormTraining  |YES
// | kBitcast            | NGraphOpHandler::HandleBitcast            |
// | kBroadcast          | NGraphOpHandler::HandleBroadcast          |YES
// | kCall               | NGraphOpHandler::HandleCall               |
// | kCeil               | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kClamp              | NGraphOpHandler::HandleClamp              |
// | kConcatenate        | NGraphOpHandler::HandleConcatenate        |
// | kConstant           | NGraphOpHandler::HandleConstant           |YES
// | kConvert            | NGraphOpHandler::HandleConvert            |YES
// | kConvolution        | NGraphOpHandler::HandleConvolution        |YES
// | kCopy               | NGraphOpHandler::HandleCopy               |
// | kCos                | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kCrossReplicaSum    | NGraphOpHandler::HandleCrossReplicaSum    |
// | kCustomCall         | NGraphOpHandler::HandleCustomCall         |
// | kDivide             | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kDot                | NGraphOpHandler::HandleDot                |YES
// | kDynamicSlice       | NGraphOpHandler::HandleDynamicSlice       |
// | kDynamicUpdateSlice | NGraphOpHandler::HandleDynamicUpdateSlice |
// | kEq                 | NGraphOpHandler::HandleCompare            |YES
// | kExp                | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kFloor              | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kFusion             | NGraphOpHandler::HandleFusion             |PARTIAL*
// | kGe                 | NGraphOpHandler::HandleCompare            |YES
// | kGetTupleElement    | NGraphOpHandler::HandleGetTupleElement    |YES
// | kGt                 | NGraphOpHandler::HandleCompare            |YES
// | kIndex              | Not handled by HloInstruction::Visit      |
// | kInfeed             | NGraphOpHandler::HandleInfeed             |
// | kIsFinite           | NGraphOpHandler::HandleElementwiseUnary   |
// | kLe                 | NGraphOpHandler::HandleCompare            |YES
// | kLog                | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kLogicalAnd         | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kLogicalNot         | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kLogicalOr          | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kLt                 | NGraphOpHandler::HandleCompare            |YES
// | kMap                | NGraphOpHandler::HandleMap                |
// | kMaximum            | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kMinimum            | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kMultiply           | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kNe                 | NGraphOpHandler::HandleCompare            |YES
// | kNegate             | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kOutfeed            | NGraphOpHandler::HandleOutfeed            |
// | kPad                | NGraphOpHandler::HandlePad                |YES
// | kParameter          | NGraphOpHandler::HandleParameter          |YES
// | kPower              | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kRecv               | NGraphOpHandler::HandleRecv               |
// | kReduce             | NGraphOpHandler::HandleReduce             |YES
// | kReducePrecision    | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kReduceWindow       | NGraphOpHandler::HandleReduceWindow       |YES
// | kRemainder          | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kReshape            | NGraphOpHandler::HandleReshape            |YES
// | kReverse            | NGraphOpHandler::HandleReverse            |YES
// | kRng                | NGraphOpHandler::HandleRng                |
// | kSelect             | NGraphOpHandler::HandleSelect             |YES
// | kSelectAndScatter   | NGraphOpHandler::HandleSelectAndScatter   |YES
// | kSend               | NGraphOpHandler::HandleSend               |
// | kSign               | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kSin                | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kSlice              | NGraphOpHandler::HandleSlice              |YES
// | kSort               | NGraphOpHandler::HandleSort               |
// | kSubtract           | NGraphOpHandler::HandleElementwiseBinary  |YES
// | kTanh               | NGraphOpHandler::HandleElementwiseUnary   |YES
// | kTrace              | Not handled by HloInstruction::Visit      |
// | kTranspose          | NGraphOpHandler::HandleTranspose          |YES
// | kTuple              | NGraphOpHandler::HandleTuple              |YES
// | kUpdate             | Not handled by HloInstruction::Visit      |
// | kWhile              | NGraphOpHandler::HandleWhile              |
//
// * kFusion only supported for kCustom fusions
namespace xla {
namespace ngraph_plugin {

using HloInstructionPtr = HloInstruction*;

// This class is a gateway to the XLA HLO operations supported by nGraph. This
// class overrides various HLO operation handlers that are called when a HLO DAG
// is visited post order by the base class. The operations supported by NGraph
// are handled by the virtual methods "ProcessXYZ(...)".
//
// The virtual methnods defined by the HloVisitor class (e.g.,
// HandleElementwiseUnary(...)) are marked final so that a derived class of
// NGraphOpHandler explicitely overrides the virtual methods "ProcessXYZ(...)
// defined by this class. This allows for generating a list of operations that
// are supported by nGraph for various types of DL models.
//
// When used standalone, this class provides a list of supported and unsupported
// HLO operations - which is useful inspecting individual HLO operations and
// their arguments.
class NGraphOpHandler : private DfsHloVisitorWithDefault {
 public:
  NGraphOpHandler() : m_ostream(std::cout) {}
  NGraphOpHandler(std::ostream& output_stream) : m_ostream(output_stream) {}
  ~NGraphOpHandler() {}
  xla::DfsHloVisitor* GetVisitor() { return this; }

  // The default action in this case just logs the instruction and retruns OK so
  // that the HLO graph traversal can continue. For an actual concrete class
  // that derives from this class, the default acion must return
  // Unimplemented(...) so that the graph traversal is stopped when an
  // unsupported operation is encountered.
  Status DefaultAction(HloInstruction* hlo) override {
    m_ostream << GetUnsupportedInstructionMsg(hlo);
    NGRAPH_VLOG(1) << GetUnsupportedInstructionMsg(hlo);
    return Status::OK();
  }

  Status FinishVisit(HloInstruction* hlo_root) override {
    return DfsHloVisitorWithDefault::FinishVisit(hlo_root);
  }

  Status HandleElementwiseUnary(HloInstructionPtr hlo) final {
    return ProcessElementwiseUnary(hlo, hlo->opcode());
  }

  Status HandleElementwiseBinary(HloInstructionPtr hlo) final {
    return ProcessElementwiseBinary(hlo, hlo->opcode());
  }

  Status HandleBatchNormTraining(HloInstructionPtr hlo) final {
    return ProcessBatchNormTraining(hlo);
  }

  Status HandleBatchNormInference(HloInstructionPtr hlo) final {
    return ProcessBatchNormInference(hlo);
  }

  Status HandleBatchNormGrad(HloInstructionPtr hlo) final {
    return ProcessBatchNormGrad(hlo);
  }

  Status HandleClamp(HloInstructionPtr clamp) final {
    return DefaultAction(clamp);
  }
  Status HandleConcatenate(HloInstructionPtr concatenate) final {
    return ProcessConcatenate(concatenate, concatenate->operands());
  }
  Status HandleConvert(HloInstructionPtr convert) final {
    return ProcessConvert(convert);
  }

  Status HandleCopy(HloInstructionPtr copy) final {
    return DefaultAction(copy);
  }

  Status HandleSelect(HloInstructionPtr select) final {
    return ProcessSelect(select, select->operand(0), select->operand(1),
                         select->operand(2));
  }
  Status HandleDot(HloInstructionPtr dot) final {
    return ProcessDot(dot, dot->operand(0), dot->operand(1));
  }
  Status HandleConvolution(HloInstructionPtr convolution) final {
    return ProcessConvolution(convolution, convolution->operand(0),
                              convolution->operand(1), convolution->window());
  }
  Status HandleCrossReplicaSum(HloInstructionPtr crs) final {
    return DefaultAction(crs);
  }
  Status HandleCompare(HloInstructionPtr compare) final {
    return ProcessCompare(compare, compare->opcode(), compare->operand(0),
                          compare->operand(1));
  }
  Status HandleRng(HloInstructionPtr random) final {
    return DefaultAction(random);
  }
  Status HandleInfeed(HloInstructionPtr infeed) final {
    return DefaultAction(infeed);
  }
  Status HandleOutfeed(HloInstructionPtr outfeed) final {
    return DefaultAction(outfeed);
  }
  Status HandleReverse(HloInstructionPtr reverse) final {
    return ProcessReverse(reverse, reverse->operand(0));
  }
  Status HandleSort(HloInstructionPtr sort) final {
    return DefaultAction(sort);
  }
  Status HandleConstant(HloInstructionPtr constant) final {
    return ProcessConstant(constant, constant->literal());
  }
  Status HandleGetTupleElement(HloInstructionPtr get_tuple_element) final {
    return ProcessGetTupleElement(get_tuple_element,
                                  get_tuple_element->operand(0));
  }
  Status HandleParameter(HloInstructionPtr parameter) final {
    return ProcessParameter(parameter);
  }
  Status HandleFusion(HloInstructionPtr fusion) final {
    if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom)
      return ProcessFusion(fusion);
    else
      return DefaultAction(fusion);
  }
  Status HandleCall(HloInstructionPtr call) final {
    return DefaultAction(call);
  }
  Status HandleCustomCall(HloInstructionPtr custom_call) final {
    return DefaultAction(custom_call);
  }
  Status HandleSlice(HloInstructionPtr slice) final {
    return ProcessSlice(slice, slice->operand(0));
  }
  Status HandleDynamicSlice(HloInstructionPtr dynamic_slice) final {
    return DefaultAction(dynamic_slice);
  }
  Status HandleDynamicUpdateSlice(
      HloInstructionPtr dynamic_update_slice) final {
    return DefaultAction(dynamic_update_slice);
  }
  Status HandleTuple(HloInstructionPtr tuple) final {
    return ProcessTuple(tuple, tuple->operands());
  }
  Status HandleMap(HloInstructionPtr map) final { return DefaultAction(map); }
  Status HandleReduce(HloInstructionPtr reduce) final {
    return ProcessReduce(reduce, reduce->operand(0), reduce->operand(1),
                         reduce->dimensions(),
                         reduce->called_computations().back());
  }
  Status HandleReduceWindow(HloInstructionPtr reduce_window) final {
    return ProcessReduceWindow(reduce_window, reduce_window->operand(0),
                               reduce_window->window(),
                               reduce_window->called_computations().back());
  }
  Status HandleSelectAndScatter(HloInstructionPtr select_and_scatter) final {
    return ProcessSelectAndScatter(select_and_scatter);
  }
  Status HandleBitcast(HloInstructionPtr bitcast) final {
    return DefaultAction(bitcast);
  }
  Status HandleBroadcast(HloInstructionPtr broadcast) final {
    return ProcessBroadcast(broadcast);
  }
  Status HandlePad(HloInstructionPtr pad) final { return ProcessPad(pad); }

  Status HandleReshape(HloInstructionPtr reshape) final {
    return ProcessReshape(reshape);
  }
  Status HandleTranspose(HloInstructionPtr transpose) final {
    return ProcessTranspose(transpose);
  }
  Status HandleWhile(HloInstructionPtr xla_while) final {
    return DefaultAction(xla_while);
  }
  Status HandleSend(HloInstructionPtr send) final {
    return DefaultAction(send);
  }
  Status HandleRecv(HloInstructionPtr recv) final {
    return DefaultAction(recv);
  }

 protected:
  // The following methods are meant for the derived class to implement.
  // We are providing the default implementatin here so that we can use this
  // class as is to determine which opetations are handled by the ngraph
  virtual Status ProcessElementwiseBinary(HloInstruction* hlo,
                                          HloOpcode opcode) {
    m_ostream << GetSupportedInstructionMsg(hlo);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(hlo);
    return Status::OK();
  }

  virtual Status ProcessBatchNormTraining(HloInstruction* hlo) {
    m_ostream << GetSupportedInstructionMsg(hlo);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(hlo);
    return Status::OK();
  }

  virtual Status ProcessBatchNormInference(HloInstruction* hlo) {
    m_ostream << GetSupportedInstructionMsg(hlo);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(hlo);
    return Status::OK();
  }

  virtual Status ProcessBatchNormGrad(HloInstruction* hlo) {
    m_ostream << GetSupportedInstructionMsg(hlo);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(hlo);
    return Status::OK();
  }

  virtual Status ProcessConcatenate(
      HloInstruction* concatenate,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
    m_ostream << GetSupportedInstructionMsg(concatenate);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(concatenate);
    return Status::OK();
  }

  virtual Status ProcessElementwiseUnary(HloInstruction* hlo,
                                         HloOpcode opcode) {
    m_ostream << GetSupportedInstructionMsg(hlo);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(hlo);
    return Status::OK();
  }

  virtual Status ProcessConvert(HloInstruction* hlo) {
    m_ostream << GetSupportedInstructionMsg(hlo);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(hlo);
    return Status::OK();
  }

  virtual Status ProcessSelect(HloInstruction* hlo,
                               const HloInstruction* /*pred*/,
                               const HloInstruction* /*on_true*/,
                               const HloInstruction* /*on_false*/) {
    m_ostream << GetSupportedInstructionMsg(hlo);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(hlo);
    return Status::OK();
  }

  virtual Status ProcessConvolution(HloInstruction* convolution,
                                    const HloInstruction* /*lhs*/,
                                    const HloInstruction* /* rhs */,
                                    const Window& /*window*/) {
    m_ostream << GetSupportedInstructionMsg(convolution);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(convolution);
    return Status::OK();
  }

  virtual Status ProcessDot(HloInstruction* dot, const HloInstruction* /*lhs*/,
                            const HloInstruction* /* rhs */) {
    m_ostream << GetSupportedInstructionMsg(dot);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(dot);
    return Status::OK();
  }

  virtual Status ProcessCompare(HloInstruction* compare, HloOpcode opcode,
                                const HloInstruction* lhs,
                                const HloInstruction* rhs) {
    m_ostream << GetSupportedInstructionMsg(compare);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(compare);
    return Status::OK();
  }

  virtual Status ProcessReverse(HloInstruction* reverse,
                                const HloInstruction* operand) {
    m_ostream << GetSupportedInstructionMsg(reverse);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(reverse);
    return Status::OK();
  }

  virtual Status ProcessConstant(HloInstruction* constant,
                                 const Literal& literal) {
    m_ostream << GetSupportedInstructionMsg(constant);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(constant);
    return Status::OK();
  }

  virtual Status ProcessGetTupleElement(HloInstruction* get_tuple_element,
                                        const HloInstruction* operand) {
    m_ostream << GetSupportedInstructionMsg(get_tuple_element);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(get_tuple_element);
    return Status::OK();
  }

  virtual Status ProcessParameter(HloInstruction* parameter) {
    m_ostream << GetSupportedInstructionMsg(parameter);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(parameter);
    return Status::OK();
  }

  virtual Status ProcessSlice(HloInstruction* slice,
                              const HloInstruction* operand) {
    m_ostream << GetSupportedInstructionMsg(slice);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(slice);
    return Status::OK();
  }

  virtual Status ProcessTuple(
      HloInstruction* tuple,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
    m_ostream << GetSupportedInstructionMsg(tuple);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(tuple);
    return Status::OK();
  }

  virtual Status ProcessReduce(HloInstruction* reduce,
                               const HloInstruction* arg,
                               const HloInstruction* init_value,
                               const std::vector<int64>& dimensions,
                               const HloComputation* function) {
    m_ostream << GetSupportedInstructionMsg(reduce);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(reduce);
    return Status::OK();
  }

  virtual Status ProcessReduceWindow(HloInstruction* reduce_window,
                                     const HloInstruction* operand,
                                     const Window& window,
                                     const HloComputation* function) {
    m_ostream << GetSupportedInstructionMsg(reduce_window);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(reduce_window);
    return Status::OK();
  }

  virtual Status ProcessSelectAndScatter(HloInstruction* select_and_scatter) {
    m_ostream << GetSupportedInstructionMsg(select_and_scatter);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(select_and_scatter);
    return Status::OK();
  }

  virtual Status ProcessBroadcast(HloInstruction* broadcast) {
    m_ostream << GetSupportedInstructionMsg(broadcast);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(broadcast);
    return Status::OK();
  }

  virtual Status ProcessReshape(HloInstruction* reshape) {
    m_ostream << GetSupportedInstructionMsg(reshape);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(reshape);
    return Status::OK();
  }

  virtual Status ProcessTranspose(HloInstruction* transpose) {
    m_ostream << GetSupportedInstructionMsg(transpose);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(transpose);
    return Status::OK();
  }

  virtual Status ProcessPad(HloInstruction* pad) {
    m_ostream << GetSupportedInstructionMsg(pad);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(pad);
    return Status::OK();
  }

  virtual Status ProcessFusion(HloInstruction* fusion) {
    m_ostream << GetSupportedInstructionMsg(fusion);
    NGRAPH_VLOG(1) << GetSupportedInstructionMsg(fusion);
    return Status::OK();
  }

 private:
  std::string GetSupportedInstructionMsg(HloInstruction* hlo) {
    std::ostringstream msg;
    msg << "Supported HLO Op: \"" << hlo->name() << "\" Details : \""
        << hlo->ToString() << "\"\n";
    return msg.str();
  }

  std::string GetUnsupportedInstructionMsg(HloInstruction* hlo) {
    std::ostringstream msg;
    msg << "Unsupported HLO Op: \"" << hlo->name() << "\" Details: \""
        << hlo->ToString() << "\"\n";
    return msg.str();
  }

  std::ostream& m_ostream;

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphOpHandler);
};

}  // namespace ngraph_plugin
}  // namespace xla

#endif  // NGRAPH_OP_HANDLER_H_
