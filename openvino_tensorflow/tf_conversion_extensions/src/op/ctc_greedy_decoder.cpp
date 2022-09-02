// Copyright (C) 2018-2021 int64_tel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extensions.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;
namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_ctc_greedy_decoder_op(
    const ov::frontend::NodeContext& node) {
  auto op_type = node.get_op_type();
  auto ng_inputs = node.get_input(0);
  auto ng_sequence_length = node.get_input(1);

  ov::Shape transpose_order{1, 0, 2};
  auto input_order = make_shared<Constant>(
      ov::element::u64, ov::Shape{transpose_order.size()}, transpose_order);
  ng_inputs = make_shared<opset8::Transpose>(ng_inputs, input_order);

  ov::Shape ng_inputs_shape = ng_inputs.get_shape();
  auto batch_size = static_cast<long>(ng_inputs_shape.at(0));
  auto max_time_steps = static_cast<long>(ng_inputs_shape.at(1));

  auto ng_indicator = make_shared<Range>(
      make_shared<Constant>(ov::element::i64, Shape{}, 0),
      make_shared<Constant>(ov::element::i64, ov::Shape{}, max_time_steps),
      make_shared<Constant>(ov::element::i64, ov::Shape{}, 1),
      ov::element::i64);

  auto ng_indicator_1 = make_shared<Unsqueeze>(
      ng_indicator, make_shared<Constant>(ov::element::i64, ov::Shape{1},
                                          std::vector<int64_t>({0})));

  auto ng_indicator_2 = make_shared<Tile>(
      ng_indicator_1,
      make_shared<Constant>(ov::element::i64, ov::Shape{2},
                            std::vector<int64_t>({batch_size, 1})));

  auto ng_sequence_endpoint64_ts = make_shared<Unsqueeze>(
      ng_sequence_length, make_shared<Constant>(ov::element::i64, ov::Shape{1},
                                                std::vector<int64_t>({1})));

  auto ng_sequence_endpoint64_ts_1 = make_shared<Tile>(
      ng_sequence_endpoint64_ts,
      make_shared<Constant>(ov::element::i64, Shape{2},
                            std::vector<int64_t>({1, max_time_steps})));

  auto ng_sequence_endpoint64_ts_2 =
      make_shared<Convert>(ng_sequence_endpoint64_ts_1, ov::element::i64);

  auto ng_valid_time_steps_mask =
      make_shared<Less>(ng_indicator, ng_sequence_endpoint64_ts_2);
  auto ng_valid_time_steps_mask_1 = make_shared<Convert>(
      ng_valid_time_steps_mask, ng_inputs.get_element_type());

  auto ng_max_axis = make_shared<Constant>(ov::element::i64, ov::Shape{}, 2);
  auto ng_sum_axis_1 = make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);

  auto ng_log_probs = make_shared<ReduceMax>(ng_inputs, ng_max_axis, 0);
  auto ng_log_probs_1 =
      make_shared<Multiply>(ng_log_probs, ng_valid_time_steps_mask_1);
  auto ng_log_probs_2 =
      make_shared<ReduceSum>(ng_log_probs_1, ng_sum_axis_1, 1);
  auto ng_log_probs_3 = make_shared<Multiply>(
      ng_log_probs_2,
      make_shared<Constant>(ng_inputs.get_element_type(), ov::Shape{}, -1));

  auto merge_repeated = node.get_attribute<bool>("merge_repeated");
  auto blank_index = node.get_attribute<int64_t>("blank_index");

  if (blank_index < 0) {
    int64_t num_classes = static_cast<int64_t>(ng_inputs_shape.at(2));
    blank_index = num_classes + blank_index;
  }

  auto ng_blank_index =
      make_shared<Constant>(ov::element::i64, ov::Shape{}, blank_index);

  auto ng_ctc_outputs = make_shared<CTCGreedyDecoderSeqLen>(
      ng_inputs, ng_sequence_length, ng_blank_index, merge_repeated,
      ov::element::i64);
  auto ng_ctc_decoded_classes = ng_ctc_outputs->output(0);

  auto ng_ignore_value =
      make_shared<Constant>(ov::element::i64, ov::Shape{}, -1);
  auto ng_decoded_mask =
      make_shared<NotEqual>(ng_ctc_decoded_classes, ng_ignore_value);

  auto ng_indices = make_shared<NonZero>(ng_decoded_mask);

  ov::Shape indices_transpose_order{1, 0};
  auto ng_indices_transpose_order = make_shared<Constant>(
      ov::element::u64, ov::Shape{2}, indices_transpose_order);
  auto ng_indices_1 =
      make_shared<opset8::Transpose>(ng_indices, ng_indices_transpose_order);
  auto ng_values = make_shared<GatherND>(ng_ctc_decoded_classes, ng_indices_1);

  // Compute the shape of the smallest dense tensor that can contain the sparse
  // matrix represented by ng_indices and ng_values.
  auto ng_batch_size = make_shared<Constant>(ov::element::i64, ov::Shape{1},
                                             std::vector<long>({batch_size}));
  auto ng_ctc_decoded_sequence_lens =
      make_shared<Convert>(ng_ctc_outputs->output(1), ov::element::i64);
  auto ng_decoded_max_time_steps = make_shared<ReduceMax>(
      ng_ctc_decoded_sequence_lens,
      make_shared<Constant>(ov::element::i64, ov::Shape{}, 0), 1);

  auto ng_decoded_shape = make_shared<Concat>(
      ov::OutputVector({ng_batch_size, ng_decoded_max_time_steps}), 0);

  OutputVector res;
  res.push_back(ng_indices_1);
  res.push_back(ng_values);
  res.push_back(ng_decoded_shape);
  res.push_back(ng_log_probs_3);

  return res;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
