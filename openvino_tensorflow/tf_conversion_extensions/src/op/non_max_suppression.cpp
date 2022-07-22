// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extensions.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;
namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_non_max_suppression_op(
    const ov::frontend::NodeContext& node) {
    auto boxes = node.get_input(0);
    auto scores = node.get_input(1);
    auto max_output_size = node.get_input(2);
    auto iou_threshold = node.get_input(3);

    auto axis = make_shared<Constant>(ov::element::i64, Shape{1}, 0);
    auto boxes_unsqueezed = make_shared<Unsqueeze>(boxes, axis);

    auto axis_scores = make_shared<Constant>(ov::element::i64, Shape{2}, vector<int64_t>{0, 1});
    auto scores_unsqueezed = make_shared<Unsqueeze>(scores, axis_scores);

    auto backend_name = node.get_attribute<std::string>("_ovtf_backend_name");

    std::shared_ptr<NonMaxSuppression> ng_nms;
    const auto& op_type = node.get_op_type();
    if (op_type == "NonMaxSuppressionV5") {
        auto score_threshold = node.get_input(4);
        auto soft_nms_sigma = node.get_input(5);
        // TODO: pad_to_max_output_size and then remove the corresponding constraint
        // check from OCM
        ng_nms = make_shared<NonMaxSuppression>(boxes_unsqueezed,
                                                scores_unsqueezed,
                                                max_output_size,
                                                iou_threshold,
                                                score_threshold,
                                                soft_nms_sigma,
                                                NonMaxSuppression::BoxEncodingType::CORNER,
                                                false,
                                                ov::element::Type_t::i32);
        //set_node_name(node.get_name(), res);
        //return res->outputs();
    } else if (op_type == "NonMaxSuppressionV4") {
        auto score_threshold = node.get_input(4);
        // TODO: pad_to_max_output_size and then remove the corresponding constraint
        // check from OCM
        ng_nms = make_shared<NonMaxSuppression>(boxes_unsqueezed,
                                                scores_unsqueezed,
                                                max_output_size,
                                                iou_threshold,
                                                score_threshold,
                                                NonMaxSuppression::BoxEncodingType::CORNER,
                                                false,
                                                ov::element::Type_t::i32);
        //set_node_name(node.get_name(), res);
        //return res->outputs();
    } else if (op_type == "NonMaxSuppressionV3") {
        auto score_threshold = node.get_input(4);
        ng_nms = make_shared<NonMaxSuppression>(boxes_unsqueezed,
                                                scores_unsqueezed,
                                                max_output_size,
                                                iou_threshold,
                                                score_threshold,
                                                NonMaxSuppression::BoxEncodingType::CORNER,
                                                false,
                                                ov::element::Type_t::i32);
        //set_node_name(node.get_name(), res);
        //return {res->output(0)};
    } else if (op_type == "NonMaxSuppressionV2" || op_type == "NonMaxSuppression") {
        ng_nms = make_shared<NonMaxSuppression>(boxes_unsqueezed,
                                                scores_unsqueezed,
                                                max_output_size,
                                                iou_threshold,
                                                NonMaxSuppression::BoxEncodingType::CORNER,
                                                false,
                                                ov::element::Type_t::i32);
        //set_node_name(node.get_name(), res);
        //return {res->output(0)};
    } else {
      //TENSORFLOW_OP_VALIDATION(node, false, "No translator found.");
    }

    OutputVector res;

    // selected_indices output from OV doesn't have same structure as of TF for
    // CPU device for all the NMS ops
    auto begin = make_shared<Constant>(ov::element::i64,
                                       ov::Shape{2},
                                       std::vector<int64_t>({0, 2}));
    //set_node_name(node.get_name(), begin);
    auto end = make_shared<Constant>(ov::element::i64,
                                     ov::Shape{2},
                                     std::vector<int64_t>({0, -1}));
    //set_node_name(node.get_name(), end);
    auto ng_nms_selected_indices = make_shared<StridedSlice>(
        ng_nms->outputs()[0], begin, end,
        std::vector<int64_t>{1, 0}, std::vector<int64_t>{1, 0},
        std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 1});
    //set_node_name(node.get_name(), ng_nms_selected_indices);

    //SaveNgOp(ng_op_map, op->name(), ng_nms_selected_indices);
    res.push_back(ng_nms_selected_indices->outputs()[0]);

    if (backend_name == "CPU") {
        // selected_scores and valid_outputs shape from OV is not in sync with
        // TF output and needs extra transformation
        if (op_type == "NonMaxSuppressionV5") {
          // selected_scores needs same transformation as selected_indices
          auto ng_nms_selected_scores = make_shared<StridedSlice>(
              ng_nms->outputs()[1], begin, end,
              std::vector<int64_t>{1, 0}, std::vector<int64_t>{1, 0},
              std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 1});
          //set_node_name(node.get_name(), ng_nms_selected_scores);
          // valid_outputs is 1D tensor in case of OV, and 0D tensor in case of TF
          auto ng_squeeze_axis = make_shared<Constant>(
              ov::element::i32, ov::Shape{}, 0);
          //set_node_name(node.get_name(), ng_squeeze_axis);
          auto valid_outputs = make_shared<Squeeze>(
              ng_nms->outputs()[2], ng_squeeze_axis);
          //set_node_name(node.get_name(), valid_outputs);
          //SaveNgOp(ng_op_map, op->name(), ng_nms_selected_scores);
          res.push_back(ng_nms_selected_scores->outputs()[0]);
          //SaveNgOp(ng_op_map, op->name(), valid_outputs);
          res.push_back(valid_outputs->outputs()[0]);
        } else if (op_type == "NonMaxSuppressionV4") {
          auto ng_squeeze_axis = make_shared<Constant>(
              ov::element::i32, ov::Shape{}, 0);
          //set_node_name(node.get_name(), ng_squeeze_axis);
          auto valid_outputs = make_shared<Squeeze>(
              ng_nms->outputs()[1], ng_squeeze_axis);
          //set_node_name(node.get_name(), valid_outputs);
          //SaveNgOp(ng_op_map, op->name(), valid_outputs);
          res.push_back(valid_outputs->outputs()[0]);
        }
    } else {
      // for GPU and MYRIAD the default output works properly
      // except for valid_outputs in NMSV5 and NMSV4
      //SaveNgOp(ng_op_map, op->name(), ng_nms->outputs()[0]);
      res.push_back(ng_nms->outputs()[0]);
      if (op_type == "NonMaxSuppressionV5") {
        // valid_outputs is 1D tensor in case of OV, and 0D tensor in case of TF
        auto ng_squeeze_axis = make_shared<Constant>(
            ov::element::i32, ov::Shape{}, 0);
        //set_node_name(node.get_name(), ng_squeeze_axis);
        auto valid_outputs = make_shared<Squeeze>(
            ng_nms->outputs()[2], ng_squeeze_axis);
        //set_node_name(node.get_name(), valid_outputs);
        //SaveNgOp(ng_op_map, op->name(), ng_nms->outputs()[1]);
        res.push_back(ng_nms->outputs()[0]);
        //SaveNgOp(ng_op_map, op->name(), valid_outputs);
        res.push_back(valid_outputs->outputs()[0]);
      } else if (op_type == "NonMaxSuppressionV4") {
        auto ng_squeeze_axis = make_shared<Constant>(
            ov::element::i32, ov::Shape{}, 0);
        //set_node_name(node.get_name(), ng_squeeze_axis);
        auto valid_outputs = make_shared<Squeeze>(
            ng_nms->outputs()[1], ng_squeeze_axis);
        //set_node_name(node.get_name(), valid_outputs);
        //SaveNgOp(ng_op_map, op->name(), valid_outputs);
        res.push_back(valid_outputs->outputs()[0]);
      }
    }

    return res;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
