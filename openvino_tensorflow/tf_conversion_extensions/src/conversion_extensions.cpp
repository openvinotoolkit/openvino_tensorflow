// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extensions.hpp"
#include <openvino/core/extension.hpp>
#include <openvino/frontend/tensorflow/extension/conversion.hpp>

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "_FusedConv2D",
        ov::frontend::tensorflow::op::translate_fused_conv_2d_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "_FusedMatMul",
        ov::frontend::tensorflow::op::translate_fused_mat_mul_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "_FusedBatchNormEx",
        ov::frontend::tensorflow::op::translate_fused_batch_norm_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "FusedBatchNorm",
        ov::frontend::tensorflow::op::translate_fused_batch_norm_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "FusedBatchNormV3",
        ov::frontend::tensorflow::op::translate_fused_batch_norm_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "_FusedDepthwiseConv2dNative",
        ov::frontend::tensorflow::op::translate_depthwise_conv_2d_native_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "_MklSwish", ov::frontend::tensorflow::op::translate_mkl_swish_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "Concat", ov::frontend::tensorflow::op::translate_concat_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "ConcatV2", ov::frontend::tensorflow::op::translate_concat_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "NonMaxSuppression",
        ov::frontend::tensorflow::op::translate_non_max_suppression_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "NonMaxSuppressionV2",
        ov::frontend::tensorflow::op::translate_non_max_suppression_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "NonMaxSuppressionV3",
        ov::frontend::tensorflow::op::translate_non_max_suppression_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "NonMaxSuppressionV4",
        ov::frontend::tensorflow::op::translate_non_max_suppression_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "NonMaxSuppressionV5",
        ov::frontend::tensorflow::op::translate_non_max_suppression_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "Slice", ov::frontend::tensorflow::op::translate_slice_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "Cast", ov::frontend::tensorflow::op::translate_cast_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "CTCGreedyDecoder",
        ov::frontend::tensorflow::op::translate_ctc_greedy_decoder_op),
    std::make_shared<ov::frontend::tensorflow::ConversionExtension>(
        "SparseToDense",
        ov::frontend::tensorflow::op::translate_sparse_to_dense_op),
}));
