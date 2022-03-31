/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "openvino_tensorflow/layout_conversions.h"

namespace tensorflow {
namespace openvino_tensorflow {

void NHWCtoNCHW(const string& op_name, bool is_nhwc,
                ov::Output<ov::Node>& node) {
  if (is_nhwc) {
    auto rank = node.get_shape().size();
    if (rank == 4) {
      Transpose<0, 3, 1, 2>(node);
    } else if (rank == 5) {
      Transpose3D<0, 4, 1, 2, 3>(node);
    }
    Builder::SetTracingInfo(op_name, node);
  }
}

void NCHWtoNHWC(const string& op_name, bool is_nhwc,
                ov::Output<ov::Node>& node) {
  if (is_nhwc) {
    auto rank = node.get_shape().size();
    if (rank == 4) {
      Transpose<0, 2, 3, 1>(node);
    } else if (rank == 5) {
      Transpose3D<0, 2, 3, 4, 1>(node);
    }
    Builder::SetTracingInfo(op_name, node);
  }
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
