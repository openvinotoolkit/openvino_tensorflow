/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef OPENVINO_TF_BRIDGE_DEFAULT_OPSET_H_
#define OPENVINO_TF_BRIDGE_DEFAULT_OPSET_H_
#pragma once

#include "openvino/opsets/opset7.hpp"

namespace tensorflow {
namespace openvino_tensorflow {

namespace opset = ov::opset7;
namespace default_opset = ov::opset7;

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_BRIDGE_DEFAULT_OPSET_H_
