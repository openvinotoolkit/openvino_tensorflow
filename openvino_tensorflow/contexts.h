/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef CONTEXTS_H
#define CONTEXTS_H

#include "openvino/openvino.hpp"

namespace tensorflow {
namespace openvino_tensorflow {

struct GlobalContext {
  ov::Core ie_core;
};

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif
