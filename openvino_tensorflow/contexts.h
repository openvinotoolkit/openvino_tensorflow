/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef CONTEXTS_H
#define CONTEXTS_H

//#include <inference_engine.hpp>
#include "openvino/openvino.hpp"

namespace tensorflow {
namespace openvino_tensorflow {

struct GlobalContext {
  ov::Core ie_core;
};

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif
