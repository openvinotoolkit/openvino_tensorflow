/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef CONTEXTS_H
#define CONTEXTS_H

#include <inference_engine.hpp>

namespace tensorflow {
namespace openvino_tensorflow {

struct GlobalContext {
  InferenceEngine::Core ie_core;
};

} // namespace openvino_tensorflow
} // namespace tensorflow

#endif
