/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#pragma once

#ifndef OPENVINO_TF_ADD_IDENTITYN_H_
#define OPENVINO_TF_ADD_IDENTITYN_H_

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "logging/ovtf_log.h"

namespace tensorflow {

namespace openvino_tensorflow {

Status AddIdentityN(Graph* graph, std::set<string> skip_these_nodes);

}  // namespace openvino_tensorflow

}  // namespace tensorflow
#endif  // OPENVINO_TF_ADD_IDENTITYN_H_
