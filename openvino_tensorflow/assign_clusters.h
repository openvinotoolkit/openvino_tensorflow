/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef OPENVINO_TF_BRIDGE_ASSIGN_CLUSTERS_H_
#define OPENVINO_TF_BRIDGE_ASSIGN_CLUSTERS_H_
#pragma once

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace openvino_tensorflow {

Status AssignClusters(Graph* graph);
Status GetNodeCluster(const Node* node, int* cluster);

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_BRIDGE_ASSIGN_CLUSTERS_H_
