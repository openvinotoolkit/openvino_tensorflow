/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef OPENVINO_TF_BRIDGE_DEASSIGN_CLUSTERS_H_
#define OPENVINO_TF_BRIDGE_DEASSIGN_CLUSTERS_H_
#pragma once

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/analytical_cost_estimator.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"

namespace tensorflow {

namespace openvino_tensorflow {

Status DeassignClusters(Graph* graph, tensorflow::grappler::Cluster* cluster);

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_BRIDGE_DEASSIGN_CLUSTERS_H_
