/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#pragma once

#ifndef OPENVINO_TF_MARK_FOR_CLUSTERING_H_
#define OPENVINO_TF_MARK_FOR_CLUSTERING_H_

#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"

#include "openvino_tensorflow/backend.h"

namespace tensorflow {
namespace openvino_tensorflow {

Status IsSupportedByBackend(
    const Node* node, std::shared_ptr<Backend> op_backend,
    const std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>&
        TFtoNgraphOpMap,
    bool& is_supported);
bool NodeIsMarkedForClustering(const Node* node);

// Returns the static input indexes in vector static_input_indexes
void GetStaticInputs(const Node* node,
                     std::vector<int32>* static_input_indexes);

// Returns True if the index-th input is static
bool InputIsStatic(const Node* node, int index);

// Returns the static input indexes of the graph in vector static_input_indexes
Status GetStaticInputs(Graph* graph, std::vector<int32>* static_input_indexes);

using SetAttributesFunction = std::function<Status(Node*)>;
const std::map<std::string, SetAttributesFunction>& GetAttributeSetters();

const std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>&
GetTFToNgOpMap();

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_MARK_FOR_CLUSTERING_H_
