/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#pragma once

#ifndef NGRAPH_TF_MARK_FOR_CLUSTERING_H_
#define NGRAPH_TF_MARK_FOR_CLUSTERING_H_

#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"

#include "ngraph_bridge/backend.h"

namespace tensorflow {
namespace ngraph_bridge {

Status MarkForClustering(Graph* graph, std::set<string> skip_these_nodes);
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

using TypeConstraintMap =
    std::map<std::string, std::map<std::string, gtl::ArraySlice<DataType>>>;
const TypeConstraintMap& GetTypeConstraintMap();

using ConfirmationFunction = std::function<Status(Node*, bool*)>;
const std::map<std::string, ConfirmationFunction>& GetConfirmationMap();

const std::map<std::string, std::set<std::shared_ptr<ngraph::Node>>>&
GetTFToNgOpMap();

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_MARK_FOR_CLUSTERING_H_
