/*******************************************************************************
 * Copyright 2019 Intel Corporation
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
#ifndef NGRAPH_TF_REPLACE_OP_UTILITIES_H_
#define NGRAPH_TF_REPLACE_OP_UTILITIES_H_

#pragma once

#include "tensorflow/core/graph/graph.h"

using namespace std;
namespace tensorflow {

namespace ngraph_bridge {

Status ReplaceApplyGradientDescent(Graph* graph, Node* node, Node** replacement,
                                   const string replacement_node_name,
                                   const string replacement_op_type,
                                   const bool just_looking,
                                   const bool outputs_ng_supported,
                                   const int graph_id,
                                   const bool is_backend_set);

Status ReplaceAssign(Graph* graph, Node* node, Node** replacement,
                     const string replacement_node_name,
                     const string replacement_op_type, const bool just_looking,
                     const bool outputs_ng_supported, const int graph_id,
                     const bool is_backend_set);

Status ReplaceVariable(Graph* graph, Node* node, Node** replacement,
                       const string replacement_node_name,
                       const string replacement_op_type,
                       const bool just_looking, const bool outputs_ng_supported,
                       const int graph_id, const bool is_backend_set);

// Adds the edges that are incoming control edges to node
// as incoming control edges to the replacement node
// Removes the original edges
Status ReplaceInputControlEdges(Graph* graph, Node* node, Node* replacement);

// Adds the edges that are outgoing from node
// as outgoing edges to the replacement node
// Removes the original edges
Status ReplaceOutputEdges(Graph* graph, Node* node, Node* replacement);

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif  // NGRAPH_TF_REPLACE_OP_UTILITIES_H_