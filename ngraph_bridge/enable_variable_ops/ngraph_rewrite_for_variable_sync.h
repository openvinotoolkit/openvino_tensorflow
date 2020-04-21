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

#ifndef NGRAPH_TF_REWRITE_FOR_VARIABLE_SYNC_H_
#define NGRAPH_TF_REWRITE_FOR_VARIABLE_SYNC_H_
#pragma once

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace ngraph_bridge {

// Rewrite for synchronization of variables
// 1. Assigns "update_tf_tensor: true" attribute to NGVariable and NGAssign Ops
//    when the Variable is going to be used (read) by a TF Op
//    Responsible for updating the NGraphVariable's TFTensor
//    inside the kernel of NGVariable and NGAssign Op
// 2. Adds NGraphVariableUpdateNGTensor Nodes
//    when the variable has been updated (written to) by a TF Op
//    Responsible for updating the NGraphVariable's NGTensor inside its kernel
Status RewriteForVariableSync(Graph* graph, int graph_id);

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_REWRITE_FOR_VARIABLE_SYNC_H_