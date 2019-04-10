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
#ifndef NGRAPH_TF_ENTER_IN_CATALOG_H_
#define NGRAPH_TF_ENTER_IN_CATALOG_H_
#pragma once

#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"
#include "ngraph_catalog.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// 1. Populate the NGraphCatalog
// 2. Attach Graph Ids to the node

// Some terms:
// NGraphSupported Ops : NGraphVariable, NGraphAssign, NGraphEncapsulate
// NGraphVariableType Ops : NGraphVariable, NGraphAssign
// NG-Tensor : ngraph backend tensor

// TF's Variable Op is a wrapper on a persistent TF-tensor which is stored
// in the TF Container and can be accessed/retrieved by TF Resource Manager
// The NGraphVariable Op is a wrapper on a pair of TF-Tensor and NG-Tensor that
// are synced lazily (when required)

// We collect the below information for the catalog
// 1. If the NGraphSupportedOp gets input from a NGraphVariableType Op,
// it can directly access the ng-tensor via the TF Resource Manager using the
// shared Name
// We add mapping of {graphId_nodename_InputIndex : Shared_Name} to the
// InputVariableSharedNameMap
//
// 2. If the output of NGraphEncapsulate Op is an input to NGraphVariableType
// Op, we store this NG-Tensor
// so that it can be directly accessed in compute call of NGraphVariableType.
// We add mapping of {graphId_encapnodename_OutputIndex : NG-Tensor} to the
// EncapOutputTensorMap
//
// 3. If the output of NGraphEncapsulate Op is not required by a TF Op or
// NGraphEncapsulate Op,
// then we can avoid copying it to HOST
// We add mapping of {encapnodename : set of OutputIndexes that need a copy} to
// the EncapsulateOutputCopyIndexesMap
//

Status EnterInCatalog(Graph* graph, int graph_id);

}  // ngraph_bridge
}  // tensorflow

#endif
