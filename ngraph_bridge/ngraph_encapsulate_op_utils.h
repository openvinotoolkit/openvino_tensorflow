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

#ifndef NGRAPH_TF_BRIDGE_GET_PIPELINED_TENSORS_H
#define NGRAPH_TF_BRIDGE_GET_PIPELINED_TENSORS_H

#pragma once

#include "tensorflow/core/graph/graph.h"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_pipelined_tensors.h"
#include "ngraph_bridge/ngraph_tensor_manager.h"

using namespace std;
namespace tensorflow {

namespace ngraph_bridge {

// This function does the following
// 1. Gets pipelined tensors for current execution from pipelined tensor store
// (PTS)
// 2. If prefetch is enabled
//          a. if prefetch shared resource is not created
//               creates it
//               gets next set of tensors from PTS and adds it to the shared
//               object for prefetching
//          b.  else
//               gets the tensors from prefetch object and adds the tensors from
//               step 1 to the prefetch object
// 3. Copies the tf input tensors that are not prefetched to the ngraph
// pipelined input tensors
//

Status GetPipelinedIOTensorsReadyForExecution(
    OpKernelContext* ctx, const vector<Tensor>& tf_input_tensors,
    const shared_ptr<PipelinedTensorsStore>& pipelined_tensor_store,
    const shared_ptr<NGraphTensorManager>& tensor_manager,
    tuple<int, PipelinedTensorVector, PipelinedTensorVector>&
        pipelined_io_tensors);

// Assembles the different types of input and output tensors
// Variable tensors and pipelined tensors are put together in the right order
// into ng_inputs and ng_outputs
// 1. For input indexes that are fed by variables, get the variable tensors from
// context
// 2. For output indexes that are updating variables, get the variable tensors
// from context
//    This enable update-in-place
// 3. For input and output indexes that are pipelined, get the respective tensor
//
Status GetIOTensorsReadyForExecution(
    OpKernelContext* ctx, const shared_ptr<NGraphTensorManager>& tensor_manager,
    const PipelinedTensorVector& pipelined_in_tensors,
    const PipelinedTensorVector& pipelined_out_tensors,
    vector<shared_ptr<ng::runtime::Tensor>>& ng_inputs,
    vector<shared_ptr<ng::runtime::Tensor>>& ng_outputs);

// Gets the Tensor from OpKernelContext's Container for the given shared_name
Status GetTensorFromContext(const OpKernelContext* ctx,
                            const string& shared_name,
                            shared_ptr<ng::runtime::Tensor>& ng_tensor);

// Encapsulate Op updates the NGVariable's device tensor in-place
// ie. the NGVariable's backend tensor is updated
// Some of these Variables may be required by the TF ops and they will use the
// host tensor
// These were marked as "copy-to-tf" True in the Rewrite Phase
// We will update these tensors here
Status SyncOutputVarTensors(
    const OpKernelContext* ctx,
    const shared_ptr<NGraphTensorManager>& tensor_manager);

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_GET_PIPELINED_TENSORS_H
