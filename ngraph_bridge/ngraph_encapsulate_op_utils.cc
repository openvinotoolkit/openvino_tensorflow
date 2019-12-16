/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include "ngraph_bridge/ngraph_encapsulate_op_utils.h"
#include "ngraph_bridge/ngraph_prefetch_shared_data.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status GetPipelinedIOTensorsReadyForExecution(
    OpKernelContext* ctx, std::vector<Tensor>& tf_input_tensors,
    shared_ptr<PipelinedTensorsStore>& pipelined_tensor_store,
    shared_ptr<NGraphTensorManager>& tensor_manager,
    std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>&
        pipelined_io_tensors) {
  auto io_tensors = pipelined_tensor_store->get_tensors();

  int current_iter_pipeline_depth = get<0>(io_tensors);
  PipelinedTensorVector ng_pipelined_inputs = get<1>(io_tensors);
  PipelinedTensorVector ng_pipelined_outputs = get<2>(io_tensors);
  auto pipelined_input_indexes = tensor_manager->GetPipelinedInputIndexes();
  auto pipelined_output_indexes = tensor_manager->GetPipelinedOutputIndexes();

  if (current_iter_pipeline_depth < 0) {
    return errors::Internal("No free tensor available");
  }

  if (pipelined_input_indexes.size() != ng_pipelined_inputs.size()) {
    return errors::Internal(
        "Pipelined input tensors size ", ng_pipelined_inputs.size(),
        " does not match the no. of pipelined inputs indexes ",
        pipelined_input_indexes.size());
  }

  if (pipelined_output_indexes.size() != ng_pipelined_outputs.size()) {
    return errors::Internal(
        "Pipelined output tensors size ", ng_pipelined_outputs.size(),
        " does not match the no. of pipelined output indexes ",
        pipelined_output_indexes.size());
  }

  bool skip_tf2ng_copy = false;
  // Prefetch only if there are input tensors that are prefetched && prefetch
  // has been requested
  // [TODO] we support prefetching only when there is atmost 1 encap
  // that has prefetched inputs
  if (std::getenv(NGraphPrefetchSharedResouce::NGRAPH_TF_USE_PREFETCH) !=
          nullptr &&
      !(tensor_manager->GetPipelinedInputIndexesThatArePrefetched()).empty()) {
    NGRAPH_VLOG(2) << "[PREFETCH] NGRAPH_TF_USE_PREFETCH Set";
    // Set the prefetch shared obj if applicable
    NGraphPrefetchSharedResouce* shared_data = nullptr;
    Status s = ctx->resource_manager()->Lookup(
        NGraphPrefetchSharedResouce::CONTAINER_NAME,
        NGraphPrefetchSharedResouce::RESOURCE_NAME, &shared_data);

    if (!s.ok()) {
      // We are using this for the first time i.e., we need to do the following
      // 1. Create the shared data object
      // 2. We get another pipelined tensor pair for the current iteration and
      //   add it to the shared data. It will be accessed by prefetcher to copy
      //   the prefetched inputs to device
      auto ng_prefetch_input_indexes =
          tensor_manager->GetPipelinedInputIndexesThatArePrefetched();

      shared_data = new NGraphPrefetchSharedResouce(
          tensor_manager->GetName(), tensor_manager->GetGraphId(),
          tensor_manager->GetClusterId(), ng_prefetch_input_indexes);

      // Get the set of IO tensors for the next iteration
      std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>
          io_tensors_next_iter;
      io_tensors_next_iter = pipelined_tensor_store->get_tensors();

      // Save the prefetched input ngTensors for the next iteration
      NGraphPrefetchSharedResouce::IOTensorBundle next_io_tensor_bundle{
          get<0>(io_tensors_next_iter), get<1>(io_tensors_next_iter),
          get<2>(io_tensors_next_iter)};

      if (current_iter_pipeline_depth != (!next_io_tensor_bundle.Id)) {
        return errors::Internal("Current Pipeline Depth is ",
                                current_iter_pipeline_depth,
                                " and next iter pipeline depth is also  ",
                                next_io_tensor_bundle.Id);
      }

      shared_data->AddNextIOTensorBundleForDeviceTransfer(
          next_io_tensor_bundle);

      ctx->SetStatus(ctx->resource_manager()->Create(
          NGraphPrefetchSharedResouce::CONTAINER_NAME,
          NGraphPrefetchSharedResouce::RESOURCE_NAME, shared_data));
      // Continue the execution with the currently supplied TF tensor for the
      // last time
      NGRAPH_VLOG(2) << "[PREFETCH] COMPUTE: Creating the shared object to "
                        "signal prefetching";
    } else {
      int prefetch_buffer_depth = shared_data->GetBufferDepth();
      int skip_count = shared_data->GetSkipCount();
      NGRAPH_VLOG(2) << "[PREFETCH] COMPUTE: DEPTH: " << prefetch_buffer_depth
                     << " skip count; " << skip_count;
      if (skip_count >= prefetch_buffer_depth) {
        // We have been using the pipelined tensors - therefore do the
        // following:
        // 1. Save the prefetched Input/Output tensors for the current iteration
        //    to the shared data object so that the prefetcher
        //    can continue with copying the next set of inout tensor to the
        //    device
        // 3. Execute the nGraph call for this iteration using the
        //    nG prefeteched input tensors we got from the shared data

        // Add the current prefetched tensors for the next iteration
        // Get prefetched inputs
        NGraphPrefetchSharedResouce::IOTensorBundle prefetch_io_tensor_bundle{
            current_iter_pipeline_depth, ng_pipelined_inputs,
            ng_pipelined_outputs};
        shared_data->AddNextIOTensorBundleForDeviceTransfer(
            prefetch_io_tensor_bundle);

        // Update the input_tensors with the one ready for exdcution
        auto ng_io_tensor_bundle_ready =
            shared_data->GetNextIOTensorBundleReadyForDeviceExecution();
        current_iter_pipeline_depth = ng_io_tensor_bundle_ready.Id;
        ng_pipelined_inputs = ng_io_tensor_bundle_ready.Inputs;
        ng_pipelined_outputs = ng_io_tensor_bundle_ready.Outputs;
        if (current_iter_pipeline_depth != (!prefetch_io_tensor_bundle.Id)) {
          return errors::Internal("Current Pipeline Depth is ",
                                  current_iter_pipeline_depth,
                                  " and next iter pipeline depth is ", "also ",
                                  prefetch_io_tensor_bundle.Id);
        }
        skip_tf2ng_copy = true;
        NGRAPH_VLOG(2) << "[PREFETCH] COMPUTE: Using device tensors";
      }
      shared_data->IncrSkipCount();
    }
  }

  // Allocate the input/
  ngraph::Event event_copy_input_tensor("Copy Pipelined Input Tensors", "", "");

  if (!skip_tf2ng_copy) {
    // All pipelined inputs are copied

    for (auto i = 0; i < pipelined_input_indexes.size(); i++) {
      int tf_index = pipelined_input_indexes[i];

      ng::element::Type ng_element_type;
      TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(
          tf_input_tensors[tf_index].dtype(), &ng_element_type));
      void* current_src_ptr =
          (void*)DMAHelper::base(&tf_input_tensors[tf_index]);
      try {
        ng_pipelined_inputs[i]->write(
            current_src_ptr, ng_pipelined_inputs[i]->get_element_count() *
                                 ng_element_type.size());
      } catch (const std::exception& exp) {
        return errors::Internal("Error copying TF tensor to device tensor: ",
                                exp.what());
      } catch (...) {
        return errors::Internal("Error copying TF tensor to device tensor");
      }
    }
  } else {
    // All pipelined inputs that are not prefetched are copied
    // Note skip_tf2ng_copy will be true only when PREFETCH is enabled via env
    // flag

    // Gives the TF input index : wrt to all inputs
    auto pipelined_not_prefetched_input_indexes =
        tensor_manager->GetPipelinedButNotPrefetchedInputIndexes();

    // Gives the corresponding pipelined input index : wrt pipelined
    auto pipelined_input_indexes_not_prefetched =
        tensor_manager->GetPipelinedInputIndexesThatAreNotPrefetched();

    // Gives the mapping for corresponding
    for (auto i = 0; i < pipelined_input_indexes_not_prefetched.size(); i++) {
      int tf_index = pipelined_not_prefetched_input_indexes[i];
      int ng_index = pipelined_input_indexes_not_prefetched[i];
      ng::element::Type ng_element_type;
      TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(
          tf_input_tensors[tf_index].dtype(), &ng_element_type));
      void* current_src_ptr =
          (void*)DMAHelper::base(&tf_input_tensors[tf_index]);
      try {
        ng_pipelined_inputs[ng_index]->write(
            current_src_ptr,
            ng_pipelined_inputs[ng_index]->get_element_count() *
                ng_element_type.size());
      } catch (const std::exception& exp) {
        return errors::Internal("Error copying TF tensor to device tensor: ",
                                exp.what());
      } catch (...) {
        return errors::Internal("Error copying TF tensor to device tensor");
      }
    }
  }
  event_copy_input_tensor.Stop();
  ngraph::Event::write_trace(event_copy_input_tensor);

  pipelined_io_tensors = make_tuple(current_iter_pipeline_depth,
                                    ng_pipelined_inputs, ng_pipelined_outputs);

  return Status::OK();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
