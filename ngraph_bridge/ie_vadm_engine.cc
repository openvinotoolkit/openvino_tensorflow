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

#include <iostream>

#include "ngraph_bridge/ie_utils.h"
#include "ngraph_bridge/ie_vadm_engine.h"

namespace tensorflow {
namespace ngraph_bridge {

IE_VADM_Engine::IE_VADM_Engine(InferenceEngine::CNNNetwork ie_network)
    : IE_Backend_Engine(ie_network, "HDDL"),
      m_orig_batch_size(ie_network.getBatchSize()) {}

IE_VADM_Engine::~IE_VADM_Engine() {}

void IE_VADM_Engine::infer(
    std::vector<std::shared_ptr<IETensor>>& inputs,
    std::vector<std::string>& input_names,
    std::vector<std::shared_ptr<IETensor>>& outputs,
    std::vector<std::string>& output_names,
    std::vector<std::shared_ptr<IETensor>>& hoisted_params,
    std::vector<std::string>& param_names) {
  // Batch size is 0 and the number of requests is 1 when
  // multi request execution is disabled.
  int num_req = 1;
  int batch_size = 0;

  if (m_multi_req_execution && inputs.size() == 1 && inputs[0] != nullptr &&
      inputs[0]->get_blob()->getTensorDesc().getDims().size() > 1) {
    // Set the batch size per request and number of requests
    batch_size = IE_Utils::GetInputBatchSize(
        inputs[0]->get_blob()->getTensorDesc().getDims()[0], m_device);
    num_req = inputs[0]->get_blob()->getTensorDesc().getDims()[0] / batch_size;
    if (m_network.getBatchSize() != batch_size)
      m_network.setBatchSize(batch_size);
  }

  // Create requests
  load_network();
  while (m_infer_reqs.size() < num_req) {
    m_infer_reqs.push_back(m_exe_network.CreateInferRequest());
  }
  std::vector<InferenceEngine::MemoryBlob::Ptr> in_blobs(inputs.size() *
                                                         num_req);
  std::vector<InferenceEngine::MemoryBlob::Ptr> param_blobs(
      hoisted_params.size());
  std::vector<InferenceEngine::MemoryBlob::Ptr> out_blobs(outputs.size() *
                                                          num_req);
  //  Prepare input blobs
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] == nullptr) continue;
    InferenceEngine::TensorDesc desc = inputs[i]->get_blob()->getTensorDesc();
    InferenceEngine::Precision prec = desc.getPrecision();
    const void* input_data_pointer = inputs[i]->get_data_ptr();
    std::string input_name = input_names[i];
    size_t size = inputs[i]->get_blob()->byteSize();

    InferenceEngine::SizeVector req_shape(desc.getDims());
    if (batch_size != 0) {
      req_shape[0] = batch_size;
      desc.setDims(req_shape);
    }
    for (int j = 0; j < num_req; j++) {
      size_t req_size = size / num_req;
      const void* data_ptr =
          (void*)((uint64_t)(input_data_pointer) + req_size * j);
      int in_idx = i * num_req + j;
      IE_Utils::CreateBlob(desc, prec, data_ptr, req_size, in_blobs[in_idx]);
      m_infer_reqs[j].SetBlob(input_name, in_blobs[in_idx]);
    }
  }
  for (int i = 0; i < hoisted_params.size(); i++) {
    if (hoisted_params[i] == nullptr) continue;
    InferenceEngine::TensorDesc desc =
        hoisted_params[i]->get_blob()->getTensorDesc();
    InferenceEngine::Precision prec = desc.getPrecision();
    const void* param_data_pointer = hoisted_params[i]->get_data_ptr();
    std::string param_name = param_names[i];
    size_t size = hoisted_params[i]->get_blob()->byteSize();

    InferenceEngine::SizeVector req_shape(desc.getDims());
    IE_Utils::CreateBlob(desc, prec, param_data_pointer, size, param_blobs[i]);
    for (int j = 0; j < num_req; j++) {
      m_infer_reqs[j].SetBlob(param_name, param_blobs[i]);
    }
  }

  // Prepare output blobs
  for (int i = 0; i < outputs.size(); i++) {
    out_blobs[i] = nullptr;
    if (outputs[i] != nullptr) {
      InferenceEngine::TensorDesc desc =
          outputs[i]->get_blob()->getTensorDesc();
      InferenceEngine::Precision prec = desc.getPrecision();
      InferenceEngine::Layout layout = desc.getLayout();
      const void* output_data_pointer = outputs[i]->get_data_ptr();
      std::string output_name = output_names[i];
      size_t size = outputs[i]->get_blob()->byteSize();

      InferenceEngine::SizeVector req_shape(desc.getDims());
      if (batch_size != 0) {
        req_shape[0] = batch_size;
        desc.setDims(req_shape);
      }

      InferenceEngine::TensorDesc req_desc(prec, req_shape, layout);
      for (int j = 0; j < num_req; j++) {
        size_t req_size = size / num_req;
        const void* data_ptr =
            (void*)((uint64_t)(output_data_pointer) + req_size * j);
        int out_idx = i * num_req + j;
        IE_Utils::CreateBlob(req_desc, prec, data_ptr, req_size,
                             out_blobs[out_idx]);
        m_infer_reqs[j].SetBlob(output_name, out_blobs[out_idx]);
      }
    }
  }

  // Start Inference Requests
  for (int i = 0; i < num_req; i++) {
    start_async_inference(i);
  }
  // Complete Inference Requests
  for (int i = 0; i < num_req; i++) {
    complete_async_inference(i);
  }

  // Set dynamic output blobs
  for (int i = 0; i < outputs.size(); i++) {
    if (outputs[i] == nullptr) {
      auto blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(
          m_infer_reqs[0].GetBlob(output_names[i]));
      outputs[i] = std::make_shared<IETensor>(blob);
    }
  }

  for (int i = 0; i < in_blobs.size(); i++) {
    in_blobs[i]->deallocate();
  }
  for (int i = 0; i < out_blobs.size(); i++) {
    if (out_blobs[i] != nullptr) {
      out_blobs[i]->deallocate();
    }
  }
  for (int i = 0; i < param_blobs.size(); i++) {
    param_blobs[i]->deallocate();
  }
}
}
}
