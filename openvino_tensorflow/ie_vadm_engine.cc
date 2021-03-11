/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>

#include "openvino_tensorflow/ie_utils.h"
#include "openvino_tensorflow/ie_vadm_engine.h"

namespace tensorflow {
namespace openvino_tensorflow {

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
    if(m_device != "MYRIAD" && m_device != "HDDL") {
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
    } else {
      const void* input_data_pointer = inputs[i]->get_data_ptr();
      size_t size = inputs[i]->get_blob()->byteSize();
      //size_t req_size = size / num_req;
      for (int j = 0; j < num_req; j++) {
        auto input_blob = m_infer_reqs[j].GetBlob(input_names[i]);
       InferenceEngine::MemoryBlob::Ptr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(input_blob);
        auto minputHolder = minput->wmap();

        auto inputBlobData = minputHolder.as<uint8_t*>();
        size_t input_data_size = input_blob->byteSize();
       auto data_ptr = (uint8_t*)((uint64_t)(input_data_pointer)+input_data_size*j);
       std::copy(data_ptr, data_ptr + input_data_size, inputBlobData);
        //inputs[i]->read((void*)inputBlobData, input_data_size);
      }
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
      InferenceEngine::TensorDesc desc = blob->getTensorDesc();
      InferenceEngine::Precision prec = desc.getPrecision();
      InferenceEngine::Layout layout = desc.getLayout();
      InferenceEngine::SizeVector out_shape(desc.getDims());
      if (batch_size != 0) {
        out_shape[0] = m_orig_batch_size;
        desc.setDims(out_shape);
      }
      size_t req_size = blob->byteSize();
      size_t out_size = req_size * num_req;

      InferenceEngine::MemoryBlob::Ptr out_blob;
      IE_Utils::CreateBlob(desc, prec, nullptr, out_size,
                           out_blob);
      outputs[i] = std::make_shared<IETensor>(out_blob);
      auto lm = out_blob->rwmap();
      uint8_t* out_ptr = lm.as<uint8_t*>();
      for (int j = 0; j < num_req; j++) {
        blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(
            m_infer_reqs[j].GetBlob(output_names[i]));
        auto req_lm = blob->rwmap();
        uint8_t* req_ptr = req_lm.as<uint8_t*>();
       std::copy(req_ptr, req_ptr + req_size, out_ptr + (req_size*j));
      }
    }
  }
}
}// namespace openvino_tensorflow
}// namespace tensorflow
