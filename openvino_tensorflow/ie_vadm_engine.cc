/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>
#include <memory>

#include "openvino_tensorflow/ie_utils.h"
#include "openvino_tensorflow/ie_vadm_engine.h"

namespace tensorflow {
namespace openvino_tensorflow {

IE_VADM_Engine::IE_VADM_Engine(std::shared_ptr<ov::Model> model)
    : IE_Backend_Engine(model, "HDDL"), m_orig_batch_size(0) {
  ov::Dimension batch_dim = ov::get_batch(m_model);
  m_orig_batch_size = (batch_dim.is_static() ? batch_dim.get_length() : 1);
}

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

  int multi_req_support = false;
  int tmp_batch = 0;
  if (m_multi_req_execution && hoisted_params.size() == 0) {
    multi_req_support = true;
    for (int i = 0; i < inputs.size(); i++) {
      if (inputs[i] == nullptr) {
        continue;
      }
      if (inputs[i]->get_shape().size() < 2) {
        multi_req_support = false;
        break;
      }
      if (inputs[i]->get_shape()[0] < 2) {
        multi_req_support = false;
        break;
      }
      if (tmp_batch == 0) tmp_batch = inputs[i]->get_shape()[0];
      if (inputs[i]->get_shape()[0] != tmp_batch) {
        multi_req_support = false;
        break;
      }
    }
  }

  if (multi_req_support && tmp_batch != 0) {
    // Set the batch size per request and number of requests
    batch_size = IE_Utils::GetInputBatchSize(tmp_batch, m_device);
    assert(batch_size > 0);
    num_req = tmp_batch / batch_size;
    ov::Dimension batch_dim = ov::get_batch(m_model);
    int64_t model_batch_size =
        (batch_dim.is_static() ? batch_dim.get_length() : 1);
    if (model_batch_size != batch_size)
      ov::set_batch(m_model, ov::Dimension(batch_size));
  } else if (m_multi_req_execution) {
    // Batching is enabled but the cluster is not compatible
    std::cout << "OVTF_MESSAGE: Batching is disabled. The graph is"
              << " not compatible for batching." << std::endl;
  }

  // Create requests
  load_network();
  while (m_infer_reqs.size() < num_req) {
    m_infer_reqs.push_back(m_compiled_model.create_infer_request());
  }
  // std::vector<InferenceEngine::MemoryBlob::Ptr> in_blobs(inputs.size() *
  //                                                       num_req);
  // std::vector<InferenceEngine::MemoryBlob::Ptr> param_blobs(
  //    hoisted_params.size());
  // std::vector<InferenceEngine::MemoryBlob::Ptr> out_blobs(outputs.size() *
  //                                                        num_req);
  //  Prepare input blobs
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] == nullptr) continue;
    const void* input_data_pointer = inputs[i]->data();
    size_t size = inputs[i]->get_byte_size();
    for (int j = 0; j < num_req; j++) {
      const int in_idx = get_input_idx(input_names[i]);
      if (in_idx < 0) {
        throw std::runtime_error("Input parameter with friendly name " +
                                 input_names[i] + " not found in ov::Model");
      }
      auto tensor = m_infer_reqs[0].get_input_tensor(in_idx);
      size_t input_data_size = tensor.get_byte_size();
      auto data_ptr =
          (uint8_t*)((uint64_t)(input_data_pointer) + input_data_size * j);
      std::copy((uint8_t*)(data_ptr), ((uint8_t*)(data_ptr)) + input_data_size,
                (uint8_t*)(tensor.data()));
      ;
    }
  }
  for (int i = 0; i < hoisted_params.size(); i++) {
    if (hoisted_params[i] == nullptr) continue;
    const int in_idx = get_input_idx(param_names[i]);
    if (in_idx < 0) {
      throw std::runtime_error("Input parameter with friendly name " +
                               param_names[i] + " not found in ov::Model");
    }
    auto tensor = m_infer_reqs[0].get_input_tensor(in_idx);
    size_t input_data_size = tensor.get_byte_size();
    std::copy((uint8_t*)(hoisted_params[i]->data()),
              ((uint8_t*)(hoisted_params[i]->data())) + input_data_size,
              (uint8_t*)(tensor.data()));
    ;
  }

  // Start Inference Requests
  for (int i = 0; i < num_req; i++) {
    start_async_inference(i);
  }
  // Complete Inference Requests
  for (int i = 0; i < num_req; i++) {
    complete_async_inference(i);
  }

  // Set output tensors
  for (int i = 0; i < outputs.size(); i++) {
    if (outputs[i] == nullptr) {
      const int out_idx = get_output_idx(output_names[i]);
      if (out_idx < 0) {
        throw std::runtime_error("Output with friendly name " +
                                 output_names[i] + " not found in ov::Model");
      }
      auto tensor = m_infer_reqs[0].get_output_tensor(out_idx);
      ov::Shape out_shape = tensor.get_shape();
      if (batch_size == 0 || out_shape.size() < 2 ||
          out_shape[0] != batch_size) {
        outputs[i] = std::make_shared<IETensor>(tensor.get_element_type(),
                                                out_shape, tensor.data());
      } else {
        out_shape[0] = m_orig_batch_size;
        size_t req_size = tensor.get_byte_size();
        size_t out_size = req_size * num_req;
        outputs[i] =
            std::make_shared<IETensor>(tensor.get_element_type(), out_shape);
        uint8_t* out_ptr = (uint8_t*)(outputs[i]->data());
        for (int j = 0; j < num_req; j++) {
          auto req_tensor = m_infer_reqs[j].get_output_tensor(out_idx);
          uint8_t* req_ptr = (uint8_t*)(req_tensor.data());
          std::copy(req_ptr, req_ptr + req_size, out_ptr + (req_size * j));
        }
      }
    }
  }
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
