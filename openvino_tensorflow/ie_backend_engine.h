/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef IE_BACKEND_ENGINE_H_
#define IE_BACKEND_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include <ie_core.hpp>

#include "openvino_tensorflow/ie_tensor.h"

namespace tensorflow {
namespace openvino_tensorflow {

class IE_Backend_Engine {
 public:
  IE_Backend_Engine(InferenceEngine::CNNNetwork ie_network, std::string device);
  ~IE_Backend_Engine();

  // Executes the inference
  virtual void infer(std::vector<std::shared_ptr<IETensor>>& inputs,
                     std::vector<std::string>& input_names,
                     std::vector<std::shared_ptr<IETensor>>& outputs,
                     std::vector<std::string>& output_names,
                     std::vector<std::shared_ptr<IETensor>>& hoisted_params,
                     std::vector<std::string>& param_names) = 0;

  // Returns output batch size based on the input batch size and the device
  // FIXME: This may not be needed
  virtual size_t get_output_batch_size(size_t inputBatchSize) const;

  // Enables multi request execution if the execution engine supprts
  void enable_multi_req_execution();
  // Disables multi request execution
  void disable_multi_req_execution();

  // Returns the NGraph Function from the CNNNetwork
  std::shared_ptr<ngraph::Function> get_func();

  virtual const std::vector<size_t> get_output_shape(const int i) = 0;

 protected:
  InferenceEngine::CNNNetwork m_network;
  std::shared_ptr<ngraph::Function> m_func;
  std::vector<InferenceEngine::InferRequest> m_infer_reqs;
  std::string m_device;
  bool m_multi_req_execution;
  InferenceEngine::ExecutableNetwork m_exe_network;
  bool m_network_ready;

  virtual void start_async_inference(const int req_id);
  virtual void complete_async_inference(const int req_id);
  virtual void load_network();
};
}  // namespace openvino_tensorflow
}  // namesoace tensorflow

#endif  // IE_BACKEND_ENGINE_H_
