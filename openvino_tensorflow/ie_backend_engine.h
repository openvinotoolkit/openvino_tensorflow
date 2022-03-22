/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef IE_BACKEND_ENGINE_H_
#define IE_BACKEND_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include "openvino_tensorflow/ie_tensor.h"

namespace tensorflow {
namespace openvino_tensorflow {

class IE_Backend_Engine {
 public:
  IE_Backend_Engine(std::shared_ptr<ov::Model> model, std::string device);
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

  // Returns the OpenVINO Model
  std::shared_ptr<ov::Model> get_model();

  virtual const std::vector<size_t> get_output_shape(const int i) = 0;

  const int get_input_idx(const std::string name) const;
  const int get_output_idx(const std::string name) const;

 protected:
  std::shared_ptr<ov::Model> m_model;
  ov::CompiledModel m_compiled_model;
  std::vector<ov::InferRequest> m_infer_reqs;
  std::string m_device;
  bool m_multi_req_execution;
  bool m_network_ready;
  std::vector<int> m_in_idx;
  std::vector<int> m_out_idx;
  std::vector<int> m_param_idx;

  virtual void start_async_inference(const int req_id);
  virtual void complete_async_inference(const int req_id);
  virtual void load_network();
};
}  // namespace openvino_tensorflow
}  // namesoace tensorflow

#endif  // IE_BACKEND_ENGINE_H_
