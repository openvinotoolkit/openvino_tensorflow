/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef IE_VADM_ENGINE_H_
#define IE_VADM_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include "openvino_tensorflow/ie_backend_engine.h"

namespace tensorflow {
namespace openvino_tensorflow {

class IE_VADM_Engine : public IE_Backend_Engine {
 public:
  // IE_VADM_Engine(InferenceEngine::CNNNetwork ie_network);
  IE_VADM_Engine(std::shared_ptr<ov::Model> model);
  ~IE_VADM_Engine();

  // Executes the inference
  virtual void infer(std::vector<std::shared_ptr<IETensor>>& inputs,
                     std::vector<std::string>& input_names,
                     std::vector<std::shared_ptr<IETensor>>& outputs,
                     std::vector<std::string>& output_names,
                     std::vector<std::shared_ptr<IETensor>>& hoisted_params,
                     std::vector<std::string>& param_names);

  virtual const std::vector<size_t> get_output_shape(const int i) {
    std::vector<size_t> shape = m_model->get_results()[i]->get_shape();
    if (m_multi_req_execution && m_orig_batch_size > 0 && shape.size() > 1) {
      shape[0] = m_orig_batch_size;
    }
    return shape;
  };

 private:
  int m_orig_batch_size;
  bool m_has_batch;
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // IE_VADM_ENGINE_H_
