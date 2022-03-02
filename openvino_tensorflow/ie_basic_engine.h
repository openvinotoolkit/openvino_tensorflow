/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef IE_BASIC_ENGINE_H_
#define IE_BASIC_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include "openvino_tensorflow/ie_backend_engine.h"

namespace tensorflow {
namespace openvino_tensorflow {

class IE_Basic_Engine : public IE_Backend_Engine {
 public:
  // IE_Basic_Engine(InferenceEngine::CNNNetwork ie_network, std::string
  // device);
  IE_Basic_Engine(std::shared_ptr<ov::Model> model, std::string device);
  ~IE_Basic_Engine();

  // Executes the inference
  virtual void infer(std::vector<std::shared_ptr<IETensor>>& inputs,
                     std::vector<std::string>& input_names,
                     std::vector<std::shared_ptr<IETensor>>& outputs,
                     std::vector<std::string>& output_names,
                     std::vector<std::shared_ptr<IETensor>>& hoisted_params,
                     std::vector<std::string>& param_names);

  virtual const std::vector<size_t> get_output_shape(const int i) {
    return m_model->get_results()[i]->get_shape();
  };
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // IE_BASIC_ENGINE_H_
