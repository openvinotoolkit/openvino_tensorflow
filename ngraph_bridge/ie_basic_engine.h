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

#ifndef IE_BASIC_ENGINE_H_
#define IE_BASIC_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include <ie_core.hpp>

#include "ngraph_bridge/ie_backend_engine.h"

namespace tensorflow {
namespace ngraph_bridge {

class IE_Basic_Engine : public IE_Backend_Engine {
 public:
  IE_Basic_Engine(InferenceEngine::CNNNetwork ie_network, std::string device);
  ~IE_Basic_Engine();

  // Executes the inference
  virtual void infer(std::vector<std::shared_ptr<IETensor>>& inputs,
                     std::vector<std::string>& input_names,
                     std::vector<std::shared_ptr<IETensor>>& outputs,
                     std::vector<std::string>& output_names,
                     std::vector<std::shared_ptr<IETensor>>& hoisted_params,
                     std::vector<std::string>& param_names);

  virtual const std::vector<size_t> get_output_shape(const int i) {
    return m_func->get_results()[i]->get_shape();
  };
};
}
}

#endif  // IE_BASIC_ENGINE_H_
