//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <string>

#include "ngraph/ngraph.hpp"

#include "ngraph_bridge/executable.h"
#include "ngraph_bridge/ie_tensor.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

class Backend {
 public:
  Backend(const string& configuration_string);
  ~Backend();

  shared_ptr<Executable> compile(shared_ptr<ngraph::Function> func,
                                 bool enable_performance_data = false);
  void remove_compiled_function(std::shared_ptr<Executable> exec);
  bool is_supported(const ngraph::Node& node) const;

  shared_ptr<ngraph::runtime::Tensor> create_dynamic_tensor(
      const ngraph::element::Type& type, const ngraph::PartialShape& shape);

  static vector<string> get_registered_devices();

  shared_ptr<ngraph::runtime::Tensor> create_tensor(
      const ngraph::element::Type& element_type, const ngraph::Shape& shape);

  shared_ptr<ngraph::runtime::Tensor> create_tensor(
      const ngraph::element::Type& element_type, const ngraph::Shape& shape,
      void* data);

  template <typename T>
  shared_ptr<ngraph::runtime::Tensor> create_tensor(ngraph::element::Type type,
                                                    ngraph::Shape shape,
                                                    T* data) {
    auto tensor = make_shared<IETensor>(type, shape);
    size_t size = shape_size(shape);
    tensor->write(data, size * sizeof(T));
    return tensor;
  }

 private:
  std::mutex m_exec_map_mutex;
  std::unordered_map<std::shared_ptr<ngraph::Function>,
                     std::shared_ptr<Executable>>
      m_exec_map;
  string m_device;
};
}
}
