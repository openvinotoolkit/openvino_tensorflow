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

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"

namespace tensorflow {
namespace ngraph_bridge {

class IETensor : public ngraph::runtime::Tensor {
 public:
  IETensor(const ngraph::element::Type& element_type,
           const ngraph::Shape& shape);
  IETensor(const ngraph::element::Type& element_type,
           const ngraph::PartialShape& shape);
  IETensor(const ngraph::element::Type& element_type,
           const ngraph::Shape& shape, void* memory_pointer);

  void write(const void* src, size_t bytes) override;
  void read(void* dst, size_t bytes) const override;

  const void* get_data_ptr() const;
  InferenceEngine::MemoryBlob::Ptr get_blob() { return m_blob; }

 private:
  IETensor(const IETensor&) = delete;
  IETensor(IETensor&&) = delete;
  IETensor& operator=(const IETensor&) = delete;
  InferenceEngine::MemoryBlob::Ptr m_blob;
};
}
}
