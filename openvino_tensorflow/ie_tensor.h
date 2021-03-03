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

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"

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
  IETensor(InferenceEngine::Blob::Ptr blob);
  ~IETensor() override;

  void write(const void* src, size_t bytes) override;
  void read(void* dst, size_t bytes) const override;

  const void* get_data_ptr() const;
  InferenceEngine::Blob::Ptr get_blob() { return m_blob; }

 private:
  IETensor(const IETensor&) = delete;
  IETensor(IETensor&&) = delete;
  IETensor& operator=(const IETensor&) = delete;
  InferenceEngine::Blob::Ptr m_blob;
};

// A simple TensorBuffer implementation that allows us to create Tensors that
// take ownership of pre-allocated memory.
class IETensorBuffer : public TensorBuffer {
 public:
  IETensorBuffer(std::shared_ptr<IETensor> tensor)
      : TensorBuffer(const_cast<void*>(tensor->get_data_ptr())),
        size_(tensor->get_size_in_bytes()),
        tensor_(tensor) {}

  size_t size() const override { return size_; }

  TensorBuffer* root_buffer() override { return this; }

  void FillAllocationDescription(AllocationDescription* proto) const override {
    proto->set_allocated_bytes(size_);
  }

 private:
  size_t size_;
  std::shared_ptr<IETensor> tensor_;
};
}
}
