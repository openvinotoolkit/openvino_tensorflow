/*****************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#pragma once

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"

#include "openvino/openvino.hpp"

namespace tensorflow {
namespace openvino_tensorflow {

class IETensor : public ov::Tensor {
 public:
  IETensor(const ov::element::Type& element_type, const ov::Shape& shape);
  IETensor(const ov::element::Type& element_type, const ov::Shape& shape,
           void* memory_pointer);
  ~IETensor();

  void write(const void* src, size_t bytes);
  void read(void* dst, size_t bytes) const;

 private:
  IETensor(const IETensor&) = delete;
  IETensor(IETensor&&) = delete;
  IETensor& operator=(const IETensor&) = delete;
};

// A simple TensorBuffer implementation that allows us to create Tensors that
// take ownership of pre-allocated memory.
class IETensorBuffer : public TensorBuffer {
 public:
  IETensorBuffer(std::shared_ptr<IETensor> tensor)
      : TensorBuffer(const_cast<void*>(tensor->data())),
        size_(tensor->get_byte_size()),
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

}  // namespace openvino_tensorflow
}  // namespace tensorflow
