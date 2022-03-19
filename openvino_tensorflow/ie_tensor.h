/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
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

}  // namespace openvino_tensorflow
}  // namespace tensorflow
