/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#include <cstring>
#include <memory>
#include <utility>

#include "ie_layouts.h"
#include "ie_precision.hpp"
#include "ie_tensor.h"
#include "ie_utils.h"

using namespace std;
using namespace ov;

namespace tensorflow {
namespace openvino_tensorflow {

IETensor::IETensor(const ov::element::Type& element_type, const Shape& shape_,
                   void* memory_pointer)
    : ov::Tensor(element_type, shape_, memory_pointer) {}

IETensor::IETensor(const ov::element::Type& element_type, const Shape& shape)
    : ov::Tensor(element_type, shape) {}

//IETensor::IETensor(const ov::element::Type& element_type, const PartialShape& shape)
//    : ov::Tensor(element_type, shape) {
//  throw runtime_error("partial shapes not supported.");
//}

IETensor::~IETensor() {}

void IETensor::write(const void* src, size_t bytes) {
  const int8_t* src_ptr = static_cast<const int8_t*>(src);
  if (src_ptr == nullptr) {
    return;
  }

  copy(src_ptr, src_ptr + bytes, (uint8_t*)(this->data()));
}

void IETensor::read(void* dst, size_t bytes) const {
  int8_t* dst_ptr = static_cast<int8_t*>(dst);
  if (dst_ptr == nullptr) {
    return;
  }

  copy((uint8_t*)(this->data()), ((uint8_t*)(this->data())) + bytes, dst_ptr);
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
