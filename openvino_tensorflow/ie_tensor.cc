/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#include <cstring>
#include <memory>
#include <utility>

#include "ngraph/ngraph.hpp"

#include "ie_layouts.h"
#include "ie_precision.hpp"
#include "ie_tensor.h"
#include "ie_utils.h"

using namespace ngraph;
using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

IETensor::IETensor(const element::Type& element_type, const Shape& shape_,
                   void* memory_pointer)
    : runtime::Tensor(
          make_shared<descriptor::Tensor>(element_type, shape_, "")) {
  InferenceEngine::SizeVector shape = shape_;
  InferenceEngine::Precision precision = IE_Utils::toPrecision(element_type);
  InferenceEngine::Layout layout =
      InferenceEngine::TensorDesc::getLayoutByDims(shape);

  auto desc = InferenceEngine::TensorDesc(precision, shape, layout);
  auto size = shape_size(shape_) * element_type.size();
  InferenceEngine::MemoryBlob::Ptr ie_blob;
  IE_Utils::CreateBlob(desc, precision, memory_pointer, size, ie_blob);
  m_blob = ie_blob;
}

IETensor::IETensor(const element::Type& element_type, const Shape& shape)
    : IETensor(element_type, shape, nullptr) {}

IETensor::IETensor(const element::Type& element_type, const PartialShape& shape)
    : runtime::Tensor(
          make_shared<descriptor::Tensor>(element_type, shape, "")) {
  throw runtime_error("partial shapes not supported.");
}

IETensor::IETensor(InferenceEngine::Blob::Ptr blob)
    : runtime::Tensor(make_shared<descriptor::Tensor>(
          IE_Utils::fromPrecision(blob->getTensorDesc().getPrecision()),
          Shape(blob->getTensorDesc().getDims()), "")),
      m_blob(blob) {}

IETensor::~IETensor() {}

void IETensor::write(const void* src, size_t bytes) {
  const int8_t* src_ptr = static_cast<const int8_t*>(src);
  if (src_ptr == nullptr) {
    return;
  }

  auto blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(m_blob);
  if (blob == nullptr) {
    THROW_IE_EXCEPTION << "blob is nullptr";
  }
  auto lm = blob->wmap();
  uint8_t* output_ptr = lm.as<uint8_t*>();
  copy(src_ptr, src_ptr + bytes, output_ptr);
}

void IETensor::read(void* dst, size_t bytes) const {
  int8_t* dst_ptr = static_cast<int8_t*>(dst);
  if (dst_ptr == nullptr) {
    return;
  }

  auto blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(m_blob);
  if (blob == nullptr) {
    THROW_IE_EXCEPTION << "blob is nullptr";
  }
  auto lm = blob->rmap();
  uint8_t* output_ptr = lm.as<uint8_t*>();
  copy(output_ptr, output_ptr + bytes, dst_ptr);
}

const void* IETensor::get_data_ptr() const {
  auto blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(m_blob);
  if (blob == nullptr) {
    THROW_IE_EXCEPTION << "blob is nullptr";
  }
  auto lm = blob->rwmap();
  return lm.as<void*>();
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
