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

#include <cstring>
#include <memory>
#include <utility>

#include "ngraph/ngraph.hpp"

#include "ie_tensor.h"

using namespace ngraph;
using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

static InferenceEngine::Layout getLayoutByDims(size_t dims) {
  switch (dims) {
    case 0:
      return InferenceEngine::Layout::SCALAR;
    case 1:
      return InferenceEngine::Layout::C;
    case 2:
      return InferenceEngine::Layout::NC;
    case 3:
      return InferenceEngine::Layout::CHW;
    case 4:
      return InferenceEngine::Layout::NCHW;
    case 5:
      return InferenceEngine::Layout::NCDHW;
    case 6:
      return InferenceEngine::Layout::GOIDHW;
    default:
      THROW_IE_EXCEPTION << "Can't convert dims " << dims << " to IE layout!";
  }
}

static InferenceEngine::Precision getPrecision(
    const element::Type& element_type) {
  switch (element_type.get_type_enum()) {
    case element::Type_t::f32:
      return InferenceEngine::Precision::FP32;
    case element::Type_t::u8:
      return InferenceEngine::Precision::U8;
    case element::Type_t::i8:
      return InferenceEngine::Precision::I8;
    case element::Type_t::u16:
      return InferenceEngine::Precision::U16;
    case element::Type_t::i16:
      return InferenceEngine::Precision::I16;
    case element::Type_t::i32:
      return InferenceEngine::Precision::I32;
    case element::Type_t::u64:
      return InferenceEngine::Precision::U64;
    case element::Type_t::i64:
      return InferenceEngine::Precision::I64;
    case element::Type_t::boolean:
      return InferenceEngine::Precision::BOOL;
    default:
      THROW_IE_EXCEPTION << "Can't convert type " << element_type
                         << " to IE precision!";
  }
}

IETensor::IETensor(const element::Type& element_type, const Shape& shape_,
                   void* memory_pointer)
    : runtime::Tensor(
          make_shared<descriptor::Tensor>(element_type, shape_, "")) {
  m_descriptor->set_tensor_layout(
      make_shared<descriptor::layout::DenseTensorLayout>(*m_descriptor));

  InferenceEngine::SizeVector shape = shape_;
  InferenceEngine::Precision precision = getPrecision(element_type);
  InferenceEngine::Layout layout = getLayoutByDims(shape.size());

  auto desc = InferenceEngine::TensorDesc(precision, shape, layout);
  auto size = (memory_pointer == nullptr)
                  ? 0
                  : shape_size(shape_) * element_type.size();

  switch (element_type) {
    case element::Type_t::f32:
      m_blob = make_shared<InferenceEngine::TBlob<float>>(
          desc, (float*)memory_pointer, size);
      break;
    case element::Type_t::u8:
      m_blob = make_shared<InferenceEngine::TBlob<uint8_t>>(
          desc, (uint8_t*)memory_pointer, size);
      break;
    case element::Type_t::i8:
      m_blob = make_shared<InferenceEngine::TBlob<int8_t>>(
          desc, (int8_t*)memory_pointer, size);
      break;
    case element::Type_t::u16:
      m_blob = make_shared<InferenceEngine::TBlob<uint16_t>>(
          desc, (uint16_t*)memory_pointer, size);
      break;
    case element::Type_t::i16:
      m_blob = make_shared<InferenceEngine::TBlob<int16_t>>(
          desc, (int16_t*)memory_pointer, size);
      break;
    case element::Type_t::i32:
      m_blob = make_shared<InferenceEngine::TBlob<int32_t>>(
          desc, (int32_t*)memory_pointer, size);
      break;
    case element::Type_t::u64:
      m_blob = make_shared<InferenceEngine::TBlob<uint64_t>>(
          desc, (uint64_t*)memory_pointer, size);
      break;
    case element::Type_t::i64:
      m_blob = make_shared<InferenceEngine::TBlob<int64_t>>(
          desc, (int64_t*)memory_pointer, size);
      break;
    case element::Type_t::boolean:
      m_blob = make_shared<InferenceEngine::TBlob<uint8_t>>(
          desc, (uint8_t*)memory_pointer, size);
      break;
    default:
      THROW_IE_EXCEPTION << "Can't create IE blob for type " << element_type
                         << " and shape " << shape_;
  }
}

IETensor::IETensor(const element::Type& element_type, const Shape& shape)
    : IETensor(element_type, shape, nullptr) {}

IETensor::IETensor(const element::Type& element_type, const PartialShape& shape)
    : runtime::Tensor(
          make_shared<descriptor::Tensor>(element_type, shape, "")) {
  throw runtime_error("partial shapes not supported.");
}

void IETensor::write(const void* src, size_t bytes) {
  const int8_t* src_ptr = static_cast<const int8_t*>(src);
  if (src_ptr == nullptr) {
    return;
  }

  auto lm = m_blob->wmap();
  uint8_t* output_ptr = lm.as<uint8_t*>();
  copy(src_ptr, src_ptr + bytes, output_ptr);
}

void IETensor::read(void* dst, size_t bytes) const {
  int8_t* dst_ptr = static_cast<int8_t*>(dst);
  if (dst_ptr == nullptr) {
    return;
  }

  auto lm = m_blob->rmap();
  uint8_t* output_ptr = lm.as<uint8_t*>();
  copy(output_ptr, output_ptr + bytes, dst_ptr);
}

const void* IETensor::get_data_ptr() const {
  auto lm = m_blob->rwmap();
  return lm.as<void*>();
}
}
}