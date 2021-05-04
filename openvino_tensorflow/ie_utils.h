/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

// The backend manager class is a singelton class that interfaces with the
// bridge to provide necessary backend

#ifndef IE_UTILS_H_
#define IE_UTILS_H_

#include <atomic>
#include <mutex>
#include <ostream>
#include <vector>

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"

using namespace ngraph;

class IE_Utils {
 public:
  // Returns the maxiumum number of requests based on the device.
  // TODO: The number of requests are hardcoded temporarly.
  // This should dynamically look at the underlying architecture
  // and compute the best performing number of requests.
  static size_t GetMaxReq(std::string device) {
    int max_req = 1;
    if (device == "HDDL") max_req = 8;
    return max_req;
  }

  // Computes the input batch size per request best on the actual input batch
  // size and the device.
  static size_t GetInputBatchSize(size_t inputBatchSize, std::string device) {
    int max_req = IE_Utils::GetMaxReq(device);
    return ((inputBatchSize + max_req - 1) / max_req);
  }

  // Gets the actual number of requests
  static size_t GetNumRequests(size_t inputBatchSize, std::string device) {
    return inputBatchSize / GetInputBatchSize(inputBatchSize, device);
  }

  static bool VPUConfigEnabled() { return true; }

  static bool VPUFastCompileEnabled() { return true; }

  // Creates a MemoryBlob for InferenceEngine
  static void CreateBlob(InferenceEngine::TensorDesc& desc,
                         InferenceEngine::Precision& precision,
                         const void* data_ptr, size_t byte_size,
                         InferenceEngine::MemoryBlob::Ptr& blob_ptr) {
#define MAKE_IE_BLOB(type_, desc_, ptr_, size_)                         \
  do {                                                                  \
    if (ptr_ == nullptr) {                                              \
      blob_ptr = std::make_shared<InferenceEngine::TBlob<type_>>(desc); \
      blob_ptr->allocate();                                             \
    } else {                                                            \
      blob_ptr = std::make_shared<InferenceEngine::TBlob<type_>>(       \
          desc, (type_*)ptr_, size_);                                   \
    }                                                                   \
  } while (0)
    switch (precision) {
      case InferenceEngine::Precision::FP32:
        MAKE_IE_BLOB(float, desc, (float*)data_ptr, byte_size);
        break;
      case InferenceEngine::Precision::U8:
        MAKE_IE_BLOB(uint8_t, desc, (uint8_t*)data_ptr, byte_size);
        break;
      case InferenceEngine::Precision::I8:
        MAKE_IE_BLOB(int8_t, desc, (int8_t*)data_ptr, byte_size);
        break;
      case InferenceEngine::Precision::U16:
        MAKE_IE_BLOB(uint16_t, desc, (uint16_t*)data_ptr, byte_size);
        break;
      case InferenceEngine::Precision::I16:
        MAKE_IE_BLOB(int16_t, desc, (int16_t*)data_ptr, byte_size);
        break;
      case InferenceEngine::Precision::I32:
        MAKE_IE_BLOB(int32_t, desc, (int32_t*)data_ptr, byte_size);
        break;
      case InferenceEngine::Precision::U64:
        MAKE_IE_BLOB(uint64_t, desc, (uint64_t*)data_ptr, byte_size);
        break;
      case InferenceEngine::Precision::I64:
        MAKE_IE_BLOB(int64_t, desc, (int64_t*)data_ptr, byte_size);
        break;
      case InferenceEngine::Precision::BOOL:
        MAKE_IE_BLOB(uint8_t, desc, (uint8_t*)data_ptr, byte_size);
        break;
      default:
        THROW_IE_EXCEPTION << "Can't create IE blob for type "
                           << precision.name();
    }
  }

  static InferenceEngine::Precision toPrecision(
      const element::Type& element_type) {
    switch (element_type) {
      case element::Type_t::f16:
        return InferenceEngine::Precision::FP16;
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
  static const element::Type fromPrecision(
      const InferenceEngine::Precision precision) {
    switch (precision) {
      case InferenceEngine::Precision::FP32:
        return element::Type_t::f32;
      case InferenceEngine::Precision::U8:
        return element::Type_t::u8;
      case InferenceEngine::Precision::I8:
        return element::Type_t::i8;
      case InferenceEngine::Precision::U16:
        return element::Type_t::u16;
      case InferenceEngine::Precision::I16:
        return element::Type_t::i16;
      case InferenceEngine::Precision::I32:
        return element::Type_t::i32;
      case InferenceEngine::Precision::U64:
        return element::Type_t::u64;
      case InferenceEngine::Precision::I64:
        return element::Type_t::i64;
      case InferenceEngine::Precision::BOOL:
        return element::Type_t::boolean;
      default:
        THROW_IE_EXCEPTION << "Can't convert IE precision " << precision
                           << " to nGraph type!";
    }
  }
};

#endif
// IE_UTILS_H
