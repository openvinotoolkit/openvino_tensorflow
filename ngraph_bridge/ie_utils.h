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

// The backend manager class is a singelton class that interfaces with the
// bridge to provide necessary backend

#ifndef IE_UTILS_H_
#define IE_UTILS_H_

#include <atomic>
#include <mutex>
#include <ostream>
#include <vector>

#include <ie_core.hpp>

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
};

#endif
// IE_UTILS_H
