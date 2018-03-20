/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TRANSFER_MANAGER_H_
#define TRANSFER_MANAGER_H_

#include "tensorflow/compiler/xla/service/transfer_manager_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

#include <vector>

namespace se = ::perftools::gputools;

class TransferManager : public xla::TransferManagerInterface {
 public:
  TransferManager();

  ~TransferManager() override {}

  // Returns the ID of the platform that this transfer manager acts on.
  perftools::gputools::Platform::Id PlatformId() const override;

  // Transfers the data held in the given ShapedBuffer into the provided literal
  // using the provided executor. literal_shape will be the shape for the
  // literal. The shape of the ShapedBuffer and DeviceShape(literal_shape) must
  // be compatible, but need not have the same layout.
  xla::StatusOr<std::unique_ptr<xla::Literal>> TransferLiteralFromDevice(
      perftools::gputools::StreamExecutor* executor,
      const xla::ShapedBuffer& device_buffer) override;

  // Transfers the given literal into the previously allocated device memory
  // represented by the given ShapedBuffer using the given executor.
  xla::Status TransferLiteralToDevice(
      perftools::gputools::StreamExecutor* executor,
      const xla::Literal& literal,
      const xla::ShapedBuffer& device_buffer) override;

  // Transfers the given literal into the Infeed interface of the device,
  // using the given executor.
  xla::Status TransferLiteralToInfeed(
      perftools::gputools::StreamExecutor* executor,
      const xla::Literal& literal) override {
    return xla::Unimplemented("TransferBufferToInfeed");
  }

  // Transfer a memory block of the given size from 'source' buffer to the
  // Infeed interface of the device using the given executor.
  //
  // size is the size to transfer from source in bytes.
  //
  // source is the source data that must be in the target-dependent layout that
  // the Infeed HLO used in the computation expects.
  xla::Status TransferBufferToInfeed(
      perftools::gputools::StreamExecutor* executor, tensorflow::int64 size,
      const void* source) override {
    return xla::Unimplemented("TransferBufferToInfeed");
  }

  // Transfers the given literal from the Outfeed interface of the device,
  // using the given executor.
  xla::Status TransferLiteralFromOutfeed(
      perftools::gputools::StreamExecutor* executor,
      const xla::Shape& literal_shape, xla::Literal* literal) override {
    return xla::Unimplemented("TransferLiteralFromOutfeed");
  }

  // Resets the devices associated with this transfer manager.
  xla::Status ResetDevices(
      tensorflow::gtl::ArraySlice<perftools::gputools::StreamExecutor*>
          executor) override {
    return xla::Unimplemented("ResetDevices");
  }

  // Determines the byte size requirement for the given shape on the underlying
  // architecture. This will be used to allocate an appropriately sized memory
  // region for a host-to-device transfer.
  tensorflow::int64 GetByteSizeRequirement(
      const xla::Shape& shape) const override {
    return xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }

  // Writes the given device-memory pointers in 'elements' to the given region
  // to construct a tuple in the platform-specific tuple representation. This
  // can handle nested tuples as well. In the nested case, the element
  // DeviceMemoryBase points to another array of pointers on the device.
  xla::Status WriteSingleTupleIndexTable(
      perftools::gputools::StreamExecutor* executor,
      tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          elements,
      const xla::Shape& shape,
      perftools::gputools::DeviceMemoryBase* region) override;

  xla::StatusOr<std::unique_ptr<xla::ShapedBuffer>> AllocateShapedBuffer(
      const xla::Shape& on_host_shape, xla::DeviceMemoryAllocator* allocator,
      int device_ordinal) override;

  xla::StatusOr<std::unique_ptr<xla::ScopedShapedBuffer>>
  AllocateScopedShapedBuffer(const xla::Shape& on_host_shape,
                             xla::DeviceMemoryAllocator* allocator,
                             int device_ordinal) override {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::ShapedBuffer> unscoped_buffer,
        AllocateShapedBuffer(on_host_shape, allocator, device_ordinal));
    return xla::ScopedShapedBuffer::MakeScoped(unscoped_buffer.get(),
                                               allocator);
  }

 private:
  xla::Shape HostShapeToDeviceShape(const xla::Shape& host_shape) const {
    return host_shape;
  }

  // Transfer a memory block of the given size from 'source' buffer to the given
  // destination of the device.
  //
  // size is the size to transfer from source in bytes.
  xla::Status TransferBufferToDevice(
      perftools::gputools::StreamExecutor* executor, tensorflow::int64 size,
      const void* source, perftools::gputools::DeviceMemoryBase* destination);

  // Transfer a memory block of the given size from the device source into the
  // 'destination' buffer.
  //
  // size is the size to transfer to destination in bytes.
  xla::Status TransferBufferFromDevice(
      perftools::gputools::StreamExecutor* executor,
      const perftools::gputools::DeviceMemoryBase& source,
      tensorflow::int64 size, void* destination);

  // Given an allocated ShapedBuffer, constructs the tuple index table(s) in
  // each buffer of the given ShapedBuffer corresponding to tuple shapes. If the
  // ShapedBuffer is array-shaped this method does nothing.
  xla::Status WriteTupleIndexTables(
      perftools::gputools::StreamExecutor* executor,
      const xla::ShapedBuffer& device_buffer);

  static perftools::gputools::Platform::Id m_PlatformId;
  TF_DISALLOW_COPY_AND_ASSIGN(TransferManager);
};

#endif  // TRANSFER_MANAGER_H_
