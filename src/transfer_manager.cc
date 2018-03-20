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

#include "transfer_manager.h"

#include <memory>

#include "tensorflow/compiler/xla/ptr_util.h"

/*static*/ perftools::gputools::Platform::Id TransferManager::m_PlatformId;

TransferManager::TransferManager() {}

perftools::gputools::Platform::Id TransferManager::PlatformId() const {
  return m_PlatformId;
}

xla::Status TransferManager::WriteSingleTupleIndexTable(
    perftools::gputools::StreamExecutor* executor,
    tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase> elements,
    const xla::Shape& shape, perftools::gputools::DeviceMemoryBase* region) {
  TF_RET_CHECK(elements.size() == xla::ShapeUtil::TupleElementCount(shape));

  std::vector<const void*> element_pointers;
  for (const se::DeviceMemoryBase& element : elements) {
    element_pointers.push_back(element.opaque());
  }
  return TransferBufferToDevice(executor, GetByteSizeRequirement(shape),
                                element_pointers.data(), region);
}

xla::StatusOr<std::unique_ptr<xla::Literal>>
TransferManager::TransferLiteralFromDevice(
    perftools::gputools::StreamExecutor* executor,
    const xla::ShapedBuffer& device_buffer) {
  TF_RET_CHECK(executor->device_ordinal() == device_buffer.device_ordinal());

  std::unique_ptr<xla::Literal> literal =
      xla::Literal::CreateFromShape(device_buffer.on_host_shape());

  TF_RETURN_IF_ERROR(xla::ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_host_shape(),
      [&](const xla::Shape& subshape,
          const xla::ShapeIndex& index) -> xla::Status {
        if (!xla::ShapeUtil::IsTuple(subshape)) {
          TF_RETURN_IF_ERROR(TransferBufferFromDevice(
              executor,
              /*source=*/device_buffer.buffer(index),
              /*size=*/GetByteSizeRequirement(subshape),
              /*destination=*/
              literal->untyped_data(index)));
        }

        return xla::Status::OK();
      }));
  return std::move(literal);
}

xla::Status TransferManager::TransferLiteralToDevice(
    perftools::gputools::StreamExecutor* executor, const xla::Literal& literal,
    const xla::ShapedBuffer& device_buffer) {
  // The on-host and on-device shape should always be the same for the generic
  // transfer manager.
  TF_RET_CHECK(xla::ShapeUtil::Equal(device_buffer.on_device_shape(),
                                     device_buffer.on_host_shape()));

  TF_RET_CHECK(xla::ShapeUtil::Compatible(literal.shape(),
                                          device_buffer.on_host_shape()));

  TF_RET_CHECK(executor->device_ordinal() == device_buffer.device_ordinal());
  TF_RETURN_IF_ERROR(WriteTupleIndexTables(executor, device_buffer));

  return xla::ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_host_shape(),
      [&](const xla::Shape& device_subshape,
          const xla::ShapeIndex& index) -> xla::Status {
        se::DeviceMemoryBase device_memory = device_buffer.buffer(index);
        if (xla::ShapeUtil::IsArray(device_subshape)) {
          TF_RET_CHECK(GetByteSizeRequirement(device_subshape) ==
                       device_memory.size());
          // Element is array-shaped: transfer array data to device buffer.
          const auto subliteral = xla::LiteralView::Create(literal, index);
          std::unique_ptr<xla::Literal> relayed_out_literal;
          const void* source;
          if (xla::LayoutUtil::Equal(device_subshape.layout(),
                                     subliteral.shape().layout())) {
            source = subliteral.untyped_data();
          } else {
            // Relayout data before transferring.
            relayed_out_literal = subliteral.Relayout(device_subshape.layout(),
                                                      /*shape_index=*/{});
            source = relayed_out_literal->untyped_data();
          }
          return TransferBufferToDevice(
              executor,
              /*size=*/GetByteSizeRequirement(device_subshape), source,
              &device_memory);
        }
        return xla::Status::OK();
      });
}

xla::Status TransferManager::TransferBufferFromDevice(
    se::StreamExecutor* executor, const se::DeviceMemoryBase& source,
    tensorflow::int64 size, void* destination) {
  if (source.size() < size) {
    return xla::FailedPrecondition(
        "Source allocation on device not large enough for data tranfer: "
        "%lld < %lld",
        source.size(), size);
  }
  auto copy_status = executor->SynchronousMemcpyD2H(source, size, destination);
  if (!copy_status.ok()) {
    return xla::AddStatus(
        xla::Status(static_cast<tensorflow::error::Code>(copy_status.code()),
                    copy_status.error_message()),
        "failed transfer from device to buffer");
  }
  return xla::Status::OK();
}

xla::Status TransferManager::TransferBufferToDevice(
    se::StreamExecutor* executor, tensorflow::int64 size, const void* source,
    se::DeviceMemoryBase* destination) {
  if (destination->size() < size) {
    return xla::FailedPrecondition(
        "Destination allocation on device not large enough for data tranfer: "
        "%lld < %lld",
        destination->size(), size);
  }
  auto copy_status = executor->SynchronousMemcpyH2D(source, size, destination);
  if (!copy_status.ok()) {
    return xla::AddStatus(
        xla::Status(static_cast<tensorflow::error::Code>(copy_status.code()),
                    copy_status.error_message()),
        "failed transfer of buffer to device");
  }
  return xla::Status::OK();
}
xla::Status TransferManager::WriteTupleIndexTables(
    perftools::gputools::StreamExecutor* executor,
    const xla::ShapedBuffer& device_buffer) {
  VLOG(2) << "Writing tuple index tables for " << device_buffer;

  TF_RET_CHECK(executor->device_ordinal() == device_buffer.device_ordinal());

  return xla::ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_device_shape(),
      [&](const xla::Shape& device_subshape,
          const xla::ShapeIndex& index) -> xla::Status {
        if (xla::ShapeUtil::IsTuple(device_subshape)) {
          se::DeviceMemoryBase device_memory = device_buffer.buffer(index);
          TF_RET_CHECK(GetByteSizeRequirement(device_subshape) ==
                       device_memory.size());

          std::vector<se::DeviceMemoryBase> elements;
          xla::ShapeIndex element_index = index;
          for (tensorflow::int64 i = 0;
               i < xla::ShapeUtil::TupleElementCount(device_subshape); ++i) {
            element_index.push_back(i);
            elements.push_back(device_buffer.buffer(element_index));
            element_index.pop_back();
          }
          return WriteSingleTupleIndexTable(executor, elements, device_subshape,
                                            &device_memory);
        }

        return xla::Status::OK();
      });
}

xla::StatusOr<std::unique_ptr<xla::ShapedBuffer>>
TransferManager::AllocateShapedBuffer(const xla::Shape& on_host_shape,
                                      xla::DeviceMemoryAllocator* allocator,
                                      int device_ordinal) {
  if (!xla::LayoutUtil::HasLayout(on_host_shape)) {
    return xla::InvalidArgument(
        "Shape must have a layout: %s",
        xla::ShapeUtil::HumanStringWithLayout(on_host_shape).c_str());
  }
  TF_RETURN_IF_ERROR(xla::ShapeUtil::ValidateShape(on_host_shape));
  const xla::Shape on_device_shape = HostShapeToDeviceShape(on_host_shape);
  TF_RET_CHECK(xla::LayoutUtil::HasLayout(on_device_shape));

  auto shaped_buffer = xla::WrapUnique(new xla::ShapedBuffer(
      on_host_shape, on_device_shape, allocator->platform(), device_ordinal));

  // Allocate an appropriate sized buffer for each element in the shape
  // including the tuple pointer arrays.
  for (auto& pair : shaped_buffer->buffers()) {
    const xla::ShapeIndex& index = pair.first;
    se::DeviceMemoryBase& memory_base = pair.second;
    const xla::Shape& subshape =
        xla::ShapeUtil::GetSubshape(on_device_shape, index);
    TF_ASSIGN_OR_RETURN(memory_base,
                        allocator->Allocate(shaped_buffer->device_ordinal(),
                                            GetByteSizeRequirement(subshape)));
  }

  return std::move(shaped_buffer);
}
