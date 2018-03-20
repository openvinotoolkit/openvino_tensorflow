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

#include "ngraph_executor.h"
#include <stdlib.h>
#include <string.h>
#include "ngraph_log.h"
#include "ngraph_platform_id.h"
#include "ngraph_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace ngraph_plugin {

se::host::HostStream* AsExecutorStream(se::Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<se::host::HostStream*>(stream->implementation());
}

NGraphExecutor::NGraphExecutor(const se::PluginConfig& plugin_config)
    : plugin_config_(plugin_config) {}

NGraphExecutor::~NGraphExecutor() {}

//---------------------------------------------------------------------------
// NGraphExecutor::Allocate()
//---------------------------------------------------------------------------
void* NGraphExecutor::Allocate(uint64 size) { return new char[size]; }

//---------------------------------------------------------------------------
// NGraphExecutor::AllocateSubBuffer()
//---------------------------------------------------------------------------
void* NGraphExecutor::AllocateSubBuffer(se::DeviceMemoryBase* parent,
                                        uint64 offset_bytes,
                                        uint64 size_bytes) {
  return parent + offset_bytes;
}

//---------------------------------------------------------------------------
// NGraphExecutor::Deallocate()
//---------------------------------------------------------------------------
void NGraphExecutor::Deallocate(se::DeviceMemoryBase* mem) {
  if (!mem->is_sub_buffer()) {
    delete[] static_cast<char*>(mem->opaque());
  }
}

//---------------------------------------------------------------------------
// NGraphExecutor::Memcpy()
//---------------------------------------------------------------------------
bool NGraphExecutor::Memcpy(se::Stream* stream, void* host_dst,
                            const se::DeviceMemoryBase& dev_src, uint64 size) {
  AsExecutorStream(stream)->EnqueueTask([this, host_dst, dev_src, size]() {
    se::port::Status ok = SynchronousMemcpy(host_dst, dev_src, size);
  });
  return true;
}

bool NGraphExecutor::Memcpy(se::Stream* stream, se::DeviceMemoryBase* dev_dst,
                            const void* host_src, uint64 size) {
  AsExecutorStream(stream)->EnqueueTask([this, dev_dst, host_src, size]() {
    se::port::Status ok = SynchronousMemcpy(dev_dst, host_src, size);
  });
  return true;
}

//---------------------------------------------------------------------------
// NGraphExecutor::SynchronousMemcpy()
//---------------------------------------------------------------------------
se::port::Status NGraphExecutor::SynchronousMemcpy(
    se::DeviceMemoryBase* dev_dst, const void* host_src, uint64 size) {
  memcpy(dev_dst->opaque(), host_src, size);
  return se::port::Status::OK();
}

se::port::Status NGraphExecutor::SynchronousMemcpy(
    void* host_dst, const se::DeviceMemoryBase& dev_src, uint64 size) {
  memcpy(host_dst, dev_src.opaque(), size);
  return se::port::Status::OK();
}

//---------------------------------------------------------------------------
// NGraphExecutor::HostCallback()
//---------------------------------------------------------------------------
bool NGraphExecutor::HostCallback(se::Stream* stream,
                                  std::function<void()> callback) {
  AsExecutorStream(stream)->EnqueueTask(callback);
  return true;
}

//---------------------------------------------------------------------------
// NGraphExecutor::CreateStreamDependency()
//---------------------------------------------------------------------------
bool NGraphExecutor::CreateStreamDependency(se::Stream* dependent,
                                            se::Stream* other) {
  AsExecutorStream(dependent)->EnqueueTask(
      [other]() { other->BlockHostUntilDone(); });
  AsExecutorStream(dependent)->BlockUntilDone();
  return true;
}

//---------------------------------------------------------------------------
// NGraphExecutor::StartTimer()
//---------------------------------------------------------------------------
bool NGraphExecutor::StartTimer(se::Stream* stream, se::Timer* timer) {
  dynamic_cast<se::host::HostTimer*>(timer->implementation())->Start(stream);
  return true;
}

//---------------------------------------------------------------------------
// NGraphExecutor::StopTimer()
//---------------------------------------------------------------------------
bool NGraphExecutor::StopTimer(se::Stream* stream, se::Timer* timer) {
  NGRAPH_VLOG(3) << "NGraphExecutor::StopTimer()";
  dynamic_cast<se::host::HostTimer*>(timer->implementation())->Stop(stream);
  return true;
}

//---------------------------------------------------------------------------
// NGraphExecutor::BlockHostUntilDone()
//---------------------------------------------------------------------------
se::port::Status NGraphExecutor::BlockHostUntilDone(se::Stream* stream) {
  AsExecutorStream(stream)->BlockUntilDone();
  return se::port::Status::OK();
}

//---------------------------------------------------------------------------
// NGraphExecutor::PopulateDeviceDescription()
//---------------------------------------------------------------------------
se::DeviceDescription* NGraphExecutor::PopulateDeviceDescription() const {
  se::internal::DeviceDescriptionBuilder builder;
  builder.set_device_address_bits(64);
  builder.set_name("NGraph");
  builder.set_device_vendor("VectorName");
  builder.set_platform_version("1.0");
  builder.set_driver_version("1.0");
  builder.set_runtime_version("1.0");
  builder.set_pci_bus_id("1");
  builder.set_device_memory_size(static_cast<uint64>(4) * 1024 * 1024 * 1024);
  builder.set_clock_rate_ghz(static_cast<float>(CLOCKS_PER_SEC) / 1e9);

  return builder.Build().release();
}

}  // namespace ngraph_plugin
}  // namespace xla
