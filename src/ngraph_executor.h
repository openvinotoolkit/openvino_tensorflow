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

// Declares the NGraphExecutor class, which is a CPU-only implementation of
// the StreamExecutor interface. For now, this is used for testing and to
// examine the performance of host-based StreamExecutor code.
#ifndef TENSORFLOW_COMPILER_NGRAPH_STREAM_EXECUTOR_NGRAPH_EXECUTOR_H_
#define TENSORFLOW_COMPILER_NGRAPH_STREAM_EXECUTOR_NGRAPH_EXECUTOR_H_

#include <list>
#include <mutex>
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace ngraph_plugin {

using Args = tensorflow::gtl::ArraySlice<se::DeviceMemoryBase>;

class NGraphExecutor : public se::internal::StreamExecutorInterface {
 public:
  explicit NGraphExecutor(const se::PluginConfig& plugin_config);
  ~NGraphExecutor() override;

  se::port::Status Init(int device_ordinal,
                        se::DeviceOptions device_options) override {
    return se::port::Status::OK();
  }

  bool GetKernel(const se::MultiKernelLoaderSpec& spec,
                 se::KernelBase* kernel) override {
    return false;
  }
  bool Launch(se::Stream* stream, const se::ThreadDim& thread_dims,
              const se::BlockDim& block_dims, const se::KernelBase& kernel,
              const se::KernelArgsArrayBase& args) override {
    return false;
  }

  void* Allocate(uint64 size) override;
  void* AllocateSubBuffer(se::DeviceMemoryBase* mem, uint64 offset_bytes,
                          uint64 size_bytes) override;
  void Deallocate(se::DeviceMemoryBase* mem) override;

  void* HostMemoryAllocate(uint64 size) override { return new char[size]; }
  void HostMemoryDeallocate(void* mem) override {
    delete[] static_cast<char*>(mem);
  }
  bool HostMemoryRegister(void* mem, uint64 size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }

  bool Memcpy(se::Stream* stream, void* host_dst,
              const se::DeviceMemoryBase& pop_src, uint64 size) override;
  bool Memcpy(se::Stream* stream, se::DeviceMemoryBase* pop_dst,
              const void* host_src, uint64 size) override;
  bool MemcpyDeviceToDevice(se::Stream* stream, se::DeviceMemoryBase* pop_dst,
                            const se::DeviceMemoryBase& host_src,
                            uint64 size) override {
    return false;
  }

  bool MemZero(se::Stream* stream, se::DeviceMemoryBase* location,
               uint64 size) override {
    return false;
  }
  bool Memset(se::Stream* stream, se::DeviceMemoryBase* location, uint8 pattern,
              uint64 size) override {
    return false;
  }
  bool Memset32(se::Stream* stream, se::DeviceMemoryBase* location,
                uint32 pattern, uint64 size) override {
    return false;
  }

  // No "synchronize all activity" implemented for this platform at the moment.
  bool SynchronizeAllActivity() override { return false; }
  bool SynchronousMemZero(se::DeviceMemoryBase* location,
                          uint64 size) override {
    return false;
  }

  bool SynchronousMemSet(se::DeviceMemoryBase* location, int value,
                         uint64 size) override {
    return false;
  }

  se::port::Status SynchronousMemcpy(se::DeviceMemoryBase* pop_dst,
                                     const void* host_src,
                                     uint64 size) override;
  se::port::Status SynchronousMemcpy(void* host_dst,
                                     const se::DeviceMemoryBase& pop_src,
                                     uint64 size) override;
  se::port::Status SynchronousMemcpyDeviceToDevice(
      se::DeviceMemoryBase* pop_dst, const se::DeviceMemoryBase& pop_src,
      uint64 size) override {
    return se::port::Status{se::port::error::UNIMPLEMENTED, ""};
  }

  bool HostCallback(se::Stream* stream,
                    std::function<void()> callback) override;

  se::port::Status AllocateEvent(se::Event* event) override {
    return se::port::Status{se::port::error::UNIMPLEMENTED, ""};
  }

  se::port::Status DeallocateEvent(se::Event* event) override {
    return se::port::Status{se::port::error::UNIMPLEMENTED, ""};
  }

  se::port::Status RecordEvent(se::Stream* stream, se::Event* event) override {
    return se::port::Status{se::port::error::UNIMPLEMENTED, ""};
  }

  se::port::Status WaitForEvent(se::Stream* stream, se::Event* event) override {
    return se::port::Status{se::port::error::UNIMPLEMENTED, ""};
  }

  se::Event::Status PollForEventStatus(se::Event* event) override {
    return se::Event::Status::kError;
  }

  bool AllocateStream(se::Stream* stream) override { return true; }
  void DeallocateStream(se::Stream* stream) override {}
  bool CreateStreamDependency(se::Stream* dependent,
                              se::Stream* other) override;

  bool AllocateTimer(se::Timer* timer) override { return true; }
  void DeallocateTimer(se::Timer* timer) override {}
  bool StartTimer(se::Stream* stream, se::Timer* timer) override;
  bool StopTimer(se::Stream* stream, se::Timer* timer) override;

  se::port::Status BlockHostUntilDone(se::Stream* stream) override;

  int PlatformDeviceCount() override { return 1; }

  bool DeviceMemoryUsage(int64* free, int64* total) const override {
    return false;
  }

  se::DeviceDescription* PopulateDeviceDescription() const override;

  se::port::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    return se::port::Status::OK();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    return true;
  }

  se::SharedMemoryConfig GetDeviceSharedMemoryConfig() override {
    return se::SharedMemoryConfig::kDefault;
  }

  se::port::Status SetDeviceSharedMemoryConfig(
      se::SharedMemoryConfig config) override {
    return se::port::Status{se::port::error::UNIMPLEMENTED,
                            "Shared memory not supported"};
  }

  std::unique_ptr<se::internal::EventInterface> CreateEventImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<se::internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<se::internal::StreamInterface> GetStreamImplementation()
      override {
    return std::unique_ptr<se::internal::StreamInterface>(
        new se::host::HostStream());
  }

  std::unique_ptr<se::internal::TimerInterface> GetTimerImplementation()
      override {
    return std::unique_ptr<se::internal::TimerInterface>(
        new se::host::HostTimer());
  }

  se::port::StatusOr<se::DeviceMemoryBase> ExecuteGraph(const xla::Shape& shape,
                                                        Args args);

 private:
  se::DeviceMemoryBase AllocateSingleOutput(const xla::Shape& shape);

  se::port::StatusOr<se::DeviceMemoryBase> AllocateOutputBuffer(
      const xla::Shape& shape);

  const se::PluginConfig plugin_config_;
};

}  // namespace ngraph_plugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_NGRAPH_STREAM_EXECUTOR_NGRAPH_EXECUTOR_H_
