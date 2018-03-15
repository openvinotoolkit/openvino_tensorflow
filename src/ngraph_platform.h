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

#ifndef TENSORFLOW_COMPILER_NGRAPH_STREAM_EXECUTOR_NGRAPH_PLATFORM_H_
#define TENSORFLOW_COMPILER_NGRAPH_STREAM_EXECUTOR_NGRAPH_PLATFORM_H_

#include <memory>
#include <string>
#include <vector>
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/trace_listener.h"

using namespace std;
namespace se = ::perftools::gputools;

namespace xla {
namespace ngraph_plugin {

class NGraphPlatform : public se::Platform {
 public:
  NGraphPlatform();
  ~NGraphPlatform() override;

  Platform::Id id() const override;

  // Device count is less clear-cut for CPUs than accelerators. This call
  // currently returns the number of thread units in the host, as reported by
  // base::NumCPUs().
  int VisibleDeviceCount() const override;

  const string& Name() const override;

  se::port::StatusOr<se::StreamExecutor*> ExecutorForDevice(
      int ordinal) override;

  se::port::StatusOr<se::StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const se::PluginConfig& config) override;

  se::port::StatusOr<se::StreamExecutor*> GetExecutor(
      const se::StreamExecutorConfig& config) override;

  se::port::StatusOr<std::unique_ptr<se::StreamExecutor>> GetUncachedExecutor(
      const se::StreamExecutorConfig& config) override;

  void RegisterTraceListener(
      std::unique_ptr<se::TraceListener> listener) override;

  void UnregisterTraceListener(se::TraceListener* listener) override;

 private:
  // This platform's name.
  string name_;

  // mutex that guards the ordinal-to-executor map.
  mutable se::mutex executors_mutex_;

  // Cache of created StreamExecutors.
  se::ExecutorCache executor_cache_;

  SE_DISALLOW_COPY_AND_ASSIGN(NGraphPlatform);
};

}  // namespace ngraph_plugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_NGRAPH_STREAM_EXECUTOR_NGRAPH_PLATFORM_H_
