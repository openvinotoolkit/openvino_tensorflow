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

#include "ngraph_platform.h"
#include "ngraph_executor.h"
#include "ngraph_log.h"
#include "ngraph_platform_id.h"
#include "ngraph_utils.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace xla {
namespace ngraph_plugin {

NGraphPlatform::NGraphPlatform() : name_("NGraph") {}

NGraphPlatform::~NGraphPlatform() {}

se::Platform::Id NGraphPlatform::id() const { return kNGraphPlatformId; }

int NGraphPlatform::VisibleDeviceCount() const { return 1; }

const string& NGraphPlatform::Name() const { return name_; }

se::port::StatusOr<se::StreamExecutor*> NGraphPlatform::ExecutorForDevice(
    int ordinal) {
  NGRAPH_VLOG(3) << "NGraphPlatform::ExecutorForDevice()";

  se::StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = se::PluginConfig();
  config.device_options = se::DeviceOptions::Default();
  return GetExecutor(config);
}

se::port::StatusOr<se::StreamExecutor*>
NGraphPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const se::PluginConfig& plugin_config) {
  NGRAPH_VLOG(3) << "NGraphPlatform::ExecutorForDeviceWithPluginConfig()";
  se::StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = se::DeviceOptions::Default();
  return GetExecutor(config);
}

se::port::StatusOr<se::StreamExecutor*> NGraphPlatform::GetExecutor(
    const se::StreamExecutorConfig& config) {
  NGRAPH_VLOG(3) << "NGraphPlatform::GetExecutor()";

  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

se::port::StatusOr<std::unique_ptr<se::StreamExecutor>>
NGraphPlatform::GetUncachedExecutor(const se::StreamExecutorConfig& config) {
  NGRAPH_VLOG(3) << "NGraphPlatform::GetUncachedExecutor()";

  auto executor = se::port::MakeUnique<se::StreamExecutor>(
      this, se::port::MakeUnique<NGraphExecutor>(config.plugin_config));
  auto init_status = executor->Init(config.ordinal, config.device_options);
  if (!init_status.ok()) {
    return se::port::Status{
        se::port::error::INTERNAL,
        se::port::Printf(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString().c_str())};
  }

  return std::move(executor);
}

void NGraphPlatform::RegisterTraceListener(
    std::unique_ptr<se::TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register executor trace listener";
}

void NGraphPlatform::UnregisterTraceListener(se::TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister executor trace listener";
}

static void InitializeNGraphPlatform() {
  NGRAPH_VLOG(3) << "InitializeNGraphPlatform()";
  std::unique_ptr<se::Platform> platform(new NGraphPlatform);
  SE_CHECK_OK(se::MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace ngraph_plugin
}  // namespace xla

REGISTER_MODULE_INITIALIZER(ngraph_platform,
                            xla::ngraph_plugin::InitializeNGraphPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);
// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(ngraph_platform, multi_platform_manager);
