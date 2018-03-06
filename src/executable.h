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

#ifndef XLA_EXAMPLE_PLUGIN_EXECUTABLE_H_
#define XLA_EXAMPLE_PLUGIN_EXECUTABLE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"

#include "tensorflow/compiler/xla/service/transfer_manager_interface.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

class PluginExecutable : public xla::Executable {
 public:
  PluginExecutable(std::unique_ptr<xla::HloModule> hlo_module,
                   xla::TransferManagerInterface* transfer_manager)
      : xla::Executable(std::move(hlo_module), /*hlo_profile_printer=*/nullptr,
                        /*hlo_profile_index_map=*/nullptr),
        m_transfer_manager(transfer_manager) {}
  ~PluginExecutable() {}

  xla::StatusOr<std::unique_ptr<xla::ShapedBuffer>> ExecuteOnStream(
      const xla::ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const xla::ShapedBuffer*> arguments,
      xla::HloExecutionProfile* hlo_execution_profile) override;

  xla::StatusOr<std::unique_ptr<xla::ShapedBuffer>> ExecuteAsyncOnStream(
      const xla::ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const xla::ShapedBuffer*> arguments)
      override {
    return tensorflow::errors::Unimplemented(
        "ExecuteAsyncOnStream is not yet supported on Executor.");
  }

  static tensorflow::int64 ShapeSizeBytes(const xla::Shape& shape) {
    return xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }

 private:
  xla::TransferManagerInterface* m_transfer_manager;
  TF_DISALLOW_COPY_AND_ASSIGN(PluginExecutable);
};

#endif  // TENSORFLOW_COMPILER_EXECUTOR_DRIVER_EXECUTOR_EXECUTABLE_H_
