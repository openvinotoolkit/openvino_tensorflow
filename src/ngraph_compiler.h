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

#ifndef TENSORFLOW_COMPILER_NGRAPH_COMPILER_H_
#define TENSORFLOW_COMPILER_NGRAPH_COMPILER_H_

#include <memory>
#include <mutex>
#include "ngraph/runtime/backend.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph_emitter.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace ngraph_plugin {

class NGraphCompiler {
 public:
  NGraphCompiler();
  ~NGraphCompiler() {}

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module,
      perftools::gputools::StreamExecutor* executor,
      DeviceMemoryAllocator* device_allocator);

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> hlo_module,
      perftools::gputools::StreamExecutor* stream_exec,
      DeviceMemoryAllocator* device_allocator);

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const;

  se::Platform::Id PlatformId() const;

 private:
  static std::shared_ptr<ngraph::runtime::Backend> m_ngraph_backend;

  std::mutex m_module_mutex;
  NGraphEmitter::FusedOpMap m_fusion_map;

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphCompiler);
};

}  // namespace ngraph_plugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_NGRAPH_COMPILER_H_
