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

#ifndef TENSORFLOW_COMPILER_NGRAPH_DRIVER_NGRAPH_LOG_H_
#define TENSORFLOW_COMPILER_NGRAPH_DRIVER_NGRAPH_LOG_H_

#include <dlfcn.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include "ngraph/ngraph.hpp"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace internal {

class NGraphLogMessage : public LogMessage {
 public:
  static int64 MinNGraphVLogLevel();
};

#define NGRAPH_VLOG_IS_ON(lvl) \
  ((lvl) <= ::tensorflow::internal::NGraphLogMessage::MinNGraphVLogLevel())

#define NGRAPH_VLOG(lvl)      \
  if (NGRAPH_VLOG_IS_ON(lvl)) \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::INFO)

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_NGRAPH_DRIVER_NGRAPH_LOG_H_
