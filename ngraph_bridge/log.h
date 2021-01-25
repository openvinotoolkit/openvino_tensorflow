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

#pragma once

#include <sstream>
#include <string>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace ngraph_bridge {

class LogMessage : public tensorflow::internal::LogMessage {
 public:
  static tensorflow::int64 MinNGraphVLogLevel() {
    const char* vlog_level = std::getenv("NGRAPH_TF_VLOG_LEVEL");
    if (vlog_level == nullptr) {
      return 0;
    }

    // Ideally we would use env_var / safe_strto64, but it is
    // hard to use here without pulling in a lot of dependencies,
    // so we use std:istringstream instead
    std::string min_log_level(vlog_level);
    std::istringstream ss(min_log_level);
    tensorflow::int64 level;
    if (!(ss >> level)) {
      // Invalid vlog level setting, set level to default (0)
      level = 0;
    }
    return level;
  }
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#define NGRAPH_VLOG_IS_ON(lvl) ((lvl) <= LogMessage::MinNGraphVLogLevel())

#define NGRAPH_VLOG(lvl)      \
  if (NGRAPH_VLOG_IS_ON(lvl)) \
  ::tensorflow::internal::LogMessage(__FILE__, __LINE__, tensorflow::INFO)
