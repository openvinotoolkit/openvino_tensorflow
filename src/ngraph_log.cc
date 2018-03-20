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

#include "ngraph_log.h"
#include <cstdlib>
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace tensorflow {
namespace internal {
namespace {
// Parse log level (int64) from environment variable (char*)
int64 LogLevelStrToInt(const char* tf_env_var_val) {
  if (tf_env_var_val == nullptr) {
    return 0;
  }

  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  string min_log_level(tf_env_var_val);
  std::istringstream ss(min_log_level);
  int64 level;
  if (!(ss >> level)) {
    // Invalid vlog level setting, set level to default (0)
    level = 0;
  }

  return level;
}
}  // namespace

int64 NGraphLogMessage::MinNGraphVLogLevel() {
  const char* tf_env_var_val = std::getenv("NGRAPH_VLOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
}

}  // namespace internal
}  // namespace tensorflow

// Print nGraph VLOG level
static bool InitModule() {
  NGRAPH_VLOG(0) << "NGRAPH_VLOG(0) is enabled";
  NGRAPH_VLOG(1) << "NGRAPH_VLOG(1) is enabled";
  NGRAPH_VLOG(2) << "NGRAPH_VLOG(2) is enabled";
  NGRAPH_VLOG(3) << "NGRAPH_VLOG(3) is enabled";
  return true;
}

static bool module_initialized = InitModule();
