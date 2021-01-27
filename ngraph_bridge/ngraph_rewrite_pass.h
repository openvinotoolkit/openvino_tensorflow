/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
namespace ngraph_bridge {

class NGraphRewritePass : public GraphOptimizationPass {
 public:
  NGraphRewritePass() = default;
  ~NGraphRewritePass() override = default;

  Status Rewrite(Graph* graph, std::set<string> skip_these_nodes = {},
                 std::unordered_map<std::string, std::string> = {});
  Status Run(const GraphOptimizationPassOptions& options);

 private:
  // Returns a fresh "serial number" to avoid filename collisions in the graph
  // dumps.
  static int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

}  // namespace ngraph_bridge
}  // namespace tensorflow
