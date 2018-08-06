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
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/util/device_name_utils.h"

#include "ngraph_mark_for_hostmem_pass.h"

namespace tf = tensorflow;

using namespace std;
namespace ngraph_bridge {

extern const char* const DEVICE_NGRAPH;

// TODO(amprocte): do we need to look at job name, replica, task?
// TODO(amprocte): need to quit duplicating this function >_<
static bool IsNGraphNode(const tf::Node* node) {
  tf::DeviceNameUtils::ParsedName parsed;

  if (!tf::DeviceNameUtils::ParseFullName(node->assigned_device_name(),
                                          &parsed)) {
    return false;
  }

  return (parsed.has_type && parsed.type == DEVICE_NGRAPH);
}

tf::Status NGraphMarkForHostmemPass::Run(
    const tf::GraphOptimizationPassOptions& options) {
  tf::Graph* graph = options.graph->get();

  for (auto node : graph->op_nodes()) {
    if (IsNGraphNode(node)) {
      std::vector<tf::int32> input_indices;
      for (tf::int32 i = 0; i < node->num_inputs(); i++) {
        input_indices.push_back(i);
      }
      node->AddAttr("_input_hostmem",input_indices);

      std::vector<tf::int32> output_indices;
      for (tf::int32 i = 0; i < node->num_outputs(); i++) {
        output_indices.push_back(i);
      }
      node->AddAttr("_output_hostmem",output_indices);
    }
  }

  return tf::Status::OK();
}

}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 115,
                      ngraph_bridge::NGraphMarkForHostmemPass);
}  // namespace tensorflow
