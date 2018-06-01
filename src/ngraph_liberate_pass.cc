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

// At graph construction time, TensorFlow likes to place colocation constraints
// that force variables onto the same device as their initializers. For nGraph
// this doesn't work very well, because we don't yet support RNG ops, and this
// results in randomly-initialized variables being forced onto the host.
//
// The workaround implemented here is to "liberate" nGraph-placed ops from
// colocation constraints. This pass only applies to nodes with a requested
// placement on NGRAPH_CPU, meaning that the graph will be unchanged except
// where the user has explicitly requested nGraph.

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph_utils.h"

using namespace std;
namespace ngraph_bridge {

// TODO(amprocte): this decl should probably be in a header.
extern const char* const DEVICE_NGRAPH_CPU;

class NGraphLiberatePass : public tensorflow::GraphOptimizationPass {
 public:
  tf::Status Run(const tf::GraphOptimizationPassOptions& options) {
    return LiberateNGraphPlacement(options.graph->get());
  }

 private:
  static bool IsNGraphNode(const tf::Node* node) {
    tf::DeviceNameUtils::ParsedName parsed;

    if (!tf::DeviceNameUtils::ParseFullName(node->requested_device(),
                                            &parsed)) {
      return false;
    }

    return (parsed.has_type && parsed.type == DEVICE_NGRAPH_CPU);
  }

  tf::Status LiberateNGraphPlacement(tf::Graph* graph) {
    int i = 0;

    for (auto node : graph->op_nodes()) {
      if (node->IsOp() && IsNGraphNode(node)) {
        std::vector<std::string> colo;

        if (tf::GetNodeAttr(node->attrs(), tf::kColocationAttrName, &colo) == tf::Status::OK()) {
          // For every colocation constraint, add some garbage
          // ("LIBERATED_XXX") onto the end of the colocation group name. Note
          // that blanking it out doesn't work, because the placer treats the
          // absence of constraints as implicitly placing the node into an
          // eponymous constraint group, which in many cases will happen to
          // coincide with what the name the graph constructors would have
          // given to the variable group anyway.
          for (auto& s : colo) {
            std::stringstream ss;
            ss << s << "/LIBERATED_" << (i++);
            s = ss.str();
          }

          node->ClearAttr(tf::kColocationAttrName);
          node->AddAttr(tf::kColocationAttrName, colo);
        }
      }
    }

    return tf::Status::OK();
  }
};
}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 80,
                      ngraph_bridge::NGraphLiberatePass);
}  // namespace tensorflow
