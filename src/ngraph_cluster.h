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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace ngraph_bridge {

class NGraphClusterPass : public tensorflow::GraphOptimizationPass {
 public:
  tensorflow::Status Run(
      const tensorflow::GraphOptimizationPassOptions& options);
  static const std::set<std::string>& GetUnclusterableOps() {
    return s_unclusterable_ops;
  }
  static const std::set<std::string>& GetCanBeOutsideClusterOps() {
    return s_can_be_outside_cluster_ops;
  }

 private:
  bool IsNGraphNode(const tensorflow::Node* node);
  bool IsClusterable(const tensorflow::Node* node);
  bool CanBeOutsideCluster(const tensorflow::Node* node);

  struct Cluster {
    int index;
    std::set<tensorflow::Node*> nodes;
  };

  const static std::set<std::string> s_unclusterable_ops;
  const static std::set<std::string> s_can_be_outside_cluster_ops;

  tensorflow::Status IdentifyClusters(tensorflow::Graph* graph);
};

}  // namespace ngraph_bridge
