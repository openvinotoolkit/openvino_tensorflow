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

#ifndef NGRAPH_TF_NGRAPHOPTIMIZER_H_
#define NGRAPH_TF_NGRAPHOPTIMIZER_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

#include "logging/ngraph_log.h"
#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/grappler/ngraph_add_identityn.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_deassign_clusters.h"
#include "ngraph_bridge/ngraph_encapsulate_clusters.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_utils.h"

#include <iomanip>

namespace tensorflow {
namespace ngraph_bridge {

// Custom Grappler Optimizer for NGraph-TF
class NgraphOptimizer : public tensorflow::grappler::CustomGraphOptimizer {
 public:
  NgraphOptimizer() = default;
  ~NgraphOptimizer() override = default;

  string name() const override { return "NgraphOptimizer"; };

  bool UsesFunctionLibrary() const override { return true; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  // This is a grappler pass to change a TF graph to nGraph enabled TF graph.
  // It accepts TF nodes that can be processed by nGraph and encapsulates them
  // into NGraphEncapsulateOp
  // To honour fetch (result-bearing) nodes, this pass does one of 2 things
  // (which make it different from the normal non-grappler optimization passes):
  // 1. The grappler pass attaches IdentityN nodes to fetch nodes
  // 2. In case it is not able to attach IdentityN (no outputs or outputs with
  // ref types), it rejects that node for clustering, thus ensuring functional
  // correctness
  Status Optimize(tensorflow::grappler::Cluster*,
                  const tensorflow::grappler::GrapplerItem&,
                  GraphDef*) override;

  void Feedback(tensorflow::grappler::Cluster*,
                const tensorflow::grappler::GrapplerItem&, const GraphDef&,
                double) override;

 private:
  std::unordered_map<std::string, std::string> m_config_map;
  static int FreshIndex();

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

int NgraphOptimizer::s_serial_counter = 0;
mutex NgraphOptimizer::s_serial_counter_mutex;

}  // namespace ngraph_bridge
}  // namespace tensorflow
#endif  // NGRAPH_TF_NGRAPHOPTIMIZER_H_