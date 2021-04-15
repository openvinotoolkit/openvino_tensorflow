/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#pragma once

#ifndef OPENVINO_TF_NGRAPHOPTIMIZER_H_
#define OPENVINO_TF_NGRAPHOPTIMIZER_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/version.h"
#if (TF_MAJOR_VERSION>=2) && (TF_MINOR_VERSION>2)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#else
#include "tensorflow/core/graph/graph_constructor.h"
#endif
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

#include "logging/ovtf_log.h"
#include "logging/tf_graph_writer.h"
#include "openvino_tensorflow/grappler/add_identityn.h"
#include "openvino_tensorflow/assign_clusters.h"
#include "openvino_tensorflow/deassign_clusters.h"
#include "openvino_tensorflow/encapsulate_clusters.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_utils.h"

#include <iomanip>

namespace tensorflow {
namespace openvino_tensorflow {

// Custom Grappler Optimizer for NGraph-TF
class OVTFOptimizer : public tensorflow::grappler::CustomGraphOptimizer {
 public:
  OVTFOptimizer() = default;
  ~OVTFOptimizer() override = default;

  string name() const override { return "OVTFOptimizer"; };

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

int OVTFOptimizer::s_serial_counter = 0;
mutex OVTFOptimizer::s_serial_counter_mutex;

}  // namespace openvino_tensorflow
}  // namespace tensorflow
#endif  // OPENVINO_TF_NGRAPHOPTIMIZER_H_