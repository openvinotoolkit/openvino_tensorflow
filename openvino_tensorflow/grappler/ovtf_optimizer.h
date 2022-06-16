/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#pragma once

#ifndef OPENVINO_GRAPPLER_OPTIMIZER_H_
#define OPENVINO_GRAPPLER_OPTIMIZER_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/version.h"
#if (TF_MAJOR_VERSION >= 2) && (TF_MINOR_VERSION > 2)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#else
#include "tensorflow/core/graph/graph_constructor.h"
#endif
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

#include "logging/ovtf_log.h"
#include "logging/tf_graph_writer.h"
#include "openvino_tensorflow/assign_clusters.h"
#include "openvino_tensorflow/deassign_clusters.h"
#include "openvino_tensorflow/encapsulate_clusters.h"
#include "openvino_tensorflow/grappler/add_identityn.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_utils.h"

#include <iomanip>

namespace tensorflow {
namespace openvino_tensorflow {

// Custom Grappler Optimizer for OpenVINO-TF
class OpenVINOGrapplerOptimizer : public tensorflow::grappler::CustomGraphOptimizer {
 public:
  OpenVINOGrapplerOptimizer() = default;
  ~OpenVINOGrapplerOptimizer() override = default;

  string name() const override { return "OpenVINOGrapplerOptimizer"; };

  bool UsesFunctionLibrary() const override { return true; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  // This is a grappler pass to change a TF graph to OpenVINO enabled TF graph.
  // It accepts TF nodes that can be processed by OpenVINO and encapsulates them
  // into OpenVINOEncapsulateOp
  // To honour fetch (result-bearing) nodes, this pass does one of 2 things
  // (which make it different from the normal non-grappler optimization passes):
  // 1. The grappler pass attaches IdentityN nodes to fetch nodes
  // 2. In case it is not able to attach IdentityN (no outputs or outputs with
  // ref types), it rejects that node for clustering, thus ensuring functional
  // correctness
  Status Optimize(tensorflow::grappler::Cluster*,
                  const tensorflow::grappler::GrapplerItem&,
                  GraphDef*) override;

 private:
  std::unordered_map<std::string, std::string> m_config_map;
  static int FreshIndex();

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

int OpenVINOGrapplerOptimizer::s_serial_counter = 0;
mutex OpenVINOGrapplerOptimizer::s_serial_counter_mutex;

}  // namespace openvino_tensorflow
}  // namespace tensorflow
#endif  // OPENVINO_GRAPPLER_OPTIMIZER_H_