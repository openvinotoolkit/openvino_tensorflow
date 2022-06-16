/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#ifndef OPENVINO_TF_BRIDGE_OPEXECUTER_H_
#define OPENVINO_TF_BRIDGE_OPEXECUTER_H_

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/version.h"
#if (TF_MAJOR_VERSION >= 2) && (TF_MINOR_VERSION > 2)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#else
#include "tensorflow/core/graph/graph_constructor.h"
#endif
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "ngraph/ngraph.hpp"

#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace openvino_tensorflow {
namespace testing {

class OpExecuter {
 public:
  using NodeMetaData = map<Node*, vector<std::pair<Node*, int>>>;
  using NodeOutEdges = map<Node*, vector<const Edge*>>;

  // Scope sc                                : TF Scope with execution graph
  // string test_op                          : test_op_type e.g. "Add"
  // const vector<int>& static_input_indexes : input indices for test_op that
  //                                           are static for OpenVINO
  //                                         : e.g. Conv2DBackPropFilter {1}
  // vector<DataType>& op_types              : expected tf data types of the
  //                                            output e.g.
  //                                           {DT_FLOAT, DT_INT}
  // const std::vector<Output>& fetch_ops    : Output ops to be fetched,
  //                                           is passed to tf.session.run()
  OpExecuter(const Scope sc, const string test_op,
             const vector<Output>& fetch_ops);

  ~OpExecuter();

  // Creates the tf graph from tf Scope
  // Translates the tf graph to OpenVINO
  // Returns outputs as tf Tensors
  void ExecuteOnOpenVINO(vector<Tensor>& outputs);

  // Creates tf Session from tf Scope
  // Executes on TF
  // Returns outputs
  void ExecuteOnTF(vector<Tensor>& outputs);

  // Executes on OpenVINO backend, then executes on TF, and compares the results
  void RunTest(float rtol = static_cast<float>(1e-05),
               float atol = static_cast<float>(1e-08));

 private:
  Scope tf_scope_;
  const string test_op_type_;
  const std::vector<Output> sess_run_fetchoutputs_;
};

}  // namespace testing
}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_BRIDGE_OPEXECUTER_H_
