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
#ifndef NGRAPH_TF_BRIDGE_OPEXECUTER_H_
#define NGRAPH_TF_BRIDGE_OPEXECUTER_H_

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "ngraph/ngraph.hpp"

#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

class OpExecuter {
 public:
  using NodeMetaData = map<Node*, vector<std::pair<Node*, int>>>;
  using NodeOutEdges = map<Node*, vector<const Edge*>>;

  // Scope sc                                : TF Scope with execution graph
  // string test_op                          : test_op_type e.g. "Add"
  // const vector<int>& static_input_indexes : input indices for test_op that
  //                                           are static for nGraph
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
  // Translates the tf graph to nGraph
  // Returns outputs as tf Tensors
  void ExecuteOnNGraph(vector<Tensor>& outputs);

  // Creates tf Session from tf Scope
  // Executes on TF
  // Returns outputs
  void ExecuteOnTF(vector<Tensor>& outputs);

  // Executes on NGraph backend, then executes on TF, and compares the results
  void RunTest(float rtol = static_cast<float>(1e-05),
               float atol = static_cast<float>(1e-08));

 private:
  Scope tf_scope_;
  const string test_op_type_;
  const std::vector<Output> sess_run_fetchoutputs_;
};

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_OPEXECUTER_H_
