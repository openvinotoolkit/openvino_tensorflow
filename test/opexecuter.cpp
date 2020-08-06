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
#include "test/opexecuter.h"
#include <cstdlib>

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

// Validate that the graph has N allowed_nodes and 1 test_op_type node
// Graph must look like this
//
// Const1     ConstN
//   \    ...    /
//    \         /
//      Test_Op
//
// TODO check for vector allowed_nodes
// when we allow other than "Const" node type as input
// Make allowed_nodes const member of the class, use set
void OpExecuter::ValidateGraph(const Graph& graph,
                               const vector<string> allowed_nodes) {
  NGRAPH_VLOG(5) << "Validate graph";
  bool found_test_op = false;
  Node* test_op = nullptr;
  for (Node* node : graph.nodes()) {
    if (node->IsSource() || node->IsSink()) {
      continue;
    } else if (node->type_string() == test_op_type_) {
      // only one node of type test_op
      ASSERT_FALSE(found_test_op) << "Only one op of type " << test_op_type_
                                  << " should exist in the graph. Found nodes "
                                  << node->name() << " and " << test_op->name();
      found_test_op = true;
      test_op = node;
    } else {
      ASSERT_TRUE(node->type_string() == allowed_nodes[0])
          << "Op of type " << node->type_string()
          << " not allowed in the graph. Found " << node->name();
    }
  }
  ASSERT_TRUE(found_test_op) << "Not found test_op : " << test_op_type_;
  NGRAPH_VLOG(5) << "Validate graph done";
}  // namespace testing

OpExecuter::OpExecuter(const Scope sc, const string test_op,
                       const vector<Output>& sess_run_fetchops)
    : tf_scope_(sc),
      test_op_type_(test_op),
      sess_run_fetchoutputs_(sess_run_fetchops) {}

OpExecuter::~OpExecuter() {}

void OpExecuter::RunTest(float rtol, float atol) {
  vector<Tensor> ngraph_outputs;
  ExecuteOnNGraph(ngraph_outputs);
  vector<Tensor> tf_outputs;
  ExecuteOnTF(tf_outputs);

  // Override the test result tolerance
  if (std::getenv("NGRAPH_TF_UTEST_RTOL") != nullptr) {
    rtol = std::atof(std::getenv("NGRAPH_TF_UTEST_RTOL"));
  }

  if (std::getenv("NGRAPH_TF_UTEST_ATOL") != nullptr) {
    atol = std::atof(std::getenv("NGRAPH_TF_UTEST_ATOL"));
  }

  Compare(tf_outputs, ngraph_outputs, rtol, atol);
}

// Uses tf_scope to execute on TF
void OpExecuter::ExecuteOnTF(vector<Tensor>& tf_outputs) {
  // Deactivate nGraph to be able to run on TF
  DeactivateNGraph();
  ClientSession session(tf_scope_);
  ASSERT_EQ(Status::OK(), session.Run(sess_run_fetchoutputs_, &tf_outputs))
      << "Failed to run opexecutor on TF";
  for (size_t i = 0; i < tf_outputs.size(); i++) {
    NGRAPH_VLOG(5) << " TF op " << i << " " << tf_outputs[i].DebugString();
  }
  // Activate nGraph again
  ActivateNGraph();
}

// Sets NG backend before executing on NGTF
void OpExecuter::ExecuteOnNGraph(vector<Tensor>& ngraph_outputs) {
  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(tf_scope_.ToGraph(&graph));

  // For debug
  if (std::getenv("NGRAPH_TF_DUMP_GRAPHS") != nullptr) {
    GraphToPbTextFile(&graph, "unit_test_tf_graph_" + test_op_type_ + ".pbtxt");
  }

  ValidateGraph(graph, {"Const"});

  ActivateNGraph();
  string backend_name;
  TF_CHECK_OK(BackendManager::GetCurrentlySetBackendName(&backend_name));
  tf::SessionOptions options = GetSessionOptions(backend_name);
  ClientSession session(tf_scope_, options);
  TF_CHECK_OK(session.Run(sess_run_fetchoutputs_, &ngraph_outputs))
      << "Failed to run opexecutor on NGTF";
  for (size_t i = 0; i < ngraph_outputs.size(); i++) {
    NGRAPH_VLOG(5) << " NGTF op " << i << " "
                   << ngraph_outputs[i].DebugString();
  }
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
