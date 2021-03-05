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

#include <cstdlib>

#include "logging/tf_graph_writer.h"
#include "openvino_tensorflow/ovtf_builder.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_utils.h"
#include "test/opexecuter.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace openvino_tensorflow {
namespace testing {

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
  const char* rtol_char_ptr = std::getenv("OPENVINO_TF_UTEST_RTOL");
  if (rtol_char_ptr!= nullptr) {
    rtol = std::atof(rtol_char_ptr);
  }

  const char* atol_char_ptr = std::getenv("OPENVINO_TF_UTEST_ATOL");
  if (atol_char_ptr != nullptr) {
    atol = std::atof(atol_char_ptr);
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
  if (std::getenv("OPENVINO_TF_DUMP_GRAPHS") != nullptr) {
    GraphToPbTextFile(&graph, "unit_test_tf_graph_" + test_op_type_ + ".pbtxt");
  }

  ActivateNGraph();
  tf::SessionOptions options = GetSessionOptions();
  ClientSession session(tf_scope_, options);
  try {
    ASSERT_EQ(Status::OK(),
              session.Run(sess_run_fetchoutputs_, &ngraph_outputs));
  } catch (const std::exception& e) {
    NGRAPH_VLOG(0) << "Exception occured while running session " << e.what();
    EXPECT_TRUE(false);
  }
  for (size_t i = 0; i < ngraph_outputs.size(); i++) {
    NGRAPH_VLOG(5) << " NGTF op " << i << " "
                   << ngraph_outputs[i].DebugString();
  }
}

}  // namespace testing
}  // namespace openvino_tensorflow
}  // namespace tensorflow
