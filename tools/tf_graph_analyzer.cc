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
#include "ngraph_log.h"
#include "tf_graph_writer.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

class NGraphAnalyzerPass : public GraphOptimizationPass {
 public:
  NGraphAnalyzerPass() {}
  virtual ~NGraphAnalyzerPass() {}
  Status Run(const GraphOptimizationPassOptions& options) {
    NGRAPH_VLOG(0) << "nGraph analyzer pass start: ";

    int idx = s_counter++;

    std::stringstream ss;
    ss << "tensorflow_graph_" << idx;
    std::string filename_prefix = ss.str();

    if (options.graph != nullptr) {
      Graph* g = options.graph->get();

      GraphToPbTextFile(g, filename_prefix + ".pbtxt");
      GraphToDotFile(g, filename_prefix + ".dot",
                     "TensorFlow graph at POST_REWRITE_FOR_EXEC");
    }

    VLOG(0) << "nGraph dump pass done";

    return Status::OK();
  }

 private:
  static int s_counter;
};

int NGraphAnalyzerPass::s_counter = 0;

}  // namespace ngraph_bridge

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 999,
                      ngraph_bridge::NGraphAnalyzerPass);

}  // namespace tensorflow
