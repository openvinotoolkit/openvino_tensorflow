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

#include "ngraph_log.h"
#include "ngraph_utils.h"
#include "tf_graph_writer.h"

using namespace std;
namespace ngraph_bridge {

class NGraphDumpPass : public tensorflow::GraphOptimizationPass {
 public:
  NGraphDumpPass(std::string pass_name) : m_pass_name(pass_name) {}
  virtual ~NGraphDumpPass() {}
  tf::Status Run(const tf::GraphOptimizationPassOptions& options) {
    NGRAPH_VLOG(2) << "nGraph dump pass start: " << m_pass_name;

    int idx = s_counter_map[m_pass_name]++;

    std::stringstream ss;
    ss << "ngraph_dump_" << m_pass_name << "_" << idx;
    std::string filename_prefix = ss.str();

    if (options.graph != nullptr) {
      tf::Graph* g = options.graph->get();

      GraphToPbTextFile(g, filename_prefix + ".pbtxt");
      GraphToDotFile(g, filename_prefix + ".dot", "nGraph Dump: " + m_pass_name,
                     false);
    }

    if (options.partition_graphs != nullptr) {
      int sub_idx = 0;

      for (auto& kv : *options.partition_graphs) {
        tf::Graph* pg = kv.second.get();

        std::stringstream ss;
        ss << filename_prefix << "_" << sub_idx;
        std::string sub_filename_prefix = ss.str();

        GraphToPbTextFile(pg, sub_filename_prefix + ".pbtxt");
        GraphToDotFile(pg, sub_filename_prefix + ".dot",
                       "nGraph Subgraph Dump: " + m_pass_name, false);

        sub_idx++;
      }
    }

    NGRAPH_VLOG(2) << "nGraph dump pass done: " << m_pass_name;

    return tf::Status::OK();
  }

 private:
  std::string m_pass_name;
  static std::map<std::string, int> s_counter_map;
};

std::map<std::string, int> NGraphDumpPass::s_counter_map;

class NGraphDumpPrePlacement : public NGraphDumpPass {
 public:
  NGraphDumpPrePlacement() : NGraphDumpPass("pre_placement") {}
};

class NGraphDumpPostPlacement : public NGraphDumpPass {
 public:
  NGraphDumpPostPlacement() : NGraphDumpPass("post_placement") {}
};

class NGraphDumpPostReWrite : public NGraphDumpPass {
 public:
  NGraphDumpPostReWrite() : NGraphDumpPass("post_rewrite") {}
};

class NGraphDumpPostClustering : public NGraphDumpPass {
 public:
  NGraphDumpPostClustering() : NGraphDumpPass("post_clustering") {}
};

class NGraphDumpPostEncapsulation : public NGraphDumpPass {
 public:
  NGraphDumpPostEncapsulation() : NGraphDumpPass("post_encapsulation") {}
};

class NGraphDumpPostPartitioning : public NGraphDumpPass {
 public:
  NGraphDumpPostPartitioning() : NGraphDumpPass("post_partitioning") {}
};
}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 100,
                      ngraph_bridge::NGraphDumpPrePlacement);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 100,
                      ngraph_bridge::NGraphDumpPostPlacement);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 100,
                      ngraph_bridge::NGraphDumpPostReWrite);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 105,
                      ngraph_bridge::NGraphDumpPostClustering);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 115,
                      ngraph_bridge::NGraphDumpPostEncapsulation);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 100,
                      ngraph_bridge::NGraphDumpPostPartitioning);
}  // namespace tensorflow
