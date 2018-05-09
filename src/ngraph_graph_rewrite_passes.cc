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
#include "ngraph_utils.h"

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
namespace ngraph_bridge {

class NGraphPass : public tensorflow::GraphOptimizationPass {
public:
  NGraphPass(const char *title) : m_title(title) {}
  virtual ~NGraphPass() {}
  tf::Status Run(const tf::GraphOptimizationPassOptions &options) {
    VLOG(0) << "NGraph PASS: " << GetPassName();
// std::cout << "Running NGraphPass: " << GetPassName() << std::endl;

#if 0
    // NOTE: If we need to dump the proto text then we need to ensure that 
    // this is not POST_PATRITIONING
    tf::Graph* graph = options.graph->get();
    // Create the graphDef
    tf::GraphDef g_def;
    graph->ToGraphDef(&g_def);

    // Create the filename
    std::string path = "./ngraph_pass_" + GetPassName() + ".pb";
    tf::Status status = tf::WriteTextProto(tf::Env::Default(), path, g_def);
#endif

    // Call the derived class's implementation
    return RunImpl(options);
  }

protected:
  virtual std::string GetPassName() = 0;
  virtual tf::Status RunImpl(const tf::GraphOptimizationPassOptions &options) {
    return tf::Status::OK();
  }

private:
  std::string m_title;
};

class NGraphPassPrePlacement : public NGraphPass {
public:
  NGraphPassPrePlacement() : NGraphPass("NGraph PRE_PLACEMENT Pass") {}

protected:
  std::string GetPassName() { return "PRE_PLACEMENT"; }
  tf::Status RunImpl(const tf::GraphOptimizationPassOptions &options) {
    // Save the graph as ptoro text
    GraphToPbTextFile(options.graph->get(),
                      "./ngraph_pass_" + GetPassName() + ".pbtxt");

    // Save as DOT
    GraphToDotFile(options.graph->get(),
                   "./ngraph_pass_" + GetPassName() + ".dot",
                   "NGraph POST_PLACEMENT", false);
    return tf::Status::OK();
  }
};

class NGraphPassPostPlacement : public NGraphPass {
public:
  NGraphPassPostPlacement() : NGraphPass("NGraph POST_PLACEMENT") {}
  tf::Status RunImpl(const tf::GraphOptimizationPassOptions &options) {
    // Save the graph as ptoro text
    GraphToPbTextFile(options.graph->get(),
                      "./ngraph_pass_" + GetPassName() + ".pbtxt");

    // Save as DOT
    GraphToDotFile(options.graph->get(),
                   "./ngraph_pass_" + GetPassName() + ".dot",
                   "NGraph POST_PLACEMENT", true);
    return tf::Status::OK();
  }

protected:
  std::string GetPassName() { return "POST_PLACEMENT"; }
};

class NGraphPassPostReWrite : public NGraphPass {
public:
  NGraphPassPostReWrite() : NGraphPass("NGraph POST_REWRITE_FOR_EXEC Pass") {}

  tf::Status RunImpl(const tf::GraphOptimizationPassOptions &options) {
    static int count = 1;
    std::ostringstream filename;
    filename << "./ngraph_pass_" << GetPassName() << "_" << count;
    count++;
    // Save the graph as ptoro text
    GraphToPbTextFile(options.graph->get(), filename.str() + ".pbtxt");

    // Save as DOT
    GraphToDotFile(options.graph->get(), filename.str() + ".dot",
                   "NGraph POST_REWRITE_FOR_EXEC", true);
    return tf::Status::OK();
  }

protected:
  std::string GetPassName() { return "POST_REWRITE_FOR_EXEC"; }
};

class NGraphPostPartitioning : public NGraphPass {
public:
  NGraphPostPartitioning() : NGraphPass("NGraph POST_PARTITIONING Pass") {}

  tf::Status RunImpl(const tf::GraphOptimizationPassOptions &options) {
    static int count = 1;

    if (options.partition_graphs != nullptr) {
      for (auto &pg : *options.partition_graphs) {
        tf::Graph *graph = pg.second.get();
        std::ostringstream filename;
        filename << "./ngraph_pass_" << GetPassName() << count;
        count++;
        // Save the graph as ptoro text
        GraphToPbTextFile(graph, filename.str() + ".pbtxt");

        // Save as DOT
        GraphToDotFile(graph, filename.str() + ".dot",
                       "NGraph POST_PARTITIONING", true);
      }
    }

    return tf::Status::OK();
  }

protected:
  std::string GetPassName() { return "POST_PARTITIONING"; }
};

class NGraphPostEncapsulation : public NGraphPass {
public:
  NGraphPostEncapsulation() : NGraphPass("NGraph POST_ENCAPSULATION Pass") {}

  tf::Status RunImpl(const tf::GraphOptimizationPassOptions &options) {
    static int count = 1;

    if (options.partition_graphs != nullptr) {
      for (auto &pg : *options.partition_graphs) {
        tf::Graph *graph = pg.second.get();
        std::ostringstream filename;
        filename << "./ngraph_pass_" << GetPassName() << count;
        count++;
        // Save the graph as ptoro text
        GraphToPbTextFile(graph, filename.str() + ".pbtxt");

        // Save as DOT
        GraphToDotFile(graph, filename.str() + ".dot",
                       "NGraph POST_ENCAPSULATION", true);
      }
    }

    return tf::Status::OK();
  }

protected:
  std::string GetPassName() { return "POST_ENCAPSULATION"; }
};
} // namespace ngraph_bridge

namespace tensorflow {
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 100,
                      ngraph_bridge::NGraphPassPrePlacement);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 100,
                      ngraph_bridge::NGraphPassPostPlacement);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 100,
                      ngraph_bridge::NGraphPassPostReWrite);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 100,
                      ngraph_bridge::NGraphPostPartitioning);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 110,
                      ngraph_bridge::NGraphPostEncapsulation);
} // namespace tensorflow
