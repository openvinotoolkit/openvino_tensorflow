/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#ifndef NGRAPH_EXECUTOR_H_
#define NGRAPH_EXECUTOR_H_
#pragma once

#include <mutex>
#include <ostream>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_pipelined_tensors.h"

namespace tensorflow {

namespace ngraph_bridge {

class NGraphExecutor {
 public:
  // Transforms, compiles and executes TesnorFlow computation graph using nGraph
  explicit NGraphExecutor(int instance_id, int cluster_id, int graph_id,
                          unique_ptr<tensorflow::Graph>& graph,
                          const string& backend_name);

  ~NGraphExecutor();

  // Calls Compute Signature and gets ngraph executable
  // Update the cache and if called again with the same input shapes,
  // return fromm the cache
  Status GetNgExecutable(const std::vector<Tensor>& tf_input_tensors,
                         std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
                         bool& cache_hit);

  Status GetNgFunction(
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      std::shared_ptr<ngraph::Function>& ng_function);

  // TODO Rename this to DecodeAttributes
  Status ParseNodeAttributes(
      const google::protobuf::Map<string, AttrValue>& additional_attributes,
      std::unordered_map<std::string, std::string>* additional_attribute_map);

  // Returns OK when a set of i/o tensor is available. Returns error message
  // otherwse.
  // The caller can wait or come back later - based on what the application
  // demands
  Status GetTensorsFromPipeline(
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>&
          io_tensors);

  void ReturnPipelinedTensors(
      std::shared_ptr<ngraph::runtime::Executable> ng_exec, size_t idx) {
    lock_guard<mutex> lock(m_mutex);
    m_executable_pipelined_tensors_map.at(ng_exec)->return_tensors(idx);
  }

  const int& GetNgraphClusterId() { return m_ngraph_cluster_id; }

  int GetGraphId() { return m_graph_id; }

  const string& GetOpBackendName() { return m_op_backend_name; }

  bool IsTensorPipeliningSupported() { return m_executable_can_create_tensor; }

  int GetTensorPipelineDepth() {
    return m_executable_can_create_tensor ? m_depth : 1;
  }

 private:
  // Allocates the necessary tensors from the Executable (or backend in future)
  // Since the pipeline cannot be created at the construction time, we need to
  // provide this as a separate function. It's ok to call this multiple
  // times - the Pipeline will be initialized only once
  Status InitializeIOTensorPipeline(
      std::shared_ptr<ngraph::runtime::Executable> ng_exec);

  // Get tensorflow input tensors, input shapes, static_inputs to Compute
  // Signature
  Status ComputeSignature(const std::vector<Tensor>& tf_input_tensors,
                          std::vector<TensorShape>& input_shapes,
                          std::vector<const Tensor*>& static_input_map,
                          std::stringstream& signature_ss) const;

 private:
  const int m_instance_id;
  const int m_ngraph_cluster_id{-1};
  const int m_graph_id{-1};
  const unique_ptr<Graph> m_graph;

  int my_function_cache_depth_in_items = 16;
  const string m_op_backend_name;
  string m_node_name;
  std::vector<bool> m_input_is_static;
  std::list<std::string> m_lru;
  bool m_do_aot = false;
  map<string, string> m_aot_functions;
  map<string, string> m_aot_execs;

  // ng_function, ng_executable, Output and Input Cache maps
  std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Executable>>
      m_ng_exec_map;
  std::unordered_map<std::shared_ptr<ngraph::runtime::Executable>,
                     std::shared_ptr<ngraph::Function>>
      m_ng_function_map;

  std::unordered_map<
      std::shared_ptr<ngraph::runtime::Executable>,
      std::vector<std::pair<void*, shared_ptr<ng::runtime::Tensor>>>>
      m_ng_exec_input_cache_map;

  std::unordered_map<
      std::shared_ptr<ngraph::runtime::Executable>,
      std::vector<std::pair<void*, shared_ptr<ng::runtime::Tensor>>>>
      m_ng_exec_output_cache_map;

  bool m_executable_can_create_tensor;
  std::unordered_map<std::shared_ptr<ngraph::runtime::Executable>,
                     shared_ptr<PipelinedTensorsStore>>
      m_executable_pipelined_tensors_map;

  mutex m_mutex;
  int m_depth{2};  // TODO make this settable
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_EXECUTOR_H_
