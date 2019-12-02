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
#include "ngraph_bridge/ngraph_data_cache.h"
#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_pipelined_tensors.h"
#include "ngraph_bridge/ngraph_tensor_manager.h"

namespace tensorflow {

namespace ngraph_bridge {

class NGraphExecutor {
 public:
  // Transforms, compiles and executes TesnorFlow computation graph using nGraph
  explicit NGraphExecutor(int instance_id, int cluster_id, int graph_id,
                          unique_ptr<tensorflow::Graph>& graph,
                          const string& backend_name, const int cache_depth);

  ~NGraphExecutor();

  // Calls Compute Signature and gets ngraph executable
  // Update the cache and if called again with the same input shapes,
  // return fromm the cache
  Status GetExecutableFunctionAndTensors(
      const std::vector<Tensor>& tf_input_tensors,
      std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      std::string& serialized_ng_function,
      shared_ptr<PipelinedTensorsStore>& pts, bool& cache_hit);

  // TODO Rename this to DecodeAttributes
  Status ParseNodeAttributes(
      const google::protobuf::Map<string, AttrValue>& additional_attributes,
      std::unordered_map<std::string, std::string>* additional_attribute_map);

  // Callback function called from NgraphDataCache's LookUpOrCreateItem() method
  // Creates ng_executable, serialized_ng_function, and initializes I/O
  // TensorPipeline
  std::pair<Status, std::tuple<std::shared_ptr<ngraph::runtime::Executable>,
                               std::string, shared_ptr<PipelinedTensorsStore>>>
  CreateCallback(std::string signature, std::vector<TensorShape> input_shapes,
                 std::vector<const Tensor*> static_input_map,
                 ng::runtime::Backend*& op_backend);

  const int& GetNgraphClusterId() { return m_ngraph_cluster_id; }

  void DestroyCallback(
      std::tuple<std::shared_ptr<ngraph::runtime::Executable>, std::string,
                 shared_ptr<PipelinedTensorsStore>>
          evicted_ng_item,
      ng::runtime::Backend*& op_backend);
  const string& GetNgraphClusterName() { return m_node_name; }

  int GetGraphId() { return m_graph_id; }

  const string& GetOpBackendName() { return m_op_backend_name; }

  bool IsTensorPipeliningSupported() { return m_executable_can_create_tensor; }

  int GetTensorPipelineDepth() {
    return m_executable_can_create_tensor ? m_depth : 1;
  }

  const shared_ptr<NGraphTensorManager>& GetTensorManager() {
    return m_tensor_manager;
  }

 private:
  // This method is called from CreateCallback(), It compiles ngraph
  // Or load ng_executable from backend in case of AOT
  std::pair<Status, std::shared_ptr<ngraph::runtime::Executable>>
  GetNgExecutable(std::string signature,
                  std::shared_ptr<ngraph::Function>& ng_function,
                  ng::runtime::Backend*& op_backend);
  // Allocates the necessary tensors from the Executable (or backend in future)
  // Called from CreateCallback
  std::pair<Status, shared_ptr<PipelinedTensorsStore>>
  InitializeIOTensorPipeline(
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

  int my_function_cache_depth_in_items;
  const string m_op_backend_name;
  string m_node_name;
  std::vector<bool> m_input_is_static;
  std::list<std::string> m_lru;
  bool m_do_aot = false;
  map<string, string> m_aot_functions;
  map<string, string> m_aot_execs;

  // NgraphDataCache<Key, Value> where key is signature, and value is a tuple
  // of ng_executable, serialized_ng_function and PipelinedTensorsStore
  NgraphDataCache<std::string,
                  std::tuple<std::shared_ptr<ngraph::runtime::Executable>,
                             std::string, shared_ptr<PipelinedTensorsStore>>>
      m_ng_data_cache;

  bool m_executable_can_create_tensor;

  mutex m_mutex;
  int m_depth{2};  // TODO make this settable

  // NGraphTensorManager
  shared_ptr<NGraphTensorManager> m_tensor_manager;
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_EXECUTOR_H_
