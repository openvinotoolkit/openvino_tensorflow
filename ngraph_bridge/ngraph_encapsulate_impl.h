/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#ifndef NGRAPH_TF_ENCAPSULATE_IMPL_H_
#define NGRAPH_TF_ENCAPSULATE_IMPL_H_
#pragma once

#include <ostream>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/executable.h"

namespace tensorflow {
namespace ngraph_bridge {

class NGraphEncapsulateImpl {
 public:
  // Ngraph Encapsulate Implementation class for EncapsulateOp class
  explicit NGraphEncapsulateImpl();

  // Get tensorflow input tensors, input shapes, static_inputs to Compute
  // Signature
  Status ComputeSignature(const std::vector<Tensor>& tf_input_tensors,
                          std::vector<TensorShape>& input_shapes,
                          std::vector<const Tensor*>& static_input_map,
                          std::stringstream& signature_ss);

  // Calls Compute Signature and gets ngraph executable
  Status GetNgExecutable(const std::vector<Tensor>& tf_input_tensors,
                         std::vector<TensorShape>& input_shapes,
                         std::vector<const Tensor*>& static_input_map,
                         std::shared_ptr<Executable>& ng_exec);

  // Allocate nGraph tensors for given TF tensors
  Status AllocateNGTensors(
      const std::vector<Tensor>& tf_tensors,
      vector<shared_ptr<ngraph::runtime::Tensor>>& ng_tensors);

  // Clear all maps with ng_exec as keys
  void ClearExecMaps();

  // Accessors(getters and setters) for the private data members of
  // NgraphEncapsulateImpl class
  // needed by NgraphEncapsulateOp class

  int GetGraphId() { return m_graph_id; }

  void SetGraphId(const int& graph_id) { m_graph_id = graph_id; }

  const int& GetNgraphCluster() { return m_ngraph_cluster; }

  void SetNgraphCluster(const int& cluster) { m_ngraph_cluster = cluster; }

  const int& GetFunctionCache() { return m_function_cache_depth_in_items; }

  const int& GetNumberOfOutputs() { return m_number_outputs; }

  void SetNumberOfOutputs(const int& n) { m_number_outputs = n; }

  const int& GetNumberOfInputs() { return m_number_inputs; }

  void SetNumberOfInputs(const int& n) { m_number_inputs = n; }

  const int& GetInstanceId() { return my_instance_id; }

  const std::vector<bool> GetStaticInputVector() { return m_input_is_static; }

  void ResizeStaticInputVector(const int& size) {
    m_input_is_static.resize(size);
  }
  void SetStaticInputVector(const int& index, bool value) {
    m_input_is_static[index] = value;
  }

  void SetName(string name) { m_name = name; }

  Status ParseNodeAttributes(
      const google::protobuf::Map<string, AttrValue>& additional_attributes,
      std::unordered_map<std::string, std::string>* additional_attribute_map);

  // TF Graph for the cluster
  Graph m_graph;

 private:
  int m_ngraph_cluster{-1};
  int m_graph_id{-1};
  int m_function_cache_depth_in_items = 16;
  int m_number_outputs = -1;
  int m_number_inputs = -1;
  int my_instance_id{0};
  string m_name;
  std::vector<bool> m_input_is_static;
  std::list<std::string> m_lru;
  static int s_instance_count;

  std::unordered_map<std::string, std::shared_ptr<Executable>> m_ng_exec_map;
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_ENCAPSULATE_IMPL_H_
