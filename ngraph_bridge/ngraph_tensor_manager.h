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

#ifndef NGRAPH_TF_TENSOR_MANAGER_H_
#define NGRAPH_TF_TENSOR_MANAGER_H_
#pragma once

#include <mutex>
#include <ostream>
#include <vector>

#include "tensorflow/core/common_runtime/dma_helper.h"

#include "ngraph/ngraph.hpp"

using namespace std;
namespace ng = ngraph;
namespace tensorflow {

namespace ngraph_bridge {

class NGraphTensorManager {
 public:
  explicit NGraphTensorManager(const string ng_encap_node_name,
                               const int ng_encap_cluster_id,
                               const int ng_encap_graph_id,
                               const int number_of_inputs,
                               const int number_of_outputs);

  ~NGraphTensorManager();

  string GetName() { return m_ng_encap_node_name; }

  int GetClusterId() { return m_ng_encap_cluster_id; }

  int GetGraphId() { return m_ng_encap_graph_id; }

  const int& GetNumberOfInputs() { return m_number_of_inputs; }

  const int& GetNumberOfOutputs() { return m_number_of_outputs; }

  const vector<int>& GetInputIndexesFedByVariables() {
    return m_input_indexes_from_variables;
  }

  const vector<int>& GetOutputIndexesAssigningVariables() {
    return m_output_indexes_assigning_variable;
  }

  const vector<int>& GetOutputIndexesThatNeedCopy() {
    return m_output_indexes_that_need_copy;
  }

  const vector<int>& GetPipelinedInputIndexes() {
    return m_pipelined_input_indexes;
  }

  const vector<int>& GetPipelinedOutputIndexes() {
    return m_pipelined_output_indexes;
  }

  // wrt to all inputs
  const vector<int>& GetPrefetchedInputIndexes() {
    return m_prefetched_input_indexes;
  }

  // wrt to all inputs
  const vector<int>& GetPipelinedButNotPrefetchedInputIndexes() {
    return m_pipelined_not_prefetched_input_indexes;
  }

  // wrt to pipelined inputs
  const vector<int>& GetPipelinedInputIndexesThatArePrefetched() {
    return m_pipelined_input_indexes_that_are_prefetched;
  }

  // wrt to pipelined inputs
  const vector<int>& GetPipelinedInputIndexesThatAreNotPrefetched() {
    return m_pipelined_input_indexes_that_are_not_prefetched;
  }

  // input ng-variable shared name
  Status GetInputVariableSharedName(const int& input_index,
                                    string* input_var_shared_name);

  // output ng-variable shared name
  Status GetOutputVariableSharedName(const int& output_index,
                                     string* output_var_shared_name);

  // does output ng-variable's host-TF tensor needs to be updated
  Status GetOutputVariableCopyToTF(const int& output_index,
                                   bool* output_var_copy_to_tf);

 private:
  void Initialize();
  string m_ng_encap_node_name;
  int m_ng_encap_cluster_id;
  int m_ng_encap_graph_id;
  int m_number_of_inputs;
  int m_number_of_outputs;

  // Book-keeping for weights-on-device optimizations
  // indexes wrt all inputs/outputs
  vector<int> m_input_indexes_from_variables;
  vector<int> m_output_indexes_assigning_variable;
  vector<int> m_output_indexes_that_need_copy;

  // All indexes that are not from/to variables
  // Book-keeping primarily for data pipelining
  // These are pipelined, some of these are also prefetched
  // indexes wrt all inputs/outputs
  vector<int> m_pipelined_input_indexes;
  vector<int> m_pipelined_output_indexes;
  // indexes wrt pipelined inputs
  vector<int> m_pipelined_input_indexes_that_are_prefetched;
  vector<int> m_pipelined_input_indexes_that_are_not_prefetched;

  // indexes wrt all inputs
  vector<int> m_prefetched_input_indexes;
  vector<int> m_pipelined_not_prefetched_input_indexes;

  // Book-keeping for weights-on-device optimizations
  unordered_map<int, string> input_variable_shared_name_map;
  unordered_map<int, tuple<string, bool>> output_variable_info_map;
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_TF_TENSOR_MANAGER_H_
