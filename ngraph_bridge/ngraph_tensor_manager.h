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

using namespace std;
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

  const vector<int>& GetPrefetchedInputIndexes() {
    return m_prefetched_input_indexes;
  }

 private:
  void Initialize();
  string m_ng_encap_node_name;
  int m_ng_encap_cluster_id;
  int m_ng_encap_graph_id;
  int m_number_of_inputs;
  int m_number_of_outputs;

  // Book-keeping for weights-on-device optimizations
  vector<int> m_input_indexes_from_variables;
  vector<int> m_output_indexes_assigning_variable;
  vector<int> m_output_indexes_that_need_copy;

  // All indexes that are not for from/to variables
  vector<int> m_pipelined_input_indexes;
  vector<int> m_pipelined_output_indexes;

  //[TODO] Book-keeping for prefetched inputs
  vector<int> m_prefetched_input_indexes;
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_TF_TENSOR_MANAGER_H_
