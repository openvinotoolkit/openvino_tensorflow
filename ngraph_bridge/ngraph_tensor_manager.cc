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

#include "ngraph_bridge/ngraph_tensor_manager.h"
#include "ngraph_bridge/ngraph_catalog.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//---------------------------------------------------------------------------
//  NGraphTensorManager::NGraphTensorManager
//---------------------------------------------------------------------------
NGraphTensorManager::NGraphTensorManager(const string ng_encap_node_name,
                                         const int ng_encap_cluster_id,
                                         const int ng_encap_graph_id,
                                         const int number_of_inputs,
                                         const int number_of_outputs)
    : m_ng_encap_node_name(ng_encap_node_name),
      m_ng_encap_cluster_id(ng_encap_cluster_id),
      m_ng_encap_graph_id(ng_encap_graph_id),
      m_number_of_inputs(number_of_inputs),
      m_number_of_outputs(number_of_outputs) {
  Initialize();
}

void NGraphTensorManager::Initialize() {
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  // input variables book-keeping
  for (int index = 0; index < m_number_of_inputs; index++) {
    if (NGraphCatalog::ExistsInInputVariableSharedNameMap(
            m_ng_encap_graph_id, m_ng_encap_node_name, index)) {
      m_input_indexes_from_variables.push_back(index);
      // store the variable shared name
      try {
        auto shared_name = NGraphCatalog::GetInputVariableSharedName(
            m_ng_encap_graph_id, m_ng_encap_node_name, index);
        input_variable_shared_name_map.insert({index, shared_name});
      } catch (const std::exception& exp) {
        throw runtime_error(
            "Could not find variable shared name in catalog for input index " +
            to_string(index) + "for encapsulate op " + m_ng_encap_node_name);
      }
    }
  }

  // output variables book-keeping
  // these weights are updated in place
  for (int index = 0; index < m_number_of_outputs; index++) {
    if (NGraphCatalog::ExistsInEncapOutputInfoMap(
            m_ng_encap_graph_id, m_ng_encap_node_name, index)) {
      m_output_indexes_assigning_variable.push_back(index);

      // store the output variable shared name + update_tf_tensor info
      try {
        auto shared_name_update_tf_tensor =
            NGraphCatalog::GetInfoFromEncapOutputInfoMap(
                m_ng_encap_graph_id, m_ng_encap_node_name, index);
        output_variable_info_map.insert({index, shared_name_update_tf_tensor});
      } catch (const std::exception& exp) {
        throw runtime_error(
            "Could not find variable shared name and update_tf_tensor "
            "information in "
            "catalog for output index " +
            to_string(index) + " for encapsulate op " + m_ng_encap_node_name);
      }
    }
    if (NGraphCatalog::EncapOutputIndexNeedsCopy(m_ng_encap_graph_id,
                                                 m_ng_encap_node_name, index)) {
      m_output_indexes_that_need_copy.push_back(index);
    }
  }

  // For graphs that were run through AOT
  // Graph rewrite is not done, and there is no entry in catalog
  // If there is no entry in catalog all outputs need to be copied
  if (!NGraphCatalog::EncapOutputNeedsCopy(m_ng_encap_graph_id,
                                           m_ng_encap_node_name)) {
    m_output_indexes_that_need_copy.resize(m_number_of_outputs);
    iota(begin(m_output_indexes_that_need_copy),
         end(m_output_indexes_that_need_copy), 0);
  }

#else
  m_output_indexes_that_need_copy.resize(m_number_of_outputs);
  iota(begin(m_output_indexes_that_need_copy),
       end(m_output_indexes_that_need_copy), 0);
#endif
  m_pipelined_input_indexes =
      FindComplement(m_number_of_inputs, m_input_indexes_from_variables);
  m_pipelined_output_indexes =
      FindComplement(m_number_of_outputs, m_output_indexes_assigning_variable);

  // Prefetch indexes
  if (NGraphCatalog::ExistsInPrefetchedInputIndexMap(m_ng_encap_graph_id,
                                                     m_ng_encap_node_name)) {
    auto prefetch_index_map =
        NGraphCatalog::GetIndexesFromPrefetchedInputIndexMap(
            m_ng_encap_graph_id, m_ng_encap_node_name);

    // Since it's a map, the keys must be sorted
    for (auto itr : prefetch_index_map) {
      int prefetch_index = itr.first;
      m_prefetched_input_indexes.push_back(prefetch_index);

      // the prefetched input indexes will also be pipelined
      auto position =
          std::find(m_pipelined_input_indexes.begin(),
                    m_pipelined_input_indexes.end(), prefetch_index);
      if (position == m_pipelined_input_indexes.end()) {
        throw std::runtime_error("Prefetched input index " +
                                 to_string(prefetch_index) +
                                 " not found in pipelined inputs.");
      }

      int prefetch_index_wrt_pipeline =
          position - m_pipelined_input_indexes.begin();
      m_pipelined_input_indexes_that_are_prefetched.push_back(
          prefetch_index_wrt_pipeline);

      // The EncapOp shares with prefetch op only the pipelined tensors
      // so the map needs relative indexing for encap op
      // the corresponding iteratorgetnext output ops
      int corres_iterator_index = prefetch_index_map.at(prefetch_index);
      m_prefetch_iterator_encap_index_map.insert(
          {prefetch_index_wrt_pipeline, corres_iterator_index});
    }
  }  // if prefetch input found in catalog

  // complements
  m_pipelined_input_indexes_that_are_not_prefetched =
      FindComplement(m_pipelined_input_indexes.size(),
                     m_pipelined_input_indexes_that_are_prefetched);
  m_pipelined_not_prefetched_input_indexes =
      FindComplement(m_pipelined_input_indexes, m_prefetched_input_indexes);
}

//---------------------------------------------------------------------------
//  NGraphTensorManager::Print
//---------------------------------------------------------------------------
void NGraphTensorManager::Print() {
  auto PrintVector = [](const vector<int>& input_vector, const string title) {
    cout << title << endl;
    cout << ng::join(input_vector) << endl;
  };

  cout << "** NGEncapsulate TensorManager:" << m_ng_encap_node_name << " **"
       << endl;

  cout << "** Variables Related **" << endl;
  PrintVector(m_input_indexes_from_variables, "Input Indexes from Variables");
  PrintVector(m_output_indexes_assigning_variable,
              "Output Indexes Referring to Variables");
  PrintVector(m_output_indexes_that_need_copy, "Output Indexes to be Read");

  cout << "** Pipelined **" << endl;
  PrintVector(m_pipelined_input_indexes, "Pipelined Input Indexes");
  PrintVector(m_pipelined_output_indexes, "Pipelined Output Indexes");

  cout << "** Prefetched **" << endl;
  PrintVector(m_prefetched_input_indexes, "Prefetched Input Indexes");
  PrintVector(m_pipelined_not_prefetched_input_indexes,
              "Pipelined But Not Prefetched Input Indexes");

  cout << "** Prefetched wrt pipelined indexes **" << endl;
  PrintVector(m_pipelined_input_indexes_that_are_prefetched,
              "Prefetched Input Indexes wrt Pipelined Inputs");
  PrintVector(m_pipelined_input_indexes_that_are_not_prefetched,
              "Not Prefetched Input Indexes wrt Pipelined Inputs");
}

//---------------------------------------------------------------------------
//  NGraphTensorManager::~NGraphTensorManager
//---------------------------------------------------------------------------
NGraphTensorManager::~NGraphTensorManager() {}

//---------------------------------------------------------------------------
//  NGraphTensorManager::GetInputVariableSharedName
//---------------------------------------------------------------------------
Status NGraphTensorManager::GetInputVariableSharedName(
    const int& input_index, string* input_var_shared_name) {
  auto itr = input_variable_shared_name_map.find(input_index);
  if (itr == input_variable_shared_name_map.end()) {
    return errors::Internal("Could not find shared name info for input index ",
                            input_index, " in tensor manager ");
  }
  *input_var_shared_name = itr->second;
  return Status::OK();
}

//---------------------------------------------------------------------------
//  NGraphTensorManager::GetOutputVariableSharedName
//---------------------------------------------------------------------------
Status NGraphTensorManager::GetOutputVariableSharedName(
    const int& output_index, string* output_var_shared_name) {
  auto itr = output_variable_info_map.find(output_index);
  if (itr == output_variable_info_map.end()) {
    return errors::Internal("Could not find shared name info for output index ",
                            output_index, " in tensor manager");
  }
  *output_var_shared_name = get<0>(itr->second);
  return Status::OK();
}

//---------------------------------------------------------------------------
//  NGraphTensorManager::GetOutputVariableUpdateTFTensor
//---------------------------------------------------------------------------
Status NGraphTensorManager::GetOutputVariableUpdateTFTensor(
    const int& output_index, bool* output_var_update_tf_tensor) {
  auto itr = output_variable_info_map.find(output_index);
  if (itr == output_variable_info_map.end()) {
    return errors::Internal(
        "Could not find update_tf_tensor info for output index ", output_index,
        " in tensor manager");
  }
  *output_var_update_tf_tensor = get<1>(itr->second);
  return Status::OK();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow