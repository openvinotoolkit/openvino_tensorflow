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

#include "ngraph_bridge/ngraph_utils.h"
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
#include "ngraph_bridge/enable_variable_ops/ngraph_catalog.h"
#endif
#include "ngraph_bridge/ngraph_tensor_manager.h"

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
  for (int index = 0; index < m_number_of_inputs; index++) {
    if (NGraphCatalog::ExistsInInputVariableSharedNameMap(
            m_ng_encap_graph_id, m_ng_encap_node_name, index)) {
      m_input_indexes_from_variables.push_back(index);
    }
  }
  for (int index = 0; index < m_number_of_outputs; index++) {
    if (NGraphCatalog::ExistsInEncapOutputInfoMap(
            m_ng_encap_graph_id, m_ng_encap_node_name, index)) {
      m_output_indexes_assigning_variable.push_back(index);
    }
    if (NGraphCatalog::EncapOutputIndexNeedsCopy(m_ng_encap_graph_id,
                                                 m_ng_encap_node_name, index)) {
      m_output_indexes_that_need_copy.push_back(index);
    }
  }
#endif
  m_pipelined_input_indexes =
      FindComplement(m_number_of_inputs, m_input_indexes_from_variables);
  m_pipelined_output_indexes =
      FindComplement(m_number_of_outputs, m_output_indexes_assigning_variable);
}

//---------------------------------------------------------------------------
//  NGraphTensorManager::~NGraphTensorManager
//---------------------------------------------------------------------------
NGraphTensorManager::~NGraphTensorManager() {}

}  // namespace ngraph_bridge
}  // namespace tensorflow