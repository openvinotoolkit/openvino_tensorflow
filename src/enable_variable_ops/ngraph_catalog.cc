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
#include "tensorflow/core/lib/core/errors.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph_catalog.h"
#include "ngraph_log.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

unordered_map<string, string> NGraphCatalog::input_variable_sharedname_map_;
map<string, shared_ptr<ng::runtime::Tensor>>
    NGraphCatalog::encap_output_tensor_map_;
unordered_map<string, unordered_set<int>>
    NGraphCatalog::encap_output_copy_indexes_map_;

// Functions for Encapsulate Output Copy Indexes Map
void NGraphCatalog::AddToEncapOutputCopyIndexesMap(string key,
                                                   unordered_set<int> val) {
  NGraphCatalog::encap_output_copy_indexes_map_[key] = val;
}

unordered_set<int> NGraphCatalog::GetEncapOutputIndexesThatNeedCopy(
    string key) {
  return NGraphCatalog::encap_output_copy_indexes_map_[key];
}

bool NGraphCatalog::EncapOutputIndexNeedsCopy(string key, int index) {
  auto itr = NGraphCatalog::encap_output_copy_indexes_map_.find(key);
  if (itr != NGraphCatalog::encap_output_copy_indexes_map_.end()) {
    auto op_copy_indexes = itr->second;
    return (op_copy_indexes.find(index) != op_copy_indexes.end());
  }
  // Should not reach here
  return true;
}

string NGraphCatalog::CreateNodeKey(int graph_id, string node_name, int index) {
  if (index == 0) {
    return to_string(graph_id) + "_" + node_name;
  }
  return to_string(graph_id) + "_" + node_name + ":" + to_string(index);
}

// Functions for OutputTensorMap
void NGraphCatalog::AddToEncapOutputTensorMap(
    string key, shared_ptr<ng::runtime::Tensor> ng_val) {
  NGraphCatalog::encap_output_tensor_map_[key] = ng_val;
}

bool NGraphCatalog::ExistsInEncapOutputTensorMap(string key) {
  auto itr = NGraphCatalog::encap_output_tensor_map_.find(key);
  return itr != NGraphCatalog::encap_output_tensor_map_.end();
}

bool NGraphCatalog::ExistsInEncapOutputTensorMap(int graphid, string node_name,
                                                 int input_index) {
  return NGraphCatalog::ExistsInEncapOutputTensorMap(
      NGraphCatalog::CreateNodeKey(graphid, node_name, input_index));
}

shared_ptr<ng::runtime::Tensor>
NGraphCatalog::GetTensorFromEncapOutputTensorMap(string key) {
  return NGraphCatalog::encap_output_tensor_map_[key];
}

void NGraphCatalog::DeleteFromEncapOutputTensorMap(string key) {
  NGraphCatalog::encap_output_tensor_map_.erase(key);
}

// Functions relating Input Variable Shared Name Map
string NGraphCatalog::GetInputVariableSharedName(int graphid, string node_name,
                                                 int input_index) {
  std::string node_key =
      NGraphCatalog::CreateNodeKey(graphid, node_name, input_index);
  return NGraphCatalog::input_variable_sharedname_map_[node_key];
}

void NGraphCatalog::AddToInputVariableSharedNameMap(string key, string val) {
  NGraphCatalog::input_variable_sharedname_map_[key] = val;
}

bool NGraphCatalog::ExistsInInputVariableSharedNameMap(string key) {
  auto itr = NGraphCatalog::input_variable_sharedname_map_.find(key);
  return itr != NGraphCatalog::input_variable_sharedname_map_.end();
}

bool NGraphCatalog::ExistsInInputVariableSharedNameMap(int graphid,
                                                       string node_name,
                                                       int input_index) {
  return NGraphCatalog::ExistsInInputVariableSharedNameMap(
      NGraphCatalog::CreateNodeKey(graphid, node_name, input_index));
}

}  // ngraph_bridge
}  // tensorflow
