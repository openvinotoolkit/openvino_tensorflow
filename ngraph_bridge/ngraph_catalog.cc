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
#include "tensorflow/core/lib/core/errors.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend_manager.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_catalog.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

unordered_map<string, string> NGraphCatalog::input_variable_sharedname_map_;
unordered_map<string, unordered_set<int>>
    NGraphCatalog::encap_output_copy_indexes_map_;
unordered_map<string, tuple<string, bool>>
    NGraphCatalog::encap_output_info_map_;
unordered_map<string, map<int, int>> NGraphCatalog::prefetched_input_index_map_;

// Function to create the Node Key
string NGraphCatalog::CreateNodeKey(const int& graph_id,
                                    const string& node_name, const int& index) {
  if (index == 0) {
    return to_string(graph_id) + "_" + node_name;
  }
  return to_string(graph_id) + "_" + node_name + ":" + to_string(index);
}

string NGraphCatalog::CreateNodeKey(const int& graph_id,
                                    const string& node_name) {
  return to_string(graph_id) + "_" + node_name;
}

void NGraphCatalog::ClearCatalog() {
  NGraphCatalog::ClearInputVariableSharedNameMap();
  NGraphCatalog::ClearEncapOutputCopyIndexesMap();
  NGraphCatalog::ClearEncapOutputInfoMap();
  NGraphCatalog::ClearPrefetchedInputIndexMap();
}

// Functions for Encapsulate Output Copy Indexes Map
void NGraphCatalog::AddToEncapOutputCopyIndexesMap(
    const int& graphid, const string& node_name,
    const unordered_set<int>& val) {
  if (NGraphCatalog::EncapOutputNeedsCopy(graphid, node_name)) {
    throw runtime_error(
        "Trying to add an already existing key in EncapOutputIndexesCopy Map");
  }
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name);
  NGraphCatalog::encap_output_copy_indexes_map_.insert({key, val});
}

void NGraphCatalog::ClearEncapOutputCopyIndexesMap() {
  NGraphCatalog::encap_output_copy_indexes_map_.clear();
}

const unordered_set<int>& NGraphCatalog::GetEncapOutputIndexesThatNeedCopy(
    const int& graphid, const string& node_name) {
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name);
  return NGraphCatalog::encap_output_copy_indexes_map_.at(key);
}

bool NGraphCatalog::EncapOutputNeedsCopy(const int& graphid,
                                         const string& node_name) {
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name);
  auto itr = NGraphCatalog::encap_output_copy_indexes_map_.find(key);
  return itr != NGraphCatalog::encap_output_copy_indexes_map_.end();
}

bool NGraphCatalog::EncapOutputIndexNeedsCopy(const int& graphid,
                                              const string& node_name,
                                              const int& index) {
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name);
  auto itr = NGraphCatalog::encap_output_copy_indexes_map_.find(key);
  if (itr != NGraphCatalog::encap_output_copy_indexes_map_.end()) {
    auto op_copy_indexes = itr->second;
    return (op_copy_indexes.find(index) != op_copy_indexes.end());
  }
  return false;
}

void NGraphCatalog::DeleteFromEncapOutputCopyIndexesMap(
    const int& graphid, const string& node_name) {
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name);
  NGraphCatalog::encap_output_copy_indexes_map_.erase(key);
}

// Functions relating Input Variable Shared Name Map
void NGraphCatalog::AddToInputVariableSharedNameMap(const string& key,
                                                    const string& val) {
  if (NGraphCatalog::ExistsInInputVariableSharedNameMap(key)) {
    throw runtime_error(
        "Trying to add an already existing key in InputVariableSharedName Map");
  }
  NGraphCatalog::input_variable_sharedname_map_.insert({key, val});
}

void NGraphCatalog::ClearInputVariableSharedNameMap() {
  NGraphCatalog::input_variable_sharedname_map_.clear();
}

const string& NGraphCatalog::GetInputVariableSharedName(
    const int& graphid, const string& node_name, const int& input_index) {
  string node_key =
      NGraphCatalog::CreateNodeKey(graphid, node_name, input_index);
  return NGraphCatalog::input_variable_sharedname_map_.at(node_key);
}

bool NGraphCatalog::ExistsInInputVariableSharedNameMap(const string& key) {
  auto itr = NGraphCatalog::input_variable_sharedname_map_.find(key);
  return itr != NGraphCatalog::input_variable_sharedname_map_.end();
}

bool NGraphCatalog::ExistsInInputVariableSharedNameMap(const int& graphid,
                                                       const string& node_name,
                                                       const int& input_index) {
  return NGraphCatalog::ExistsInInputVariableSharedNameMap(
      NGraphCatalog::CreateNodeKey(graphid, node_name, input_index));
}

void NGraphCatalog::DeleteFromInputVariableSharedNameMap(const string& key) {
  NGraphCatalog::input_variable_sharedname_map_.erase(key);
}

// Functions for EncapOutputInfo Map
void NGraphCatalog::AddToEncapOutputInfoMap(const string& key,
                                            const tuple<string, bool>& val) {
  if (NGraphCatalog::ExistsInEncapOutputInfoMap(key)) {
    throw runtime_error(
        "Trying to add an already existing key in EncapOutputInfo Map");
  }
  NGraphCatalog::encap_output_info_map_.insert({key, val});
}

void NGraphCatalog::AddToEncapOutputInfoMap(const string& key,
                                            const string& shared_name,
                                            const bool& update_tf_tensor) {
  if (NGraphCatalog::ExistsInEncapOutputInfoMap(key)) {
    throw runtime_error(
        "Trying to add an already existing key in EncapOutputInfo Map");
  }

  // create a tuple
  tuple<string, bool> val = make_tuple(shared_name, update_tf_tensor);
  NGraphCatalog::encap_output_info_map_.insert({key, val});
}

bool NGraphCatalog::ExistsInEncapOutputInfoMap(const string& key) {
  auto itr = NGraphCatalog::encap_output_info_map_.find(key);
  return itr != NGraphCatalog::encap_output_info_map_.end();
}

bool NGraphCatalog::ExistsInEncapOutputInfoMap(const int& graphid,
                                               const string& node_name,
                                               const int& output_index) {
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name, output_index);
  auto itr = NGraphCatalog::encap_output_info_map_.find(key);
  return itr != NGraphCatalog::encap_output_info_map_.end();
}

const tuple<string, bool>& NGraphCatalog::GetInfoFromEncapOutputInfoMap(
    const int& graphid, const string& node_name, const int& output_index) {
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name, output_index);
  return NGraphCatalog::GetInfoFromEncapOutputInfoMap(key);
}

const tuple<string, bool>& NGraphCatalog::GetInfoFromEncapOutputInfoMap(
    const string& key) {
  return NGraphCatalog::encap_output_info_map_.at(key);
}

const string& NGraphCatalog::GetVariableSharedNameFromEncapOutputInfoMap(
    const string& key) {
  tuple<string, bool>& val = NGraphCatalog::encap_output_info_map_.at(key);
  return get<0>(val);
}

const bool& NGraphCatalog::GetUpdateTFTensorFromEncapOutputInfoMap(
    const string& key) {
  tuple<string, bool>& val = NGraphCatalog::encap_output_info_map_.at(key);
  return get<1>(val);
}

void NGraphCatalog::DeleteFromEncapOutputInfoMap(const string& key) {
  NGraphCatalog::encap_output_info_map_.erase(key);
}

void NGraphCatalog::ClearEncapOutputInfoMap() {
  NGraphCatalog::encap_output_info_map_.clear();
}

void NGraphCatalog::PrintEncapOutputInfoMap() {
  NGRAPH_VLOG(4) << "EncapOutputInfoMap";
  for (auto it : encap_output_info_map_) {
    NGRAPH_VLOG(4) << "Key: (GraphId_NodeName:OutputIndex) " << it.first
                   << " Value: (shared_name, update_tf_tensor) "
                   << get<0>(it.second) << " " << get<1>(it.second);
  }
}

// Functions for PrefetchedInputIndex Map
void NGraphCatalog::AddToPrefetchedInputIndexMap(
    const int& graphid, const string& node_name,
    const map<int, int>& encap_inp_index_map) {
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name);
  if (NGraphCatalog::ExistsInPrefetchedInputIndexMap(key)) {
    throw runtime_error("Trying to add an already existing key ( " + key +
                        " ) in PrefetchedInputIndexMap ");
  }
  NGraphCatalog::prefetched_input_index_map_.insert({key, encap_inp_index_map});
}

bool NGraphCatalog::ExistsInPrefetchedInputIndexMap(const int& graphid,
                                                    const string& node_name) {
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name);
  return NGraphCatalog::ExistsInPrefetchedInputIndexMap(key);
}

bool NGraphCatalog::ExistsInPrefetchedInputIndexMap(const string& key) {
  auto itr = NGraphCatalog::prefetched_input_index_map_.find(key);
  return itr != NGraphCatalog::prefetched_input_index_map_.end();
}

const map<int, int>& NGraphCatalog::GetIndexesFromPrefetchedInputIndexMap(
    const int& graphid, const string& node_name) {
  string key = NGraphCatalog::CreateNodeKey(graphid, node_name);
  return NGraphCatalog::prefetched_input_index_map_.at(key);
}

void NGraphCatalog::ClearPrefetchedInputIndexMap() {
  NGraphCatalog::prefetched_input_index_map_.clear();
}

void NGraphCatalog::PrintPrefetchedInputIndexMap() {
  NGRAPH_VLOG(4) << "PrefetchedInputIndexMap";
  for (auto it : prefetched_input_index_map_) {
    NGRAPH_VLOG(4) << "Key: (GraphId_NodeName) " << it.first;
    for (auto itr = it.second.begin(); itr != it.second.end(); ++itr) {
      NGRAPH_VLOG(4) << " NGEncap Input Index: " << itr->first
                     << ", IteratorGetNext Output Index: " << itr->second;
    }
  }
}
}  // ngraph_bridge
}  // tensorflow
