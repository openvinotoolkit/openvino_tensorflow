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

#include "logging/ngraph_log.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_catalog.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

unordered_map<string, string> NGraphCatalog::input_variable_sharedname_map_;
unordered_map<string, unordered_set<int>>
    NGraphCatalog::encap_output_copy_indexes_map_;
unordered_map<string, tuple<string, bool, bool>>
    NGraphCatalog::encap_output_info_map_;

// Function to create the Node Key
string NGraphCatalog::CreateNodeKey(int graph_id, string node_name, int index) {
  if (index == 0) {
    return to_string(graph_id) + "_" + node_name;
  }
  return to_string(graph_id) + "_" + node_name + ":" + to_string(index);
}

// Functions for Encapsulate Output Copy Indexes Map
void NGraphCatalog::AddToEncapOutputCopyIndexesMap(int graphid,
                                                   string node_name,
                                                   unordered_set<int> val) {
  string key = graphid + "_" + node_name;
  NGraphCatalog::encap_output_copy_indexes_map_[key] = val;
}

unordered_set<int> NGraphCatalog::GetEncapOutputIndexesThatNeedCopy(
    int graphid, string node_name) {
  string key = graphid + "_" + node_name;
  return NGraphCatalog::encap_output_copy_indexes_map_[key];
}

bool NGraphCatalog::EncapOutputIndexNeedsCopy(int graphid, string node_name,
                                              int index) {
  string key = graphid + "_" + node_name;
  auto itr = NGraphCatalog::encap_output_copy_indexes_map_.find(key);
  if (itr != NGraphCatalog::encap_output_copy_indexes_map_.end()) {
    auto op_copy_indexes = itr->second;
    return (op_copy_indexes.find(index) != op_copy_indexes.end());
  }
  // Should not reach here
  return true;
}

void NGraphCatalog::DeleteFromEncapOutputCopyIndexesMap(int graphid,
                                                        string node_name) {
  string key = graphid + "_" + node_name;
  NGraphCatalog::encap_output_copy_indexes_map_.erase(key);
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

void NGraphCatalog::DeleteFromInputVariableSharedNameMap(string key) {
  NGraphCatalog::input_variable_sharedname_map_.erase(key);
}

// Functions for EncapOutputInfo Map
void NGraphCatalog::AddToEncapOutputInfoMap(string key,
                                            tuple<string, bool, bool> val) {
  NGraphCatalog::encap_output_info_map_[key] = val;
}

void NGraphCatalog::AddToEncapOutputInfoMap(string key, string shared_name,
                                            bool copy_to_tf,
                                            bool is_tf_just_looking) {
  // create a tuple
  tuple<string, bool, bool> val =
      make_tuple(shared_name, copy_to_tf, is_tf_just_looking);
  NGraphCatalog::encap_output_info_map_[key] = val;
}

bool NGraphCatalog::ExistsInEncapOutputInfoMap(string key) {
  auto itr = NGraphCatalog::encap_output_info_map_.find(key);
  return itr != NGraphCatalog::encap_output_info_map_.end();
}

bool NGraphCatalog::ExistsInEncapOutputInfoMap(int graphid, string node_name,
                                               int input_index) {
  std::string key =
      NGraphCatalog::CreateNodeKey(graphid, node_name, input_index);
  auto itr = NGraphCatalog::encap_output_info_map_.find(key);
  return itr != NGraphCatalog::encap_output_info_map_.end();
}

tuple<string, bool, bool> NGraphCatalog::GetInfoFromEncapOutputInfoMap(
    string key) {
  return NGraphCatalog::encap_output_info_map_[key];
}

string NGraphCatalog::GetVariableSharedNameFromEncapOutputInfoMap(string key) {
  tuple<string, bool, bool> val = NGraphCatalog::encap_output_info_map_[key];
  return get<0>(val);
}

bool NGraphCatalog::GetCopyToTFFromEncapOutputInfoMap(string key) {
  tuple<string, bool, bool> val = NGraphCatalog::encap_output_info_map_[key];
  return get<1>(val);
}

bool NGraphCatalog::GetIsTFJustLookingFromEncapOutputInfoMap(string key) {
  tuple<string, bool, bool> val = NGraphCatalog::encap_output_info_map_[key];
  return get<2>(val);
}

void NGraphCatalog::DeleteFromEncapOutputInfoMap(string key) {
  NGraphCatalog::encap_output_info_map_.erase(key);
}

void NGraphCatalog::ClearEncapOutputInfoMap() {
  NGraphCatalog::encap_output_info_map_.clear();
}

void NGraphCatalog::PrintEncapOutputInfoMap() {
  NGRAPH_VLOG(4) << "EncapOutputInfoMap";
  for (auto it : encap_output_info_map_) {
    NGRAPH_VLOG(4) << "Key: (GraphId_NodeName:OutputIndex) " << it.first
                   << " Value: (shared_name, copy_to_tf, is_tf_just_looking) "
                   << get<0>(it.second) << " " << get<1>(it.second) << " "
                   << get<2>(it.second);
  }
}

}  // ngraph_bridge
}  // tensorflow
