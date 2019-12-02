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

#ifndef NGRAPH_TF_CATALOG_H_
#define NGRAPH_TF_CATALOG_H_

#include <atomic>
#include <mutex>
#include <ostream>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend_manager.hpp"

#include "logging/ngraph_log.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

class NGraphCatalog {
 private:
  // Map keeps track of nodes whose input is a variable tensor
  // Will be used by Assign/Optimizers and NGraphEncapsulate Op
  // Map of
  // Key
  //   when op index ==0
  //      string : GraphId + _ + nodename
  //   otherwise
  //     string : GraphId + _ + nodename + : + input_index
  // Value : variable shared_name
  // LOCK?
  static unordered_map<string, string> input_variable_sharedname_map_;

  // Map keeps track of output indexes of NGraphEncapsulate Op
  // that will be used by TF Nodes or other NGraphEncapsulate Op
  // Will be used by NGraphEncapsulateOP
  // Map of
  // Key
  //  string : GraphId + _ + nodename
  // Value : Set of indices
  static unordered_map<string, unordered_set<int>>
      encap_output_copy_indexes_map_;

  // Map keeps track of NGraphAssigns whose value to be assigned
  // has been computed by NGraph and will be eliminated from the
  // graph.
  // Map of
  // Key
  //  string : representing encap_output_index
  //   when op index == 0
  //      string : GraphId + _ + nodename
  //   otherwise
  //     string : GraphId + _ + nodename + : + output_index
  // Value : 3 element tuple
  //  string : NGraphAssign‘s variable shared_name
  //  bool : NGraphAssign‘s copy_to_tf attribute ‘s value
  static unordered_map<string, tuple<string, bool>> encap_output_info_map_;

  // Map keeps track of encap nodes whose input is an IteratorGenNext.
  // This is map from the node to the input indexs of the
  // encapsulate node that are prefetched.
  // Will be used by NGraphEncapsulate Op.
  // Map of
  // Key
  //      string : GraphId + _ + nodename
  // Value : Set of indices
  static unordered_map<string, unordered_set<int>> prefetched_input_index_map_;

 public:
  // Utility to create key to query the maps
  static string CreateNodeKey(const int& graph_id, const string& node_name,
                              const int& index);
  static string CreateNodeKey(const int& graph_id, const string& node_name);

  // Clear all the maps
  static void ClearCatalog();

  // Utility Functions for the data structures
  // Functions for EncapsulateOutputCopyIndexes Map
  static void AddToEncapOutputCopyIndexesMap(const int& graphid,
                                             const string& node_name,
                                             const unordered_set<int>& val);

  static void ClearEncapOutputCopyIndexesMap();

  static bool EncapOutputNeedsCopy(const int& graphid, const string& node_name);

  static bool EncapOutputIndexNeedsCopy(const int& graphid,
                                        const string& node_name,
                                        const int& index);
  static const unordered_set<int>& GetEncapOutputIndexesThatNeedCopy(
      const int& graphid, const string& node_name);
  static void DeleteFromEncapOutputCopyIndexesMap(const int& graphid,
                                                  const string& node_name);

  // Functions for InputVariableSharedName Map
  static void AddToInputVariableSharedNameMap(const string& key,
                                              const string& val);

  static void ClearInputVariableSharedNameMap();
  static const string& GetInputVariableSharedName(const int& graphid,
                                                  const string& node_name,
                                                  const int& input_index);
  static bool ExistsInInputVariableSharedNameMap(const string& key);
  static bool ExistsInInputVariableSharedNameMap(const int& graphid,
                                                 const string& node_name,
                                                 const int& input_index);
  static void DeleteFromInputVariableSharedNameMap(const string& key);

  // Functions for EncapOutputInfo Map
  static void AddToEncapOutputInfoMap(const string& key,
                                      const tuple<string, bool>& val);
  static void AddToEncapOutputInfoMap(const string& key,
                                      const string& shared_name,
                                      const bool& copy_to_tf);
  static bool ExistsInEncapOutputInfoMap(const string& key);
  static bool ExistsInEncapOutputInfoMap(const int& graphid,
                                         const string& node_name,
                                         const int& output_index);
  static const tuple<string, bool>& GetInfoFromEncapOutputInfoMap(
      const string& key);
  static const string& GetVariableSharedNameFromEncapOutputInfoMap(
      const string& key);
  static const bool& GetCopyToTFFromEncapOutputInfoMap(const string& key);
  static void DeleteFromEncapOutputInfoMap(const string& key);
  static void ClearEncapOutputInfoMap();
  static void PrintEncapOutputInfoMap();

  // Functions for PrefetedInputs Map
  static void AddToPrefetchedInputIndexMap(const int& graphid,
                                           const string& node_name,
                                           const unordered_set<int>& val);
  static bool ExistsInPrefetchedInputIndexMap(const int& graphid,
                                              const string& node_name);
  static bool ExistsInPrefetchedInputIndexMap(const string& key);
  static const unordered_set<int>& GetIndexesFromPrefetchedInputIndexMap(
      const int& graphid, const string& node_name);

  static void ClearPrefetchedInputIndexMap();
  static void PrintPrefetchedInputIndexMap();
};

}  // ngraph_bridge
}  // tensorflow

#endif
