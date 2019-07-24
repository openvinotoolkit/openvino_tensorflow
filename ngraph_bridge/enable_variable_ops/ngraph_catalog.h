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
  //  bool : NGraphAssign‘s is_tf_just_looking_
  static unordered_map<string, tuple<string, bool, bool>>
      encap_output_info_map_;

 public:
  // Utility to create key to query the maps
  static string CreateNodeKey(int graph_id, string node_name, int index);

  // Utility Functions for the data structures
  // Functions for EncapsulateOutputCopyIndexes Map
  static void AddToEncapOutputCopyIndexesMap(int graphid, string node_name,
                                             unordered_set<int> val);
  static bool EncapOutputIndexNeedsCopy(int graphid, string node_name,
                                        int index);
  static unordered_set<int> GetEncapOutputIndexesThatNeedCopy(int graphid,
                                                              string node_name);
  static void DeleteFromEncapOutputCopyIndexesMap(int graphid,
                                                  string node_name);

  // Functions for InputVariableSharedName Map
  static string GetInputVariableSharedName(int graphid, string node_name,
                                           int input_index);

  static void AddToInputVariableSharedNameMap(string key, string val);

  static bool ExistsInInputVariableSharedNameMap(string key);
  static bool ExistsInInputVariableSharedNameMap(int graphid, string node_name,
                                                 int input_index);
  static void DeleteFromInputVariableSharedNameMap(string key);

  // Functions for EncapOutputInfo Map
  static void AddToEncapOutputInfoMap(string key,
                                      tuple<string, bool, bool> val);
  static void AddToEncapOutputInfoMap(string key, string shared_name,
                                      bool copy_to_tf, bool is_tf_just_looking);
  static bool ExistsInEncapOutputInfoMap(string key);
  static bool ExistsInEncapOutputInfoMap(int graphid, string node_name,
                                         int input_index);
  static tuple<string, bool, bool> GetInfoFromEncapOutputInfoMap(string key);
  static string GetVariableSharedNameFromEncapOutputInfoMap(string key);
  static bool GetCopyToTFFromEncapOutputInfoMap(string key);
  static bool GetIsTFJustLookingFromEncapOutputInfoMap(string key);
  static void DeleteFromEncapOutputInfoMap(string key);
  static void ClearEncapOutputInfoMap();
  static void PrintEncapOutputInfoMap();
};

}  // ngraph_bridge
}  // tensorflow

#endif
