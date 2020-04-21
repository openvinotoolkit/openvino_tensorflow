/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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
#include "gtest/gtest.h"

#include "tensorflow/core/common_runtime/dma_helper.h"

#include "ngraph/ngraph.hpp"

#include "ngraph_bridge/ngraph_catalog.h"
#include "ngraph_bridge/ngraph_tensor_manager.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

class NGraphTensorManagerTest : public ::testing::Test {
 protected:
  // Utility to Simulate entering variable info in catalog
  void EnterVarInCatalog(const int& ng_encap_graph_id,
                         const string& ng_encap_node_name,
                         const vector<int>& var_inp_indexes,
                         const vector<int>& var_out_indexes,
                         const vector<int>& out_indexes_need_copy) {
    for (int index : var_inp_indexes) {
      string key = NGraphCatalog::CreateNodeKey(ng_encap_graph_id,
                                                ng_encap_node_name, index);
      NGraphCatalog::AddToInputVariableSharedNameMap(key, "abc");
    }

    for (int index : var_out_indexes) {
      string key = NGraphCatalog::CreateNodeKey(ng_encap_graph_id,
                                                ng_encap_node_name, index);
      NGraphCatalog::AddToEncapOutputInfoMap(key, make_tuple("abc", true));
    }

    unordered_set<int> indexes_need_copy;
    for (int index : out_indexes_need_copy) {
      indexes_need_copy.insert(index);
    }
    NGraphCatalog::AddToEncapOutputCopyIndexesMap(
        ng_encap_graph_id, ng_encap_node_name, indexes_need_copy);
  }

  // Utility to Simulate entering prefetch info in catalog
  void EnterPrefetchInCatalog(const int& ng_encap_graph_id,
                              const string& ng_encap_node_name,
                              const map<int, int>& prefetched_inp_indexes_map) {
    NGraphCatalog::AddToPrefetchedInputIndexMap(
        ng_encap_graph_id, ng_encap_node_name, prefetched_inp_indexes_map);
  }

  // Clears the Catalog
  void ClearCatalog() { NGraphCatalog::ClearCatalog(); }

  // returns {0,1,2, ... , size-1}
  vector<int> FillRange(int size) {
    vector<int> vout(size);
    iota(vout.begin(), vout.end(), 0);
    return vout;
  }

  // Utility to Simulate entering variable shared name info info in catalog
  void EnterVarSharedInfoInCatalog(
      const int& ng_encap_graph_id, const string& ng_encap_node_name,
      const unordered_map<int, string>& input_var_info_map,
      const unordered_map<int, tuple<string, bool>>& output_var_info_map) {
    for (auto itr : input_var_info_map) {
      string key = NGraphCatalog::CreateNodeKey(ng_encap_graph_id,
                                                ng_encap_node_name, itr.first);
      NGraphCatalog::AddToInputVariableSharedNameMap(key, itr.second);
    }

    for (auto itr : output_var_info_map) {
      string key = NGraphCatalog::CreateNodeKey(ng_encap_graph_id,
                                                ng_encap_node_name, itr.first);
      NGraphCatalog::AddToEncapOutputInfoMap(key, itr.second);
    }
  }
};

TEST(NGraphUtils, FindComplement1) {
  vector<int> input = {0, 3, 5, 8, 9};
  vector<int> complement = FindComplement(10, input);

  vector<int> expected = {1, 2, 4, 6, 7};
  ASSERT_EQ(expected, complement);

  // test 2
  input = {-1, 3, 5};
  complement = FindComplement(5, input);
  expected = {0, 1, 2, 4};
  ASSERT_EQ(expected, complement);
}

// Tests scenario when the graph has no variables
// and no prefetched inputs
TEST_F(NGraphTensorManagerTest, NoVariablesNoPrefetch) {
  string ng_encap_node_name = "xyz_1";
  int ng_encap_cluster_id = 1;
  int ng_encap_graph_id = 1;
  int number_of_inputs = 5;
  int number_of_outputs = 2;

  // expected
  vector<int> empty;
  map<int, int> empty_map;
  vector<int> expected_pipelined_inp_indexes = FillRange(number_of_inputs);
  vector<int> expected_pipelined_out_indexes = FillRange(number_of_outputs);
  vector<int> expected_out_indexes_need_copy = FillRange(number_of_outputs);

  if (ngraph_tf_are_variables_enabled()) {
    EnterVarInCatalog(ng_encap_graph_id, ng_encap_node_name, empty, empty,
                      expected_out_indexes_need_copy);
  }
  NGraphTensorManager tensor_manager(ng_encap_node_name, ng_encap_cluster_id,
                                     ng_encap_graph_id, number_of_inputs,
                                     number_of_outputs);

  // var related
  ASSERT_EQ(empty, tensor_manager.GetInputIndexesFedByVariables());
  ASSERT_EQ(empty, tensor_manager.GetOutputIndexesAssigningVariables());
  ASSERT_EQ(expected_out_indexes_need_copy,
            tensor_manager.GetOutputIndexesThatNeedCopy());
  // pipelined
  ASSERT_EQ(expected_pipelined_inp_indexes,
            tensor_manager.GetPipelinedInputIndexes());
  ASSERT_EQ(expected_pipelined_out_indexes,
            tensor_manager.GetPipelinedOutputIndexes());

  // prefetched
  ASSERT_EQ(empty, tensor_manager.GetPrefetchedInputIndexes());
  ASSERT_EQ(expected_pipelined_inp_indexes,
            tensor_manager.GetPipelinedButNotPrefetchedInputIndexes());

  // prefetched wrt pipelined
  ASSERT_EQ(empty, tensor_manager.GetPipelinedInputIndexesThatArePrefetched());
  ASSERT_EQ(expected_pipelined_inp_indexes,
            tensor_manager.GetPipelinedInputIndexesThatAreNotPrefetched());

  ASSERT_EQ(empty_map, tensor_manager.GetInputIndexesForPrefetchSharedObject());
  // clean up
  ClearCatalog();
}

// Tests scenario when the graph has variables but no prefetched inputs
//   1. For Var build: catalog is populated
//   2. For others: catalog is not populated
TEST_F(NGraphTensorManagerTest, HasVariablesNoPrefetch) {
  string ng_encap_node_name = "xyz_1";
  int ng_encap_cluster_id = 1;
  int ng_encap_graph_id = 1;
  int number_of_inputs = 5;
  int number_of_outputs = 2;

  // expected
  vector<int> expected_pipelined_inp_indexes, expected_pipelined_out_indexes,
      expected_var_inp_indexes, expected_var_out_indexes,
      expected_out_indexes_need_copy, expected_prefetched_inp_indexes,
      expected_pipelined_not_prefetched_input_indexes,
      expected_pipelined_input_indexes_prefetched,
      expected_pipelined_input_indexes_not_prefetched;
  map<int, int> expected_prefetch_indexes_map;

  // expected values
  if (ngraph_tf_are_variables_enabled()) {
    // pipelined
    expected_pipelined_inp_indexes = {1, 3, 4};
    expected_pipelined_out_indexes = {1};
    // var
    expected_var_inp_indexes =
        FindComplement(number_of_inputs, expected_pipelined_inp_indexes);
    expected_var_out_indexes =
        FindComplement(number_of_outputs, expected_pipelined_out_indexes);
    expected_out_indexes_need_copy = {1};

    // prefetched
    expected_prefetched_inp_indexes = {};
    expected_pipelined_not_prefetched_input_indexes =
        expected_pipelined_inp_indexes;

    // prefetched relative to pipelined tensors
    expected_pipelined_input_indexes_prefetched = {};
    expected_pipelined_input_indexes_not_prefetched = {0, 1, 2};

    expected_prefetch_indexes_map = {};

    // enter in catalog
    EnterVarInCatalog(ng_encap_graph_id, ng_encap_node_name,
                      expected_var_inp_indexes, expected_var_out_indexes,
                      expected_out_indexes_need_copy);

  } else {
    // pipelined
    expected_pipelined_inp_indexes = FillRange(number_of_inputs);
    expected_pipelined_out_indexes = FillRange(number_of_outputs);
    // var
    expected_var_inp_indexes = {};
    expected_var_out_indexes = {};
    expected_out_indexes_need_copy = FillRange(number_of_outputs);
    // prefetched
    expected_prefetched_inp_indexes = {};
    expected_pipelined_not_prefetched_input_indexes =
        expected_pipelined_inp_indexes;

    // prefetched relative to pipelined tensors
    expected_pipelined_input_indexes_prefetched = {};
    expected_pipelined_input_indexes_not_prefetched =
        expected_pipelined_not_prefetched_input_indexes;
    expected_prefetch_indexes_map = {};
  }

  NGraphTensorManager tensor_manager(ng_encap_node_name, ng_encap_cluster_id,
                                     ng_encap_graph_id, number_of_inputs,
                                     number_of_outputs);

  // var
  ASSERT_EQ(expected_var_inp_indexes,
            tensor_manager.GetInputIndexesFedByVariables());
  ASSERT_EQ(expected_var_out_indexes,
            tensor_manager.GetOutputIndexesAssigningVariables());
  ASSERT_EQ(expected_out_indexes_need_copy,
            tensor_manager.GetOutputIndexesThatNeedCopy());

  // pipelined
  ASSERT_EQ(expected_pipelined_inp_indexes,
            tensor_manager.GetPipelinedInputIndexes());
  ASSERT_EQ(expected_pipelined_out_indexes,
            tensor_manager.GetPipelinedOutputIndexes());

  // prefetched
  ASSERT_EQ(expected_prefetched_inp_indexes,
            tensor_manager.GetPrefetchedInputIndexes());
  ASSERT_EQ(expected_pipelined_not_prefetched_input_indexes,
            tensor_manager.GetPipelinedButNotPrefetchedInputIndexes());

  // prefetched wrt pipelined
  ASSERT_EQ(expected_pipelined_input_indexes_prefetched,
            tensor_manager.GetPipelinedInputIndexesThatArePrefetched());
  ASSERT_EQ(expected_pipelined_input_indexes_not_prefetched,
            tensor_manager.GetPipelinedInputIndexesThatAreNotPrefetched());

  ASSERT_EQ(expected_prefetch_indexes_map,
            tensor_manager.GetInputIndexesForPrefetchSharedObject());
  // clean up
  ClearCatalog();
}

// Tests scenario when the graph has no variables
// but has prefetched inputs
TEST_F(NGraphTensorManagerTest, NoVariablesHasPrefetch) {
  string ng_encap_node_name = "xyz_1";
  int ng_encap_cluster_id = 1;
  int ng_encap_graph_id = 1;
  int number_of_inputs = 5;
  int number_of_outputs = 2;

  // expected
  // var
  vector<int> empty;
  vector<int> expected_out_indexes_need_copy = FillRange(number_of_outputs);

  // pipelined
  vector<int> expected_pipelined_inp_indexes = FillRange(number_of_inputs);
  vector<int> expected_pipelined_out_indexes = FillRange(number_of_outputs);

  // prefetched
  vector<int> expected_prefetched_inp_indexes = {1, 3};
  vector<int> expected_pipelined_not_prefetched_input_indexes = {0, 2, 4};

  // relative to pipelined tensors
  // all pipelined are prefetched
  vector<int> expected_pipelined_input_indexes_prefetched =
      expected_prefetched_inp_indexes;
  vector<int> expected_pipelined_input_indexes_not_prefetched =
      expected_pipelined_not_prefetched_input_indexes;

  map<int, int> expected_prefetch_indexes_map = {{1, 3}, {3, 6}};

  if (ngraph_tf_are_variables_enabled()) {
    EnterVarInCatalog(ng_encap_graph_id, ng_encap_node_name, empty, empty,
                      expected_out_indexes_need_copy);
  }

  EnterPrefetchInCatalog(ng_encap_graph_id, ng_encap_node_name,
                         expected_prefetch_indexes_map);

  NGraphTensorManager tensor_manager(ng_encap_node_name, ng_encap_cluster_id,
                                     ng_encap_graph_id, number_of_inputs,
                                     number_of_outputs);

  // var related
  ASSERT_EQ(empty, tensor_manager.GetInputIndexesFedByVariables());
  ASSERT_EQ(empty, tensor_manager.GetOutputIndexesAssigningVariables());
  ASSERT_EQ(expected_out_indexes_need_copy,
            tensor_manager.GetOutputIndexesThatNeedCopy());
  // pipelined
  ASSERT_EQ(expected_pipelined_inp_indexes,
            tensor_manager.GetPipelinedInputIndexes());
  ASSERT_EQ(expected_pipelined_out_indexes,
            tensor_manager.GetPipelinedOutputIndexes());

  // prefetched
  ASSERT_EQ(expected_prefetched_inp_indexes,
            tensor_manager.GetPrefetchedInputIndexes());
  ASSERT_EQ(expected_pipelined_not_prefetched_input_indexes,
            tensor_manager.GetPipelinedButNotPrefetchedInputIndexes());

  // prefetched wrt pipelined
  ASSERT_EQ(expected_pipelined_input_indexes_prefetched,
            tensor_manager.GetPipelinedInputIndexesThatArePrefetched());
  ASSERT_EQ(expected_pipelined_input_indexes_not_prefetched,
            tensor_manager.GetPipelinedInputIndexesThatAreNotPrefetched());

  ASSERT_EQ(expected_prefetch_indexes_map,
            tensor_manager.GetInputIndexesForPrefetchSharedObject());

  // clean up
  ClearCatalog();
}

// Tests scenario when the graph has variables and prefetched inputs
TEST_F(NGraphTensorManagerTest, VariablesAndPrefetch) {
  string ng_encap_node_name = "xyz_1";
  int ng_encap_cluster_id = 1;
  int ng_encap_graph_id = 1;
  int number_of_inputs = 7;
  int number_of_outputs = 4;

  // expected
  vector<int> expected_pipelined_inp_indexes, expected_pipelined_out_indexes,
      expected_var_inp_indexes, expected_var_out_indexes,
      expected_out_indexes_need_copy, expected_prefetched_inp_indexes,
      expected_pipelined_not_prefetched_input_indexes,
      expected_pipelined_inp_indexes_prefetched,
      expected_pipelined_inp_indexes_not_prefetched;

  map<int, int> input_prefetch_indexes_map, expected_prefetch_indexes_map;

  if (ngraph_tf_are_variables_enabled()) {
    // expected values
    // pipelined
    expected_pipelined_inp_indexes = {1, 3, 4, 6};
    expected_pipelined_out_indexes = {0, 2};
    // var
    expected_var_inp_indexes =
        FindComplement(number_of_inputs, expected_pipelined_inp_indexes);
    expected_var_out_indexes =
        FindComplement(number_of_outputs, expected_pipelined_out_indexes);
    expected_out_indexes_need_copy = {2, 3};

    // prefetched
    expected_prefetched_inp_indexes = {3, 6};
    expected_pipelined_not_prefetched_input_indexes = {1, 4};

    expected_pipelined_inp_indexes_prefetched = {1, 3};
    expected_pipelined_inp_indexes_not_prefetched = {0, 2};

    input_prefetch_indexes_map = {{3, 3}, {6, 8}};
    expected_prefetch_indexes_map = {{1, 3}, {3, 8}};

    // enter in catalog
    EnterVarInCatalog(ng_encap_graph_id, ng_encap_node_name,
                      expected_var_inp_indexes, expected_var_out_indexes,
                      expected_out_indexes_need_copy);

  } else {
    // pipelined
    expected_pipelined_inp_indexes = FillRange(number_of_inputs);
    expected_pipelined_out_indexes = FillRange(number_of_outputs);

    // var
    expected_var_inp_indexes = {};
    expected_var_out_indexes = {};
    expected_out_indexes_need_copy = FillRange(number_of_outputs);

    // prefetched
    expected_prefetched_inp_indexes = {3, 6};
    expected_pipelined_not_prefetched_input_indexes = {0, 1, 2, 4, 5};

    // prefetched wrt to pipelining
    expected_pipelined_inp_indexes_prefetched =
        expected_prefetched_inp_indexes;  // all inputs are pipelined
    expected_pipelined_inp_indexes_not_prefetched =
        expected_pipelined_not_prefetched_input_indexes;

    input_prefetch_indexes_map = {{3, 3}, {6, 8}};
    expected_prefetch_indexes_map = {{3, 3}, {6, 8}};
  }

  EnterPrefetchInCatalog(ng_encap_graph_id, ng_encap_node_name,
                         input_prefetch_indexes_map);

  NGraphTensorManager tensor_manager(ng_encap_node_name, ng_encap_cluster_id,
                                     ng_encap_graph_id, number_of_inputs,
                                     number_of_outputs);

  // var related
  ASSERT_EQ(expected_var_inp_indexes,
            tensor_manager.GetInputIndexesFedByVariables());
  ASSERT_EQ(expected_var_out_indexes,
            tensor_manager.GetOutputIndexesAssigningVariables());
  ASSERT_EQ(expected_out_indexes_need_copy,
            tensor_manager.GetOutputIndexesThatNeedCopy());
  // pipelined
  ASSERT_EQ(expected_pipelined_inp_indexes,
            tensor_manager.GetPipelinedInputIndexes());
  ASSERT_EQ(expected_pipelined_out_indexes,
            tensor_manager.GetPipelinedOutputIndexes());

  // prefetched
  ASSERT_EQ(expected_prefetched_inp_indexes,
            tensor_manager.GetPrefetchedInputIndexes());
  ASSERT_EQ(expected_pipelined_not_prefetched_input_indexes,
            tensor_manager.GetPipelinedButNotPrefetchedInputIndexes());

  // prefetched wrt pipelined
  ASSERT_EQ(expected_pipelined_inp_indexes_prefetched,
            tensor_manager.GetPipelinedInputIndexesThatArePrefetched());
  ASSERT_EQ(expected_pipelined_inp_indexes_not_prefetched,
            tensor_manager.GetPipelinedInputIndexesThatAreNotPrefetched());

  ASSERT_EQ(expected_prefetch_indexes_map,
            tensor_manager.GetInputIndexesForPrefetchSharedObject());

  // clean up
  ClearCatalog();
}

// check error
TEST_F(NGraphTensorManagerTest, PrefetchNotInPipeline) {
  string ng_encap_node_name = "xyz_1";
  int ng_encap_cluster_id = 1;
  int ng_encap_graph_id = 1;
  int number_of_inputs = 5;
  int number_of_outputs = 2;

  map<int, int> prefetched_inp_indexe_map = {{6, 7}, {6, 8}};
  EnterPrefetchInCatalog(ng_encap_graph_id, ng_encap_node_name,
                         prefetched_inp_indexe_map);

  ASSERT_THROW(NGraphTensorManager tensor_manager(
                   ng_encap_node_name, ng_encap_cluster_id, ng_encap_graph_id,
                   number_of_inputs, number_of_outputs),
               std::runtime_error);

  // clean up
  ClearCatalog();
}

// check book-keeping of shared information
TEST_F(NGraphTensorManagerTest, SharedName) {
  string ng_encap_node_name = "xyz_1";
  int ng_encap_cluster_id = 1;
  int ng_encap_graph_id = 1;
  int number_of_inputs = 5;
  int number_of_outputs = 6;

  unordered_map<int, string> input_var_info_map = {{0, "A"}, {3, "C"}};
  unordered_map<int, tuple<string, bool>> output_var_info_map = {
      {1, make_tuple("X", false)},
      {5, make_tuple("Y", true)},
      {0, make_tuple("Z", false)}};

  EnterVarSharedInfoInCatalog(ng_encap_graph_id, ng_encap_node_name,
                              input_var_info_map, output_var_info_map);

  NGraphTensorManager tensor_manager(ng_encap_node_name, ng_encap_cluster_id,
                                     ng_encap_graph_id, number_of_inputs,
                                     number_of_outputs);

  if (ngraph_tf_are_variables_enabled()) {
    string shared_name;
    bool copy_to_tf;
    // input var
    ASSERT_OK(tensor_manager.GetInputVariableSharedName(0, &shared_name));
    ASSERT_EQ(shared_name, "A");
    ASSERT_OK(tensor_manager.GetInputVariableSharedName(3, &shared_name));
    ASSERT_EQ(shared_name, "C");

    ASSERT_NOT_OK(tensor_manager.GetInputVariableSharedName(2, &shared_name));

    // output var
    ASSERT_OK(tensor_manager.GetOutputVariableSharedName(1, &shared_name));
    ASSERT_EQ(shared_name, "X");
    ASSERT_OK(tensor_manager.GetOutputVariableSharedName(5, &shared_name));
    ASSERT_EQ(shared_name, "Y");

    ASSERT_NOT_OK(tensor_manager.GetOutputVariableSharedName(2, &shared_name));
    ASSERT_OK(tensor_manager.GetOutputVariableSharedName(0, &shared_name));
    ASSERT_EQ(shared_name, "Z");

    // output var copy_to_tf
    ASSERT_OK(tensor_manager.GetOutputVariableUpdateTFTensor(1, &copy_to_tf));
    ASSERT_FALSE(copy_to_tf);
    ASSERT_OK(tensor_manager.GetOutputVariableUpdateTFTensor(5, &copy_to_tf));
    ASSERT_TRUE(copy_to_tf);

    ASSERT_NOT_OK(
        tensor_manager.GetOutputVariableUpdateTFTensor(2, &copy_to_tf));

    ASSERT_OK(tensor_manager.GetOutputVariableUpdateTFTensor(0, &copy_to_tf));
    ASSERT_FALSE(copy_to_tf);

  } else {
    string shared_name;
    bool copy_to_tf;
    // input var
    ASSERT_NOT_OK(tensor_manager.GetInputVariableSharedName(0, &shared_name));
    ASSERT_NOT_OK(tensor_manager.GetInputVariableSharedName(3, &shared_name));
    ASSERT_NOT_OK(tensor_manager.GetInputVariableSharedName(2, &shared_name));

    // output var
    ASSERT_NOT_OK(tensor_manager.GetOutputVariableSharedName(1, &shared_name));
    ASSERT_NOT_OK(tensor_manager.GetOutputVariableSharedName(5, &shared_name));
    ASSERT_NOT_OK(tensor_manager.GetOutputVariableSharedName(2, &shared_name));

    // output var copy_to_tf
    ASSERT_NOT_OK(
        tensor_manager.GetOutputVariableUpdateTFTensor(1, &copy_to_tf));
    ASSERT_NOT_OK(
        tensor_manager.GetOutputVariableUpdateTFTensor(5, &copy_to_tf));
    ASSERT_NOT_OK(
        tensor_manager.GetOutputVariableUpdateTFTensor(2, &copy_to_tf));
  }

  // clean up
  ClearCatalog();
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow