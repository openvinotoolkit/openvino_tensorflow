/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include "ngraph_bridge/ngraph_catalog.h"
#include "ngraph_bridge/ngraph_tensor_manager.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"
#include "test/test_utilities.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

class NGraphTensorManagerTest : public ::testing::Test {
 protected:
  // Utility to Simulate entering in catalog
  void EnterInCatalog(const int& ng_encap_graph_id,
                      const string& ng_encap_node_name,
                      const vector<int>& var_inp_indexes,
                      const vector<int>& var_out_indexes,
                      const vector<int>& out_indexes_need_copy) {
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
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
#endif
  }

  // Clears the Catalog
  void ClearCatalog() {
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
    NGraphCatalog::ClearCatalog();
#endif
  }

  // returns {0,1,2, ... , size-1}
  vector<int> FillRange(int size) {
    vector<int> vout(size);
    iota(vout.begin(), vout.end(), 0);
    return vout;
  }
};

TEST(NGraphUtils, FindComplement1) {
  bool yes;
  Status st = IsNgraphTFLogTensorCopiesEnabled(0, yes);

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
TEST_F(NGraphTensorManagerTest, NoVariables) {
  string ng_encap_node_name = "xyz_1";
  int ng_encap_cluster_id = 1;
  int ng_encap_graph_id = 1;
  int number_of_inputs = 5;
  int number_of_outputs = 2;

  NGraphTensorManager tensor_manager(ng_encap_node_name, ng_encap_cluster_id,
                                     ng_encap_graph_id, number_of_inputs,
                                     number_of_outputs);
  // expected
  vector<int> empty;
  vector<int> expected_pipelined_inp_indexes = FillRange(number_of_inputs);
  vector<int> expected_pipelined_out_indexes = FillRange(number_of_outputs);

  // var related
  ASSERT_EQ(empty, tensor_manager.GetInputIndexesFedByVariables());
  ASSERT_EQ(empty, tensor_manager.GetOutputIndexesAssigningVariables());
  ASSERT_EQ(empty, tensor_manager.GetOutputIndexesThatNeedCopy());
  ASSERT_EQ(expected_pipelined_inp_indexes,
            tensor_manager.GetPipelinedInputIndexes());
  ASSERT_EQ(expected_pipelined_out_indexes,
            tensor_manager.GetPipelinedOutputIndexes());

  // prefetched
  ASSERT_EQ(empty, tensor_manager.GetPrefetchedInputIndexes());
}

// Tests scenario when the graph has variables
//   1. For Var build: catalog is populated
//   2. For others: no notion of catalog
TEST_F(NGraphTensorManagerTest, Variables) {
  string ng_encap_node_name = "xyz_1";
  int ng_encap_cluster_id = 1;
  int ng_encap_graph_id = 1;
  int number_of_inputs = 5;
  int number_of_outputs = 2;

  // expected
  vector<int> expected_pipelined_inp_indexes, expected_pipelined_out_indexes,
      expected_var_inp_indexes, expected_var_out_indexes,
      expected_out_indexes_need_copy, expected_prefetched_inp_indexes;

  if (ngraph_tf_are_variables_enabled()) {
    // expected values
    expected_pipelined_inp_indexes = {1, 3, 4};
    expected_pipelined_out_indexes = {1};
    expected_var_inp_indexes =
        FindComplement(number_of_inputs, expected_pipelined_inp_indexes);
    expected_var_out_indexes =
        FindComplement(number_of_outputs, expected_pipelined_out_indexes);
    expected_out_indexes_need_copy = {1};
    expected_prefetched_inp_indexes = {};

    // enter in catalog
    EnterInCatalog(ng_encap_graph_id, ng_encap_node_name,
                   expected_var_inp_indexes, expected_var_out_indexes,
                   expected_out_indexes_need_copy);

  } else {
    expected_pipelined_inp_indexes = FillRange(number_of_inputs);
    expected_pipelined_out_indexes = FillRange(number_of_outputs);

    expected_var_inp_indexes = {};
    expected_var_out_indexes = {};
    expected_out_indexes_need_copy = {};
    expected_prefetched_inp_indexes = {};
  }

  NGraphTensorManager tensor_manager(ng_encap_node_name, ng_encap_cluster_id,
                                     ng_encap_graph_id, number_of_inputs,
                                     number_of_outputs);

  ASSERT_EQ(expected_var_inp_indexes,
            tensor_manager.GetInputIndexesFedByVariables());
  ASSERT_EQ(expected_var_out_indexes,
            tensor_manager.GetOutputIndexesAssigningVariables());
  ASSERT_EQ(expected_out_indexes_need_copy,
            tensor_manager.GetOutputIndexesThatNeedCopy());

  ASSERT_EQ(expected_prefetched_inp_indexes,
            tensor_manager.GetPrefetchedInputIndexes());

  ASSERT_EQ(expected_pipelined_inp_indexes,
            tensor_manager.GetPipelinedInputIndexes());
  ASSERT_EQ(expected_pipelined_out_indexes,
            tensor_manager.GetPipelinedOutputIndexes());

  // var related
  if (ngraph_tf_are_variables_enabled()) {
    ClearCatalog();
  }
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow