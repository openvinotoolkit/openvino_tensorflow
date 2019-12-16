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

#include "gtest/gtest.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_capture_variables.h"
#include "ngraph_bridge/ngraph_catalog.h"
#include "ngraph_bridge/ngraph_cluster_manager.h"
#include "ngraph_bridge/ngraph_deassign_clusters.h"
#include "ngraph_bridge/ngraph_encapsulate_clusters.h"
#include "ngraph_bridge/ngraph_enter_prefetch_in_catalog.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_rewrite_for_tracking.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {
TEST(PrefetchCatalogTest, SmallGraph1) {
  // Set flag to enable prefetch
  list<string> env_vars{"NGRAPH_TF_USE_PREFETCH"};
  const unordered_map<string, string>& env_map = StoreEnv(env_vars);
  SetEnvVariable("NGRAPH_TF_USE_PREFETCH", "1");

  // Create Graph
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  Graph input_graph(OpRegistry::Global());

  // Now read the graph
  // test_catalog_for_prefetch.pbtxt was created by running the following
  // command:
  // NGRAPH_TF_DUMP_GRAPHS=1 python examples/axpy_pipelined.py
  // and using the encapsulated_0003.pbtxt
  ASSERT_OK(
      LoadGraphFromPbTxt("test_catalog_for_prefetch.pbtxt", &input_graph));

  ASSERT_OK(EnterPrefetchInCatalog(&input_graph, 0));
  ASSERT_TRUE(
      NGraphCatalog::ExistsInPrefetchedInputIndexMap("0_ngraph_cluster_4"));
  ASSERT_TRUE(
      NGraphCatalog::ExistsInPrefetchedInputIndexMap(0, "ngraph_cluster_4"));
  std::unordered_set<int> expected;
  expected.insert(0);
  std::unordered_set<int> indexes;
  indexes = NGraphCatalog::GetIndexesFromPrefetchedInputIndexMap(
      0, "ngraph_cluster_4");
  ASSERT_EQ(indexes, expected);

  // Clean up
  NGraphCatalog::ClearCatalog();
  // Unset, Restore env flga
  UnsetEnvVariable("NGRAPH_TF_USE_PREFETCH");
  RestoreEnv(env_map);
}

TEST(PrefetchCatalogTest, SmallGraph2) {
  // Set flag to enable prefetch
  list<string> env_vars{"NGRAPH_TF_USE_PREFETCH"};
  const unordered_map<string, string>& env_map = StoreEnv(env_vars);
  SetEnvVariable("NGRAPH_TF_USE_PREFETCH", "1");

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  Graph input_graph(OpRegistry::Global());

  // Now read the graph
  ASSERT_OK(
      LoadGraphFromPbTxt("test_catalog_for_prefetch_1.pbtxt", &input_graph));

  ASSERT_OK(EnterPrefetchInCatalog(&input_graph, 0));
  ASSERT_TRUE(
      NGraphCatalog::ExistsInPrefetchedInputIndexMap("0_ngraph_cluster_340"));
  ASSERT_TRUE(
      NGraphCatalog::ExistsInPrefetchedInputIndexMap(0, "ngraph_cluster_340"));
  std::unordered_set<int> expected;
  expected.insert(2);
  expected.insert(3);
  std::unordered_set<int> indexes;
  indexes = NGraphCatalog::GetIndexesFromPrefetchedInputIndexMap(
      0, "ngraph_cluster_340");
  ASSERT_EQ(indexes, expected);

  // Clean up
  NGraphCatalog::ClearCatalog();
  // Unset, restore env flags
  UnsetEnvVariable("NGRAPH_TF_USE_PREFETCH");
  RestoreEnv(env_map);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
