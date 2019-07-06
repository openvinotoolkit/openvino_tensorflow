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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "../test_utilities.h"
#include "ngraph_assign_clusters.h"
#include "ngraph_backend_manager.h"
#include "ngraph_mark_for_clustering.h"
#include "tf_graph_writer.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

/*
These tests test the Backend Handling by the bridge
Since the backend is set globaly the setting of NGRAPH_TF_BACKEND might cause
some tests to fail
For e.g. "BackendCLustering" test would fail If the NGRAPH_TF_BACKEND is set to
INTERPRETER the nodes going to CPU also go to interpreter
putting all the nodes in the same cluster.
*/

// Test SetBackendAPI
TEST(BackendManager, SetBackend) {
  ASSERT_OK(BackendManager::SetBackendName("CPU"));
  string cpu_backend = BackendManager::GetCurrentlySetBackendName();
  ASSERT_EQ(cpu_backend, "CPU");

  ASSERT_OK(BackendManager::SetBackendName("INTERPRETER"));
  string current_backend = BackendManager::GetCurrentlySetBackendName();
  ASSERT_EQ(current_backend, "INTERPRETER");

  ASSERT_NOT_OK(BackendManager::SetBackendName("temp"));
  // Setting again as clean up
  ASSERT_OK(BackendManager::SetBackendName("CPU"));
}

// Test GetSupportedBackendNames
TEST(BackendManager, GetSupportedBackendNames) {
  unordered_set<string> ng_tf_backends =
      BackendManager::GetSupportedBackendNames();

  NGRAPH_VLOG(5) << "Supported Backends";
  for (auto backend : ng_tf_backends) {
    NGRAPH_VLOG(5) << backend;
  }

  vector<string> ng_backends =
      ng::runtime::BackendManager::get_registered_backends();
  ASSERT_EQ(ng_tf_backends.size(), ng_backends.size());

  for (auto backend : ng_backends) {
    auto itr = ng_tf_backends.find(backend);
    ASSERT_NE(itr, ng_tf_backends.end());
  }
}

// Test Backend Assignment
TEST(BackendManager, BackendAssignment) {
  ASSERT_OK(BackendManager::SetBackendName("CPU"));
  Scope root = Scope::NewRootScope();
  auto A = ops::Const(root.WithOpName("A"), {1.0f, 1.0f});
  auto B = ops::Const(root.WithOpName("B"), {1.0f, 1.0f});
  auto R = ops::Add(root.WithOpName("R"), A, B);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  std::set<string> skip_these_nodes = {};

  // Set backend 1
  string backend1 = "INTERPRETER";
  ASSERT_OK(BackendManager::SetBackendName(backend1));
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, backend1));
  std::map<std::string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  string bA, bB, bR;
  ASSERT_OK(GetNodeBackend(node_map["A"], &bA));
  ASSERT_OK(GetNodeBackend(node_map["B"], &bB));
  ASSERT_OK(GetNodeBackend(node_map["R"], &bR));

  ASSERT_EQ(bA, bB);
  ASSERT_EQ(bA, bR);
  ASSERT_EQ(bA, backend1);

  // Set backend 2
  string backend2 = "CPU";
  ASSERT_OK(BackendManager::SetBackendName(backend2));
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, backend2));

  ASSERT_OK(GetNodeBackend(node_map["A"], &bA));
  ASSERT_OK(GetNodeBackend(node_map["B"], &bB));
  ASSERT_OK(GetNodeBackend(node_map["R"], &bR));

  ASSERT_EQ(bA, bB);
  ASSERT_EQ(bA, bR);
  ASSERT_EQ(bA, backend2);
}

// Test Backend Clustering
TEST(BackendManager, BackendClustering) {
  ASSERT_OK(BackendManager::SetBackendName("CPU"));
  Scope root = Scope::NewRootScope();
  auto A = ops::Const(root.WithOpName("A"), {1.0f, 1.0f});
  auto B = ops::Const(root.WithOpName("B"), {1.0f, 1.0f});
  auto R = ops::Add(root.WithOpName("R"), A, B);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  std::set<string> skip_these_nodes = {};

  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, "CPU"));

  string backend1 = "INTERPRETER";

  std::map<std::string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  SetNodeBackend(node_map["B"], backend1);
  ASSERT_OK(AssignClusters(&graph));

  int A_cluster, B_cluster, R_cluster;
  ASSERT_OK(GetNodeCluster(node_map["A"], &A_cluster));
  ASSERT_OK(GetNodeCluster(node_map["B"], &B_cluster));
  ASSERT_OK(GetNodeCluster(node_map["R"], &R_cluster));

  ASSERT_EQ(A_cluster, R_cluster);
  ASSERT_NE(A_cluster, B_cluster);
}

// Test GetBackendAdditionalAttributes API
TEST(BackendManager, GetBackendAdditionalAttributes) {
  vector<string> default_backend_optional_attrs = {"device_config"};
  vector<string> nnpi_backend_optional_attrs = {"device_id", "ice_cores",
                                                "max_batch_size"};

  auto cpu_options = BackendManager::GetBackendAdditionalAttributes("CPU");
  auto nnpi_options = BackendManager::GetBackendAdditionalAttributes("NNPI");
  auto gpu_options = BackendManager::GetBackendAdditionalAttributes("GPU");

  ASSERT_EQ(cpu_options, default_backend_optional_attrs);
  ASSERT_EQ(nnpi_options, nnpi_backend_optional_attrs);
  ASSERT_EQ(gpu_options, default_backend_optional_attrs);
}

// Test GetBackendAttributeValues API
TEST(BackendManager, GetBackendAttributeValues) {
  auto cpu_options = BackendManager::GetBackendAttributeValues("CPU");
  auto nnpi_options = BackendManager::GetBackendAttributeValues("NNPI:3,5,6");
  auto gpu_options = BackendManager::GetBackendAttributeValues("GPU:5");
  auto plaidml_options =
      BackendManager::GetBackendAttributeValues("PLAIDML:device:567:892_34");

  ASSERT_NE(cpu_options.find("ngraph_backend"), cpu_options.end());
  ASSERT_NE(cpu_options.find("_ngraph_device_config"), cpu_options.end());
  ASSERT_EQ(cpu_options["ngraph_backend"], "CPU");
  ASSERT_EQ(cpu_options["_ngraph_device_config"], "");

  ASSERT_NE(nnpi_options.find("ngraph_backend"), nnpi_options.end());
  ASSERT_NE(nnpi_options.find("_ngraph_device_config"), nnpi_options.end());
  ASSERT_EQ(nnpi_options["ngraph_backend"], "NNPI");
  ASSERT_EQ(nnpi_options["_ngraph_device_config"], "3,5,6");

  ASSERT_NE(gpu_options.find("ngraph_backend"), gpu_options.end());
  ASSERT_NE(gpu_options.find("_ngraph_device_config"), gpu_options.end());
  ASSERT_EQ(gpu_options["ngraph_backend"], "GPU");
  ASSERT_EQ(gpu_options["_ngraph_device_config"], "5");

  ASSERT_NE(plaidml_options.find("ngraph_backend"), plaidml_options.end());
  ASSERT_NE(plaidml_options.find("_ngraph_device_config"),
            plaidml_options.end());
  ASSERT_EQ(plaidml_options["ngraph_backend"], "PLAIDML");
  ASSERT_EQ(plaidml_options["_ngraph_device_config"], "device:567:892_34");
}

// Test GetBackendCreationString API
TEST(BackendManager, GetBackendCreationString) {
  unordered_map<string, string> cpu_map = {{"device_config", ""}};
  unordered_map<string, string> nnpi_map = {{"device_id", "5"}};
  unordered_map<string, string> gpu_map = {{"device_config", "678"}};

  auto cpu_backend = BackendManager::GetBackendCreationString("CPU", cpu_map);
  auto nnpi_backend =
      BackendManager::GetBackendCreationString("NNPI", nnpi_map);
  auto gpu_backend = BackendManager::GetBackendCreationString("GPU", gpu_map);

  ASSERT_EQ(cpu_backend, "CPU:");
  ASSERT_EQ(nnpi_backend, "NNPI:5");
  ASSERT_EQ(gpu_backend, "GPU:678");

  // throw errors
  unordered_map<string, string> test_empty_map = {};
  // "device_config" is not valid for NNPI
  unordered_map<string, string> test_missing_config_nnpi = {
      {"device_config", "345"}};
  // "device_id" is not valid for default configs
  unordered_map<string, string> test_missing_config_default = {
      {"device_id", "45"}};

  ASSERT_THROW(BackendManager::GetBackendCreationString("CPU", test_empty_map),
               std::out_of_range);
  ASSERT_THROW(BackendManager::GetBackendCreationString("NNPI", test_empty_map),
               std::out_of_range);
  ASSERT_THROW(BackendManager::GetBackendCreationString("GPU", test_empty_map),
               std::out_of_range);

  ASSERT_THROW(BackendManager::GetBackendCreationString(
                   "CPU", test_missing_config_default),
               std::out_of_range);
  ASSERT_THROW(BackendManager::GetBackendCreationString(
                   "NNPI", test_missing_config_nnpi),
               std::out_of_range);
  ASSERT_THROW(BackendManager::GetBackendCreationString(
                   "GPU", test_missing_config_default),
               std::out_of_range);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
