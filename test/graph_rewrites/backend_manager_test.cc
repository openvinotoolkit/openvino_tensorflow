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

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

/*
These tests test the Backend Handling by the bridge.
*/

// Test SetBackendAPI
TEST(BackendManager, SetBackend) {
  // If NGRAPH_TF_BACKEND is set, unset it
  const unordered_map<string, string>& env_map = StoreEnv();

  ASSERT_OK(BackendManager::SetBackendName("CPU"));
  string cpu_backend;
  ASSERT_OK(BackendManager::GetCurrentlySetBackendName(&cpu_backend));

  ASSERT_EQ(cpu_backend, "CPU");

  ASSERT_OK(BackendManager::SetBackendName("INTERPRETER"));
  string current_backend;
  ASSERT_OK(BackendManager::GetCurrentlySetBackendName(&current_backend));

  ASSERT_EQ(current_backend, "INTERPRETER");

  ASSERT_NOT_OK(BackendManager::SetBackendName("temp"));

  // Clean Up
  ASSERT_OK(BackendManager::SetBackendName("CPU"));
  // If NGRAPH_TF_BACKEND was set, set it back
  RestoreEnv(env_map);
}

// Test GetCurrentlySetBackendNameAPI
// Test with env variable set
TEST(BackendManager, GetCurrentlySetBackendName) {
  // If NGRAPH_TF_BACKEND is set, unset it
  const unordered_map<string, string>& env_map = StoreEnv();

  string cpu_backend = "CPU";
  string intp_backend = "INTERPRETER";

  // set backend to interpreter and env variable to CPU
  // expected CPU
  ASSERT_OK(BackendManager::SetBackendName(intp_backend));
  SetBackendUsingEnvVar(cpu_backend);
  string backend;
  ASSERT_OK(BackendManager::GetCurrentlySetBackendName(&backend));
  ASSERT_EQ(cpu_backend, backend);

  // unset env variable
  // expected interpreter
  UnsetBackendUsingEnvVar();
  ASSERT_OK(BackendManager::GetCurrentlySetBackendName(&backend));
  ASSERT_EQ(intp_backend, backend);

  // set env variable to DUMMY
  // expected ERROR
  SetBackendUsingEnvVar("DUMMY");
  ASSERT_NOT_OK(BackendManager::GetCurrentlySetBackendName(&backend));

  // set env variable to ""
  // expected ERROR
  SetBackendUsingEnvVar("");
  ASSERT_NOT_OK(BackendManager::GetCurrentlySetBackendName(&backend));

  // Clean up
  UnsetBackendUsingEnvVar();
  ASSERT_OK(BackendManager::SetBackendName("CPU"));
  // restore
  // If NGRAPH_TF_BACKEND was set, set it back
  RestoreEnv(env_map);
}

// Test CanCreateBackend
TEST(BackendManager, CanCreateBackend) {
  ASSERT_OK(BackendManager::CanCreateBackend("CPU"));
  ASSERT_OK(BackendManager::CanCreateBackend("CPU:0"));
  ASSERT_NOT_OK(BackendManager::CanCreateBackend("temp"));
  ASSERT_NOT_OK(BackendManager::CanCreateBackend(""));
}

// Test GetSupportedBackendNames
TEST(BackendManager, GetSupportedBackendNames) {
  vector<string> ng_tf_backends = BackendManager::GetSupportedBackendNames();

  NGRAPH_VLOG(5) << "Supported Backends";
  for (auto backend : ng_tf_backends) {
    NGRAPH_VLOG(5) << backend;
  }

  vector<string> ng_backends =
      ng::runtime::BackendManager::get_registered_backends();

  NGRAPH_VLOG(5) << "nGraph Supported Backends";
  for (auto backend : ng_backends) {
    NGRAPH_VLOG(5) << backend;
  }

  ASSERT_EQ(ng_tf_backends.size(), ng_backends.size());
  ASSERT_EQ(ng_tf_backends, ng_backends);
}

// Test Backend Assignment
// The backend passed to MarkForClustering is attached to the nodes
TEST(BackendManager, BackendAssignment) {
  Scope root = Scope::NewRootScope();
  auto A = ops::Const(root.WithOpName("A"), {1.0f, 1.0f});
  auto B = ops::Const(root.WithOpName("B"), {1.0f, 1.0f});
  auto R = ops::Add(root.WithOpName("R"), A, B);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  std::set<string> skip_these_nodes = {};

  string dummy_backend = "DUMMY";
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, dummy_backend));
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
  ASSERT_EQ(bA, dummy_backend);
}

// Test Backend Clustering
// Nodes with different backends are not clustered together
TEST(BackendManager, BackendClustering) {
  Scope root = Scope::NewRootScope();
  auto A = ops::Const(root.WithOpName("A"), {1.0f, 1.0f});
  auto B = ops::Const(root.WithOpName("B"), {1.0f, 1.0f});
  auto R = ops::Add(root.WithOpName("R"), A, B);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  std::set<string> skip_these_nodes = {};

  string dummy_backendA = "DUMMYA";
  string dummy_backendB = "DUMMYB";
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes, dummy_backendA));

  std::map<std::string, Node*> node_map;
  for (auto node : graph.op_nodes()) {
    node_map[node->name()] = node;
  }

  SetNodeBackend(node_map["B"], dummy_backendB);
  ASSERT_OK(AssignClusters(&graph));

  int A_cluster, B_cluster, R_cluster;
  ASSERT_OK(GetNodeCluster(node_map["A"], &A_cluster));
  ASSERT_OK(GetNodeCluster(node_map["B"], &B_cluster));
  ASSERT_OK(GetNodeCluster(node_map["R"], &R_cluster));

  ASSERT_EQ(A_cluster, R_cluster);
  ASSERT_NE(A_cluster, B_cluster);
}

// Test GetBackendAttributeValues API
TEST(BackendManager, GetBackendAttributeValues) {
  auto cpu_options = BackendManager::GetBackendAttributeValues("CPU");
  auto nnpi_options = BackendManager::GetBackendAttributeValues("NNPI:3,5,6");
  auto gpu_options = BackendManager::GetBackendAttributeValues("GPU:5");
  auto plaidml_options =
      BackendManager::GetBackendAttributeValues("PLAIDML:device:567:892_34");

  ASSERT_NE(cpu_options.find("ngraph_backend"), cpu_options.end());
  ASSERT_NE(cpu_options.find("ngraph_device_id"), cpu_options.end());
  ASSERT_EQ(cpu_options["ngraph_backend"], "CPU");
  ASSERT_EQ(cpu_options["ngraph_device_id"], "");

  ASSERT_NE(nnpi_options.find("ngraph_backend"), nnpi_options.end());
  ASSERT_NE(nnpi_options.find("ngraph_device_id"), nnpi_options.end());
  ASSERT_EQ(nnpi_options["ngraph_backend"], "NNPI");
  ASSERT_EQ(nnpi_options["ngraph_device_id"], "3,5,6");

  ASSERT_NE(gpu_options.find("ngraph_backend"), gpu_options.end());
  ASSERT_NE(gpu_options.find("ngraph_device_id"), gpu_options.end());
  ASSERT_EQ(gpu_options["ngraph_backend"], "GPU");
  ASSERT_EQ(gpu_options["ngraph_device_id"], "5");

  ASSERT_NE(plaidml_options.find("ngraph_backend"), plaidml_options.end());
  ASSERT_NE(plaidml_options.find("ngraph_device_id"), plaidml_options.end());
  ASSERT_EQ(plaidml_options["ngraph_backend"], "PLAIDML");
  ASSERT_EQ(plaidml_options["ngraph_device_id"], "device:567:892_34");
}

// Test GetBackendCreationString API
TEST(BackendManager, GetBackendCreationString) {
  string cpu_device_id = "";
  string nnpi_device_id = "5";
  string gpu_device_id = "678";

  auto cpu_backend =
      BackendManager::GetBackendCreationString("CPU", cpu_device_id);
  auto nnpi_backend =
      BackendManager::GetBackendCreationString("NNPI", nnpi_device_id);
  auto gpu_backend =
      BackendManager::GetBackendCreationString("GPU", gpu_device_id);

  ASSERT_EQ(cpu_backend, "CPU");
  ASSERT_EQ(nnpi_backend, "NNPI:5");
  ASSERT_EQ(gpu_backend, "GPU:678");
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
