/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

#include "../test_utilities.h"
#include "gtest/gtest.h"
#include "ngraph_assign_clusters.h"
#include "ngraph_backend_manager.h"
#include "ngraph_mark_for_clustering.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
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
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes));
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
  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes));

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

  ASSERT_OK(MarkForClustering(&graph, skip_these_nodes));

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

// Test Backend Run
TEST(BackendManager, BackendRun) {
  ASSERT_OK(BackendManager::SetBackendName("INTERPRETER"));
  Scope root = Scope::NewRootScope();
  auto A = ops::Placeholder(root.WithOpName("A"), DT_FLOAT);
  auto B = ops::Placeholder(root.WithOpName("B"), DT_FLOAT);
  auto R = ops::Add(root.WithOpName("R"), A, B);
  auto S = ops::Sub(root.WithOpName("S"), R, B);

  auto default_backend = BackendManager::GetCurrentlySetBackendName();
  std::vector<Tensor> cpu_outputs;
  ClientSession session_cpu(root);
  // Run and fetch v
  ASSERT_OK(session_cpu.Run({{A, {1.0f, 1.0f}}, {B, {1.0f, 1.0f}}}, {R, S},
                            &cpu_outputs));

  ASSERT_OK(BackendManager::SetBackendName("CPU"));
  auto backend2 = BackendManager::GetCurrentlySetBackendName();

  std::vector<Tensor> inter_outputs;
  ClientSession session_inter(root);
  ASSERT_OK(session_inter.Run({{A, {1.0f, 1.0f}}, {B, {1.0f, 1.0f}}}, {R, S},
                              &inter_outputs));
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow