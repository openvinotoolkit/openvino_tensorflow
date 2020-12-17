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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "logging/tf_graph_writer.h"
#include "ngraph_bridge/backend_manager.h"
#include "ngraph_bridge/ngraph_assign_clusters.h"
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
  auto env_map = StoreEnv({"NGRAPH_TF_BACKEND"});

  ASSERT_OK(BackendManager::SetBackend("CPU"));
  string backend;
  ASSERT_OK(BackendManager::GetBackendName(backend));
  ASSERT_EQ(backend, "CPU");
  ASSERT_NOT_OK(BackendManager::SetBackend("temp"));

  // Clean Up
  // If NGRAPH_TF_BACKEND was set, set it back
  RestoreEnv(env_map);
}

// Test GetBackend API
TEST(BackendManager, GetBackendName) {
  auto env_map = StoreEnv({"NGRAPH_TF_BACKEND"});

  ASSERT_OK(BackendManager::SetBackend("CPU"));
  string backend;
  ASSERT_OK(BackendManager::GetBackendName(backend));
  ASSERT_EQ(backend, "CPU");

  // expected ERROR
  SetBackendUsingEnvVar("DUMMY");
  ASSERT_OK(BackendManager::GetBackendName(backend));
  ASSERT_EQ(backend, "CPU");

  // set env variable to ""
  SetBackendUsingEnvVar("");
  ASSERT_OK(BackendManager::GetBackendName(backend));
  ASSERT_EQ(backend, "CPU");

  // set backend to dummy and env variable to CPU
  // expected CPU
  ASSERT_NOT_OK(BackendManager::SetBackend("DUMMY"));
  SetBackendUsingEnvVar("CPU");
  ASSERT_OK(BackendManager::GetBackendName(backend));
  ASSERT_EQ(backend, "CPU");

  // unset env variable
  // expected interpreter
  UnsetBackendUsingEnvVar();
  ASSERT_OK(BackendManager::GetBackendName(backend));
  ASSERT_EQ(backend, "CPU");

  // Clean up
  UnsetBackendUsingEnvVar();
  ASSERT_OK(BackendManager::SetBackend("CPU"));
  // restore
  // If NGRAPH_TF_BACKEND was set, set it back
  RestoreEnv(env_map);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
