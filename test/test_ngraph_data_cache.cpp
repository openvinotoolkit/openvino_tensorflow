/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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
#include <atomic>
#include <memory>
#include <thread>

#include "absl/synchronization/barrier.h"
#include "gtest/gtest.h"

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_data_cache.h"
#include "ngraph_bridge/version.h"

#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

class NGraphDataCacheTest : public ::testing::Test {
 protected:
  NgraphDataCache<std::string, int> m_ng_data_cache{3};
  int num_threads = 2;
  absl::Barrier* barrier_ = new absl::Barrier(num_threads);
  std::atomic<int> create_count{0};
  int destroy_count = 0;
  bool item_evicted = false;

  std::pair<Status, int> CreateItem(std::string abc) {
    create_count++;
    if (barrier_->Block()) {
      delete barrier_;
    }
    return std::make_pair(Status::OK(), 3);
  }

  std::pair<Status, int> CreateItemNoBarrier(std::string abc) {
    return std::make_pair(Status::OK(), 3);
  }

  std::pair<Status, int> CreateItemReturnError(std::string abc) {
    return std::make_pair(errors::Internal("Failed to create item"), 0);
  }
  void DestroyItem(int i) {
    item_evicted = true;
    destroy_count++;
  }
};

// Tests LooUpOrCreate(), in multithreading environment
TEST_F(NGraphDataCacheTest, SameKeyMultiThread) {
  auto worker = [&](size_t thread_id) {
    auto create_item =
        std::bind(&NGraphDataCacheTest_SameKeyMultiThread_Test::CreateItem,
                  this, std::placeholders::_1);
    auto create_item_ret_err = std::bind(
        &NGraphDataCacheTest_SameKeyMultiThread_Test::CreateItemReturnError,
        this, std::placeholders::_1);
    bool cache_hit;
    ASSERT_OK(
        (m_ng_data_cache.LookUpOrCreate("abc", create_item, cache_hit)).first);
    ASSERT_OK(
        m_ng_data_cache.LookUpOrCreate("abc", create_item, cache_hit).first);
    ASSERT_NOT_OK(
        m_ng_data_cache.LookUpOrCreate("def", create_item_ret_err, cache_hit)
            .first);
    ASSERT_EQ(
        m_ng_data_cache.LookUpOrCreate("def", create_item_ret_err, cache_hit)
            .first.error_message(),
        "Failed to create item");

  };

  std::thread thread0(worker, 0);
  std::thread thread1(worker, 1);

  thread0.join();
  thread1.join();
  // This is ensured by using Barrier inside CreateItem()
  ASSERT_EQ(create_count, 2);
  ASSERT_EQ(m_ng_data_cache.m_ng_items_map.size(), 1);
  ASSERT_EQ(m_ng_data_cache.m_ng_items_map.find("def"),
            m_ng_data_cache.m_ng_items_map.end());
}

// Testing to ensure destoy called back is called, when cache is full.
TEST_F(NGraphDataCacheTest, TestItemEviction) {
  auto create_item =
      std::bind(&NGraphDataCacheTest_TestItemEviction_Test::CreateItemNoBarrier,
                this, std::placeholders::_1);
  auto destroy_item =
      std::bind(&NGraphDataCacheTest_TestItemEviction_Test::DestroyItem, this,
                std::placeholders::_1);
  bool cache_hit;
  ASSERT_OK(
      m_ng_data_cache.LookUpOrCreate("abc", create_item, cache_hit).first);
  ASSERT_OK(m_ng_data_cache
                .LookUpOrCreate("def", create_item, destroy_item, cache_hit)
                .first);
  ASSERT_OK(m_ng_data_cache
                .LookUpOrCreate("efg", create_item, destroy_item, cache_hit)
                .first);
  ASSERT_EQ(item_evicted, false);
  ASSERT_OK(m_ng_data_cache
                .LookUpOrCreate("hij", create_item, destroy_item, cache_hit)
                .first);
  ASSERT_EQ(item_evicted, true);
}

// Testing all variations of RemoveItem/All functionality
TEST_F(NGraphDataCacheTest, RemoveItemTest) {
  auto create_item =
      std::bind(&NGraphDataCacheTest_RemoveItemTest_Test::CreateItemNoBarrier,
                this, std::placeholders::_1);
  auto destroy_item =
      std::bind(&NGraphDataCacheTest_RemoveItemTest_Test::DestroyItem, this,
                std::placeholders::_1);
  bool cache_hit;
  ASSERT_OK(
      m_ng_data_cache.LookUpOrCreate("abc", create_item, cache_hit).first);
  ASSERT_OK(
      m_ng_data_cache.LookUpOrCreate("def", create_item, cache_hit).first);
  ASSERT_OK(
      m_ng_data_cache.LookUpOrCreate("efg", create_item, cache_hit).first);
  ASSERT_EQ(item_evicted, false);
  m_ng_data_cache.RemoveAll(destroy_item);
  ASSERT_EQ(item_evicted, true);
  ASSERT_EQ(destroy_count, 3);
  destroy_count = 0;
  m_ng_data_cache.RemoveAll(destroy_item);
  ASSERT_EQ(destroy_count, 0);
  ASSERT_OK(
      m_ng_data_cache.LookUpOrCreate("abc", create_item, cache_hit).first);
  ASSERT_OK(
      m_ng_data_cache.LookUpOrCreate("def", create_item, cache_hit).first);
  ASSERT_EQ(m_ng_data_cache.m_ng_items_map.size(), 2);
  m_ng_data_cache.RemoveItem("def");
  m_ng_data_cache.RemoveItem("def", destroy_item);
  ASSERT_EQ(destroy_count, 0);
  m_ng_data_cache.RemoveItem("abc", destroy_item);
  ASSERT_EQ(destroy_count, 1);
  ASSERT_EQ(m_ng_data_cache.m_ng_items_map.size(), 0);
}
}
}
}
