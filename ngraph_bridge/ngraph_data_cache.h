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

#ifndef NGRAPH_DATA_CACHE_H_
#define NGRAPH_DATA_CACHE_H_
#pragma once

#include <deque>
#include <mutex>
#include <ostream>
#include <vector>
#include "absl/synchronization/mutex.h"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

#include "logging/ngraph_log.h"
#include "ngraph/ngraph.hpp"

#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_pipelined_tensors.h"

namespace tensorflow {

namespace ngraph_bridge {

// Forward declaration for friend class
namespace testing {
class NGraphDataCacheTest_SameKeyMultiThread_Test;
class NGraphDataCacheTest_RemoveItemTest_Test;
}

template <typename KeyType, typename ValueType>
class NgraphDataCache {
 public:
  explicit NgraphDataCache(int depth);
  ~NgraphDataCache();

  // This method performs lookup in the cache for requested key, if not found
  // it will create item, put it in the cache and returns item and status
  std::pair<Status, ValueType> LookUpOrCreate(
      KeyType key,
      std::function<std::pair<Status, ValueType>()> callback_create_item,
      std::function<void(ValueType)> callback_destroy_item);

  // Overload for above function, which doesn't take callback for destroy item
  std::pair<Status, ValueType> LookUpOrCreate(
      KeyType key,
      std::function<std::pair<Status, ValueType>()> callback_create_item);
  Status RemoveItem(KeyType key);
  Status RemoveItem(KeyType key,
                    std::function<void(ValueType)> callback_destroy_item);
  Status RemoveAll(std::function<void(ValueType)> callback_destroy_item);

 private:
  std::unordered_map<KeyType, ValueType> m_ng_items_map;
  std::deque<KeyType> m_lru;
  int m_depth;
  absl::Mutex m_mutex;

  // Test class
  friend class tensorflow::ngraph_bridge::testing::
      NGraphDataCacheTest_SameKeyMultiThread_Test;
  friend class tensorflow::ngraph_bridge::testing::
      NGraphDataCacheTest_RemoveItemTest_Test;
};

template <typename KeyType, typename ValueType>
NgraphDataCache<KeyType, ValueType>::NgraphDataCache(int depth)
    : m_depth(depth) {}

template <typename KeyType, typename ValueType>
NgraphDataCache<KeyType, ValueType>::~NgraphDataCache() {
  m_ng_items_map.clear();
  m_lru.clear();
}

template <typename KeyType, typename ValueType>
Status NgraphDataCache<KeyType, ValueType>::RemoveItem(
    KeyType key, std::function<void(ValueType)> callback_destroy_item) {
  absl::MutexLock lock(&m_mutex);
  if (m_ng_items_map.find(key) != m_ng_items_map.end()) {
    try {
      callback_destroy_item(m_ng_items_map.at(key));
    } catch (std::bad_function_call& exception) {
      return errors::Internal(
          "Failed to destroy item. Invalid Callback to Destroy");
    }
    m_ng_items_map.erase(key);
    m_lru.pop_back();
  }
  if (m_ng_items_map.size() != m_lru.size()) {
    return errors::Internal(
        "Error occured: size of m_ng_items_map is not same as that of m_lru");
  }
  return Status::OK();
}

template <typename KeyType, typename ValueType>
Status NgraphDataCache<KeyType, ValueType>::RemoveItem(KeyType key) {
  return RemoveItem(key, [](ValueType) {});
}

template <typename KeyType, typename ValueType>
Status NgraphDataCache<KeyType, ValueType>::RemoveAll(
    std::function<void(ValueType)> callback_destroy_item) {
  absl::MutexLock lock(&m_mutex);
  for (auto it = m_ng_items_map.begin(); it != m_ng_items_map.end(); it++) {
    try {
      callback_destroy_item(it->second);
    } catch (std::bad_function_call& exception) {
      return errors::Internal(
          "Failed to destroy item. Invalid Callback to Destroy");
    }
  }
  if (m_ng_items_map.size() != m_lru.size()) {
    return errors::Internal(
        "Error occured: size of m_ng_items_map is not same as that of m_lru");
  }
  m_ng_items_map.erase(m_ng_items_map.begin(), m_ng_items_map.end());
  m_lru.clear();
  return Status::OK();
}

template <typename KeyType, typename ValueType>
std::pair<Status, ValueType>
NgraphDataCache<KeyType, ValueType>::LookUpOrCreate(
    KeyType key,
    std::function<std::pair<Status, ValueType>()> callback_create_item,
    std::function<void(ValueType)> callback_destroy_item) {
  bool found_in_cache;
  // look up in the cache
  {
    absl::MutexLock lock(&m_mutex);
    auto it = m_ng_items_map.find(key);
    found_in_cache = (it != m_ng_items_map.end());
    if (found_in_cache) {
      return std::make_pair(Status::OK(), m_ng_items_map.at(key));
    }
  }
  // Item not found in cache, create item
  ValueType item;
  pair<Status, ValueType> status_item_pair;
  try {
    status_item_pair = callback_create_item();
  } catch (std::bad_function_call& exception) {
    return std::make_pair(
        errors::Internal(
            "Failed to create an item. Invalid Callback to Create"),
        item);
  }
  // If item is successfully created we will place in the cache.
  if (status_item_pair.first == Status::OK()) {
    item = status_item_pair.second;
    // lock begins
    {
      absl::MutexLock lock(&m_mutex);
      // Remove item if cache is full
      if (m_ng_items_map.size() == m_depth) {
        auto key_to_evict = m_lru.back();
        try {
          callback_destroy_item(m_ng_items_map.at(key_to_evict));
        } catch (std::bad_function_call& exception) {
          return std::make_pair(
              errors::Internal(
                  "Failed to destroy item. Invalid Callback to Destroy"),
              item);
        }
        m_ng_items_map.erase(key_to_evict);
        m_lru.pop_back();
      }
      // Add item to cache
      auto it = m_ng_items_map.emplace(key, item);
      if (it.second == true) {
        m_lru.push_front(key);
      } else {
        auto key_itr = find(m_lru.begin(), m_lru.end(), key);
        m_lru.erase(key_itr);
        m_lru.push_front(key);
      }
    }  // lock ends here.

    if (m_ng_items_map.size() != m_lru.size()) {
      return std::make_pair(
          errors::Internal("Error occured: size of m_ng_items_map is not same "
                           "as that of m_lru"),
          item);
    }
    return std::make_pair(Status::OK(), item);
  }

  return status_item_pair;
}

template <typename KeyType, typename ValueType>
std::pair<Status, ValueType>
NgraphDataCache<KeyType, ValueType>::LookUpOrCreate(
    KeyType key,
    std::function<std::pair<Status, ValueType>()> callback_create_item) {
  return LookUpOrCreate(key, callback_create_item, [](ValueType) {});
}
}
}
#endif  // NGRAPH_DATA_CACHE_H_
