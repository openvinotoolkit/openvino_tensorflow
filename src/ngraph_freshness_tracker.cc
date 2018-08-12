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
#include "ngraph_freshness_tracker.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

void NGraphFreshnessTracker::MarkFresh(const void* base_pointer,
                                       std::shared_ptr<ngraph::Function> user) {
  mutex_lock l(mu_);
  auto it = freshness_map_.find(base_pointer);
  if (it != freshness_map_.end()) {
    it->second.insert(user);
  }
}

bool NGraphFreshnessTracker::IsFresh(const void* base_pointer,
                                     std::shared_ptr<ngraph::Function> user) {
  mutex_lock l(mu_);
  auto it = freshness_map_.find(base_pointer);
  if (it == freshness_map_.end()) {
    return false;
  } else {
    return (it->second.count(user) > 0);
  }
}

void NGraphFreshnessTracker::MarkStale(const void* base_pointer) {
  mutex_lock l(mu_);
  auto it = freshness_map_.find(base_pointer);
  if (it != freshness_map_.end()) {
    it->second.clear();
  }
}

void NGraphFreshnessTracker::AddTensor(const void* base_pointer) {
  mutex_lock l(mu_);
  auto it = freshness_map_.find(base_pointer);
  if (it == freshness_map_.end()) {
    freshness_map_[base_pointer] =
        std::set<std::shared_ptr<ngraph::Function>>{};
  }
}

void NGraphFreshnessTracker::RemoveTensor(const void* base_pointer) {
  mutex_lock l(mu_);
  freshness_map_.erase(base_pointer);
}

void NGraphFreshnessTracker::RemoveUser(
    std::shared_ptr<ngraph::Function> user) {
  mutex_lock l(mu_);
  for (auto kv : freshness_map_) {
    kv.second.erase(user);
  }
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
