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
#ifndef NGRAPH_FRESHNESS_TRACKER_H_
#define NGRAPH_FRESHNESS_TRACKER_H_

#include <set>

#include "tensorflow/core/framework/resource_mgr.h"

namespace tf = tensorflow;

namespace ngraph_bridge {
//
// A class used to track freshness of tensors for the purpose of memoization.
// Tensors are tracked according to their base pointers; thus this is only
// suitable in cases where a tensor's base pointer cannot be changed. Tensors
// internal to the nGraph bridge conform to these restrictions.
//
// General usage:
//
//   NGraphFreshnessTracker* tracker;
//   ...
//   Tensor* t;
//   const void* tensor_base_ptr = (const void *)tf::DMAHelper::base(t);
//
//   [tracker->IsRegistered(tensor_base_ptr) will return false]
//   [tracker->IsFresh(tensor_base_ptr) will return false]
//
//   tracker->MarkFresh(tensor_base_ptr); // marks t as "fresh"
//
//   [tracker->IsRegistered(tensor_base_ptr) will return true]
//   [tracker->IsFresh(tensor_base_ptr) will return true]
//
//   tracker->MarkStale(tensor_base_ptr); // marks t as "stale"
//
//   [tracker->IsRegistered(tensor_base_ptr) will return true]
//   [tracker->IsFresh(tensor_base_ptr) will return false]
//
// Inside the nGraph bridge, the freshness tracker is stored as a resource in
// the ResourceMgr's default container, with the resource name
// "ngraph_freshness_tracker".
//
// TODO(amprocte): Freshness really needs to be tracked on a _per-function_
// basis.
//
class NGraphFreshnessTracker : public tf::ResourceBase {
 public:
  explicit NGraphFreshnessTracker() {}
  // Not copyable or movable.
  NGraphFreshnessTracker(const NGraphFreshnessTracker&) = delete;
  NGraphFreshnessTracker& operator=(const NGraphFreshnessTracker&) = delete;

  std::string DebugString() override { return "FreshnessTracker"; }

  void MarkFresh(const void* base_pointer);
  void MarkStale(const void* base_pointer);
  bool IsRegistered(const void* base_pointer);
  bool IsFresh(const void* base_pointer);

 private:
  tf::mutex mu_;
  std::map<const void*, bool> freshness_map_;

  ~NGraphFreshnessTracker() override {}
};
}  // namespace ngraph_bridge

#endif
