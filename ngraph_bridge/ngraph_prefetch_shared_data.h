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

#ifndef NGRAPH_PREFETCH_SHARED_DATA_H_
#define NGRAPH_PREFETCH_SHARED_DATA_H_
#pragma once

#include <mutex>
#include <ostream>
#include <string>
#include <vector>

#include "tensorflow/core/framework/resource_mgr.h"

#include "ngraph/runtime/tensor.hpp"

#include "ngraph_bridge/thread_safe_queue.h"

namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {

namespace ngraph_bridge {

class NGraphPrefetchSharedResouce : public ResourceBase {
 public:
  explicit NGraphPrefetchSharedResouce(
      const std::string& ng_enc_op_name, int cluster_id, int graph_id,
      const vector<int>& prefetch_input_indexes)
      : m_ng_enc_op_name(ng_enc_op_name),
        m_graph_id(graph_id),
        m_cluster_id(cluster_id),
        m_prefetch_input_indexes(prefetch_input_indexes) {}

  // Returns a debug string for *this.
  string DebugString() const override { return "NGraphPrefetchSharedResouce"; }

  // Returns memory used by this resource.
  int64 MemoryUsed() const override { return 0; }
  std::string GetName() const { return m_ng_enc_op_name; }
  int GetGraphId() const { return m_graph_id; }
  int GetClusterId() const { return m_cluster_id; }

  static constexpr const char* RESOURCE_NAME = "NG_PREFETCH_DATA";
  static constexpr const char* CONTAINER_NAME = "NG_PREFETCH_DATA_CONTAINER";
  static constexpr const char* NGRAPH_TF_USE_PREFETCH =
      "NGRAPH_TF_USE_PREFETCH";

  struct IOTensorBundle {
    int Id;
    std::vector<shared_ptr<ng::runtime::Tensor>> Inputs;
    std::vector<shared_ptr<ng::runtime::Tensor>> Outputs;
  };

  // Adds the given nGraph input output tensors to write to
  // Uses m_prefetch_input_indexes to figure out which input tensors
  // are prefetched and writes into them
  // This is called by the NGraphEncapOp
  void AddNextIOTensorBundleForDeviceTransfer(IOTensorBundle next) {
    m_tf_2_ng.Add(std::move(next));
  }

  // Returns the Input output tensors to be used to copy TF tensors to NG device
  // This will be called by the prefetcher
  IOTensorBundle GetNextIOTensorBundleForDeviceTransfer() {
    return std::move(m_tf_2_ng.GetNextAvailable());
  }

  // Adds the given nGraph input output tensors to write to
  // This is called by the prefetcher to add Tensors that are copied
  // from TF tensor and are now ready for the next iteration
  void AddNextIOTensorBundleReadyForDeviceExecution(IOTensorBundle next) {
    m_ng_2_tf.Add(std::move(next));
  }

  // Returns the Input output tensors ready to be executed by NG device
  // This will be called by the NGEncOp
  IOTensorBundle GetNextIOTensorBundleReadyForDeviceExecution() {
    return std::move(m_ng_2_tf.GetNextAvailable());
  }

  void SetBufferDepth(int depth) {
    m_mutex.Lock();
    // TODO assert m_prefetch_buffer_depth == -1 || m_prefetch_buffer_depth ==
    // depth
    // To make sure we never try to set a different depth once it is set
    m_prefetch_buffer_depth = depth;
    m_cv.SignalAll();
    m_mutex.Unlock();
  }
  int GetBufferDepth() {
    // Locking GetBufferDepth till SetBufferDepth is called
    // In case of races where Get is called before Set,
    // We want to ensure Set finishes before Get returns
    m_mutex.ReaderLock();
    while (m_prefetch_buffer_depth == -1) {
      m_cv.Wait(&m_mutex);
    }
    m_mutex.ReaderUnlock();
    return m_prefetch_buffer_depth;
  }

  void IncrSkipCount() { m_skip_count++; }
  int GetSkipCount() { return m_skip_count; }

  const vector<int>& GetPrefetchInputIndexes() {
    return m_prefetch_input_indexes;
  }

 private:
  const std::string m_ng_enc_op_name;
  const int m_graph_id;
  const int m_cluster_id;
  const vector<int> m_prefetch_input_indexes;
  // We need to maintain two queues as follows:
  // ----------+------------+------------+------------------------------------+
  // Queue     | Writer     | Reader     | Comments                           |
  // ----------+------------+------------+------------------------------------+
  // m_tf_2_ng | Prefetcher | NgEncOp    | TF tensors copied to the nG tensor |
  // ----------+------------+------------+------------------------------------+
  // m_ng_2_tf | NgEncOp    | Prefetcher | NGEnc enqueus empty nGTensors here |
  // ----------+------------+------------+------------------------------------+
  //
  // The interaction is as follows:
  // Iteration  Action
  // 1          NGEncOp pushes the Input/Output tensors to m_ng_2_tf queue
  // 2
  //            Prefetcher pulls Input/Output tensors out of m_ng_2_tf queue and
  //            and copies TF data to the prefetched inputs
  //            Prefetcher pushes this item to the m_tf_2_ng queue
  //            NGEncOp pushes the Input/Output tensors to m_ng_2_tf queue
  //            NGEncOp pulls Input/Output tensors from m_tf_2_ng (from previous
  //            iteration) and executes
  // 3          Repeat

  ThreadSafeQueue<IOTensorBundle> m_tf_2_ng;
  ThreadSafeQueue<IOTensorBundle> m_ng_2_tf;

  int m_prefetch_buffer_depth{-1};
  int m_skip_count{0};

  // Mutex and cond var to control m_prefetch_buffer_depth
  absl::CondVar m_cv;
  absl::Mutex m_mutex;
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_PREFETCH_SHARED_DATA_H_
