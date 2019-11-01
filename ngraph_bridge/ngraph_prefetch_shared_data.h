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

namespace tensorflow {

namespace ngraph_bridge {

class NGraphPrefetchSharedResouce : public ResourceBase {
 public:
  explicit NGraphPrefetchSharedResouce(const std::string& ng_enc_op_name,
                                       const std::string& backend_name,
                                       int cluster_id, int graph_id)
      : m_ng_enc_op_name(ng_enc_op_name),
        m_backend_name(backend_name),
        m_graph_id(graph_id),
        m_cluster_id(cluster_id) {}

  // Returns a debug string for *this.
  string DebugString() const override { return "NGraphPrefetchSharedResouce"; }

  // Returns memory used by this resource.
  int64 MemoryUsed() const override { return 0; }
  std::string GetName() const { return m_ng_enc_op_name; }
  std::string GetBackendName() const { return m_backend_name; }
  int GetGraphId() const { return m_graph_id; }
  int GetClusterId() const { return m_cluster_id; }

  static constexpr const char* RESOURCE_NAME = "NG_PREFETCH_DATA";
  static constexpr const char* CONTAINER_NAME = "NG_PREFETCH_DATA_CONTAINER";
  static constexpr const char* NGRAPH_TF_USE_PREFETCH =
      "NGRAPH_TF_USE_PREFETCH";

 private:
  const std::string m_ng_enc_op_name;
  const std::string m_backend_name;
  const int m_graph_id;
  const int m_cluster_id;
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_PREFETCH_SHARED_DATA_H_
