/*****************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"

#include "contexts.h"
#include "openvino_tensorflow/cluster_manager.h"
#include "openvino_tensorflow/executable.h"
#include "openvino_tensorflow/ie_tensor.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

class Backend {
 public:
  Backend(const string& configuration_string);
  ~Backend() {
    NGraphClusterManager::EvictAllClusters();
    if (m_device != "GPU") {
      NGraphClusterManager::EvictMRUClusters();
    }
    ReleaseGlobalContext();
  }

  shared_ptr<Executable> Compile(shared_ptr<ov::Model> func,
                                 bool enable_performance_data = false);

  static GlobalContext& GetGlobalContext();
  static void ReleaseGlobalContext();
  std::string GetDeviceType();

 private:
  string m_device;
  string m_device_type;
};
}  // end namespace openvino_tensorflow
}
