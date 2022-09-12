/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

// The backend manager class is a singelton class that interfaces with the
// bridge to provide necessary backend

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"

#include "backend.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

class BackendManager {
 public:
  // Returns the nGraph supported backend names
  static vector<string> GetSupportedBackends();

  // Set the BackendManager backend ng_backend_name_
  static Status SetBackend(const string& backend_name = "CPU");

  // Returns the currently set backend
  static shared_ptr<Backend> GetBackend();

  // Returns the currently set backend's name
  static Status GetBackendName(string& backend_name);

  // Returns if TF Frontend is disabled
  static bool TFFrontendDisabled();

  ~BackendManager();

 private:
  // Creates backend of backend_name type
  static Status CreateBackend(shared_ptr<Backend>& backend,
                              string& backend_name);

  static shared_ptr<Backend> m_backend;
  static string m_backend_name;
  static mutex m_backend_mutex;
  static bool m_tf_frontend_disabled;
};

}  // namespace openvino_tensorflow
}  // namespace tensorflow
