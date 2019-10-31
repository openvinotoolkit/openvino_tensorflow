/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

// The backend manager class is a singelton class that interfaces with the
// bridge to provide necessary backend

#ifndef NGRAPH_TF_BRIDGE_BACKEND_MANAGER_H_
#define NGRAPH_TF_BRIDGE_BACKEND_MANAGER_H_

#include <atomic>
#include <mutex>
#include <ostream>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend_manager.hpp"

#include "logging/ngraph_log.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

struct Backend {
  shared_ptr<ng::runtime::Backend> backend_ptr;
  mutex backend_mutex;
};

class BackendManager {
 public:
  // Returns the backend name currently set
  // If env variable NGRAPH_TF_BACKEND is set it has precedence
  // over the BackendManager backend ng_backend_name_
  static Status GetCurrentlySetBackendName(string* backend);

  // Returns the nGraph supported backend names
  static vector<string> GetSupportedBackendNames();

  // Returns number of nGraph supported backends
  static size_t GetNumOfSupportedBackends();

  // Set the BackendManager backend ng_backend_name_
  static Status SetBackendName(const string& backend_name);

  // Creates backend of backend_name type
  static Status CreateBackend(const string& backend_name);

  // Tries to create a backend using the backend_string
  // which is a combination of backend_name:device_id
  // Meant to check whether a backend (with or without device id)
  // can be created or not
  // For e.g CanCreateBackend("GPU") and CanCreateBackend("GPU:0")
  static Status CanCreateBackend(const string& backend_string);

  static void ReleaseBackend(const string& backend_name);

  static void SetConfig(const string& backend_name,
                        const std::unordered_map<std::string, std::string>&
                            additional_attributes_map);

  // Returns a backend pointer of the type specified by the backend name
  // The backend must have already been created (use CreateBackend(...))
  static ng::runtime::Backend* GetBackend(const string& backend_name);

  // LockBackend
  static void LockBackend(const string& backend_name);

  // UnlockBackend
  static void UnlockBackend(const string& backend_name);

  // Backend Config Functions
  // These functions facilitate getting/setting
  // of additional backend configurations by abstracting the
  // backend specific details from the user
  // They do not validate the backend type or the attribute values

  // Returns the backend specific additional attributes
  // For e.g.
  // 1. GetBackendAdditionalAttributes("CPU")
  // returns {"_ngraph_device_config"}
  // 2. GetBackendAdditionalAttributes("TEST")
  // returns {"_ngraph_device_config"}
  // 3. GetBackendAdditionalAttributes("NNPI")
  // returns {"_ngraph_device_id", "_ngraph_ice_cores","_ngraph_max_batch_size"}
  static vector<string> GetBackendAdditionalAttributes(
      const string& backend_name);

  // Given a string, splits the string into the backend name and other
  // attributes
  // This does not check whether the string corresponds to a valid backend
  // For e.g.
  // 1. GetBackendAttributeValues("CPU")
  // returns {{"ngraph_backend", "CPU"}, {"_ngraph_device_config", ""}}
  // 2. GetBackendAttributeValues("GPU:2")
  // returns {{"ngraph_backend", "GPU"}, {"_ngraph_device_config", "2"}}
  // 3. GetBackendAttributeValues("TEST:ME")
  // returns {{"ngraph_backend", "TEST"}, {"_ngraph_device_config", "ME"}}
  static unordered_map<string, string>
  GetBackendAttributeValues(  // SplitBackendConfig
      const string& backend_config);

  // Given a backend name and device id
  // joins them into a string to create ngraph backend
  static string GetBackendCreationString(const string& backend_name,
                                         const string& device_id);

  ~BackendManager();

 private:
  static string ng_backend_name_;  // currently set backend name
  static mutex ng_backend_name_mutex_;

  // map of cached backend objects
  static map<string, std::unique_ptr<Backend>> ng_backend_map_;
  static mutex ng_backend_map_mutex_;

  // Map of backends and their reference counts
  static std::map<std::string, int> ref_count_each_backend_;
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif
// NGRAPH_TF_BRIDGE_BACKEND_MANAGER_H
