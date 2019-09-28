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

#include "ngraph_bridge/ngraph_backend_manager.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

BackendManager::~BackendManager() {
  NGRAPH_VLOG(2) << "BackendManager::~BackendManager()";
}

// initialize backend manager
string BackendManager::ng_backend_name_ = "CPU";
mutex BackendManager::ng_backend_name_mutex_;
map<string, std::unique_ptr<Backend>> BackendManager::ng_backend_map_;
mutex BackendManager::ng_backend_map_mutex_;
map<std::string, int> BackendManager::ref_count_each_backend_;

Status BackendManager::SetBackendName(const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_name_mutex_);
  auto status = BackendManager::CanCreateBackend(backend_name);
  if (!status.ok()) {
    return errors::Internal("Failed to set backend: ", status.error_message());
  }
  BackendManager::ng_backend_name_ = backend_name;
  return Status::OK();
}

Status BackendManager::CreateBackend(const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_map_mutex_);
  auto itr = BackendManager::ng_backend_map_.find(backend_name);
  // if backend does not exist create it
  if (itr == BackendManager::ng_backend_map_.end()) {
    std::shared_ptr<ng::runtime::Backend> bend_ptr;
    try {
      bend_ptr = ng::runtime::Backend::create(backend_name);
    } catch (const std::exception& e) {
      return errors::Internal("Could not create backend of type ", backend_name,
                              ". Got exception: ", e.what());
    }

    if (bend_ptr == nullptr) {
      return errors::Internal("Could not create backend of type ", backend_name,
                              " got nullptr");
    }
    std::unique_ptr<Backend> bend = std::unique_ptr<Backend>(new Backend);
    bend->backend_ptr = std::move(bend_ptr);
    BackendManager::ng_backend_map_[backend_name] = std::move(bend);
    BackendManager::ref_count_each_backend_[backend_name] = 0;
  }
  BackendManager::ref_count_each_backend_[backend_name]++;

  NGRAPH_VLOG(2) << "BackendManager::CreateBackend(): " << backend_name
                 << " ref_count: "
                 << BackendManager::ref_count_each_backend_[backend_name];
  return Status::OK();
}

void BackendManager::ReleaseBackend(const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_map_mutex_);
  BackendManager::ref_count_each_backend_[backend_name]--;
  NGRAPH_VLOG(2) << "BackendManager::ReleaseBackend(): " << backend_name
                 << " ref_count: "
                 << BackendManager::ref_count_each_backend_[backend_name];
  if (BackendManager::ref_count_each_backend_[backend_name] == 0) {
    BackendManager::ng_backend_map_[backend_name]->backend_ptr.reset();
    BackendManager::ng_backend_map_.erase(backend_name);
    NGRAPH_VLOG(2) << "Deleted Backend " << backend_name;
  }
}

void BackendManager::SetConfig(
    const string& backend_name,
    const std::unordered_map<std::string, std::string>&
        additional_attributes_map) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_map_mutex_);
  ng::runtime::Backend* bend = GetBackend(backend_name);
  NGRAPH_VLOG(2) << "BackendManager::SetConfig() " << backend_name;
  std::string error;
  std::map<std::string, std::string> device_config_map;
  for (auto i = additional_attributes_map.begin();
       i != additional_attributes_map.end(); i++) {
    device_config_map.insert({i->first, i->second});
  }
  // sending all the additional attributes to the backend
  // it is backend's responsibility to find the one's it needs
  // similar to the implementation for the Interpreter backend
  if (!bend->set_config(device_config_map, error)) {
    NGRAPH_VLOG(2) << "BackendManager::SetConfig(): Could not set config. "
                   << error;
  }
}

// Returns a backend pointer of the type specified by the backend name
ng::runtime::Backend* BackendManager::GetBackend(const string& backend_name) {
  return BackendManager::ng_backend_map_.at(backend_name)->backend_ptr.get();
}

// LockBackend
void BackendManager::LockBackend(const string& backend_name) {
  BackendManager::ng_backend_map_.at(backend_name)->backend_mutex.lock();
}

// UnlockBackend
void BackendManager::UnlockBackend(const string& backend_name) {
  BackendManager::ng_backend_map_.at(backend_name)->backend_mutex.unlock();
}

// Returns the nGraph supported backend names
vector<string> BackendManager::GetSupportedBackendNames() {
  return ng::runtime::BackendManager::get_registered_backends();
}

size_t BackendManager::GetNumOfSupportedBackends() {
  return ng::runtime::BackendManager::get_registered_backends().size();
}

Status BackendManager::CanCreateBackend(const string& backend_string) {
  auto status = BackendManager::CreateBackend(backend_string);
  if (status.ok()) {
    // The call to create backend increases the ref count
    // so releasing the backend here
    BackendManager::ReleaseBackend(backend_string);
  }
  return status;
};

Status BackendManager::GetCurrentlySetBackendName(string* backend_name) {
  const char* ng_backend_env_value = std::getenv("NGRAPH_TF_BACKEND");

  // NGRAPH_TF_BACKEND is not set
  if (ng_backend_env_value == nullptr) {
    *backend_name = BackendManager::ng_backend_name_;
    NGRAPH_VLOG(1) << "Using the currently set backend " << (*backend_name);
    return Status::OK();
  }

  // NGRAPH_TF_BACKEND is set
  string backend_env = std::string(ng_backend_env_value);
  auto status = BackendManager::CanCreateBackend(backend_env);
  if (!status.ok()) {
    return errors::Internal("NGRAPH_TF_BACKEND: ", status.error_message());
  }

  *backend_name = backend_env;
  NGRAPH_VLOG(1) << "Overriding backend using the environment variable "
                    "to "
                 << (*backend_name);
  return Status::OK();
};

// Split
unordered_map<string, string> BackendManager::GetBackendAttributeValues(
    const string& backend_config) {
  unordered_map<string, string> backend_parameters;

  int delimiter_index = backend_config.find(':');
  if (delimiter_index < 0) {
    // ":" not found
    backend_parameters["ngraph_backend"] = backend_config;
    backend_parameters["ngraph_device_id"] = "";
  } else {
    backend_parameters["ngraph_backend"] =
        backend_config.substr(0, delimiter_index);
    backend_parameters["ngraph_device_id"] =
        backend_config.substr(delimiter_index + 1);
  }

  NGRAPH_VLOG(3) << "Got Backend Name " << backend_parameters["ngraph_backend"];
  NGRAPH_VLOG(3) << "Got Device Id  " << backend_parameters["ngraph_device_id"];

  return backend_parameters;
}

// Join
string BackendManager::GetBackendCreationString(const string& backend_name,
                                                const string& device_id) {
  if (device_id != "") {
    return backend_name + ":" + device_id;
  } else {
    return backend_name;
  }
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
