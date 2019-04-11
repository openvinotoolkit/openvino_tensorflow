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

#include "ngraph_backend_manager.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

BackendManager::~BackendManager() {
  NGRAPH_VLOG(2) << "BackendManager::~BackendManager() DONE";
}
// initialize backend manager
string BackendManager::ng_backend_name_ = "CPU";
mutex BackendManager::ng_backend_name_mutex_;
map<string, Backend*> BackendManager::ng_backend_map_;
mutex BackendManager::ng_backend_map_mutex_;
vector<string> ng_supported_backends =
    ng::runtime::BackendManager::get_registered_backends();
unordered_set<string> BackendManager::ng_supported_backends_(
    ng_supported_backends.begin(), ng_supported_backends.end());

std::map<std::string, int> BackendManager::ref_count_each_backend_;

Status BackendManager::SetBackendName(const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_name_mutex_);
  if (backend_name.empty() || !IsSupportedBackend(backend_name)) {
    return errors::Internal("Backend ", backend_name,
                            " is not supported on nGraph");
  }
  BackendManager::ng_backend_name_ = backend_name;
  return Status::OK();
}

void BackendManager::CreateBackend(const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_map_mutex_);
  auto itr = BackendManager::ng_backend_map_.find(backend_name);
  // if backend does not exist create it
  if (itr == BackendManager::ng_backend_map_.end()) {
    Backend* bend = new Backend;
    std::unique_ptr<ng::runtime::Backend> bend_ptr =
        ng::runtime::Backend::create(backend_name);
    bend->backend_ptr = std::move(bend_ptr);
    BackendManager::ng_backend_map_[backend_name] = bend;
    BackendManager::ref_count_each_backend_[backend_name] = 0;
  }
  BackendManager::ref_count_each_backend_[backend_name]++;

  NGRAPH_VLOG(2) << "BackendManager::CreateBackend(): " << backend_name
                 << " ref_count: "
                 << BackendManager::ref_count_each_backend_[backend_name];
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
unordered_set<string> BackendManager::GetSupportedBackendNames() {
  return ng_supported_backends_;
}

bool BackendManager::IsSupportedBackend(const string& backend_name) {
  auto itr = BackendManager::ng_supported_backends_.find(
      backend_name.substr(0, backend_name.find(':')));
  if (itr == BackendManager::ng_supported_backends_.end()) {
    return false;
  }
  return true;
};

}  // namespace ngraph_bridge
}  // namespace tensorflow
