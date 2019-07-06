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

#ifndef NGRAPH_TF_BRIDGE_BACKEND_CONFIG_H_
#define NGRAPH_TF_BRIDGE_BACKEND_CONFIG_H_

#include <ostream>

#include "tensorflow/core/lib/core/errors.h"

#include "ngraph_log.h"
using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

class BackendConfig {
 public:
  BackendConfig() = delete;
  BackendConfig(const string& backend_name);
  vector<string> GetAdditionalAttributes();

  virtual unordered_map<string, string> Split(const string& backend_config);
  virtual string Join(
      const unordered_map<string, string>& additional_parameters);
  virtual ~BackendConfig();

 protected:
  string backend_name_;
  vector<string> additional_attributes_;
};

class BackendNNPIConfig : public BackendConfig {
 public:
  BackendNNPIConfig();
  string Join(
      const unordered_map<string, string>& additional_parameters) override;
  virtual ~BackendNNPIConfig();
};

class BackendInterpreterConfig : public BackendConfig {
 public:
  BackendInterpreterConfig();
  unordered_map<string, string> Split(const string& backend_config) override;
  string Join(
      const unordered_map<string, string>& additional_parameters) override;
  virtual ~BackendInterpreterConfig();
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif
// NGRAPH_TF_BRIDGE_BACKEND_CONFIG_H