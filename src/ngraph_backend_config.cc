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

#include "ngraph_backend_config.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

BackendConfig::BackendConfig(const string& backend_name) {
  NGRAPH_VLOG(3) << "BackendConfig() ";
  backend_name_ = backend_name;
  additional_attributes_ = {"device_config"};
}

string BackendConfig::Join(
    const unordered_map<string, string>& additional_parameters) {
  // If device_config is not found throw an error
  try {
    additional_parameters.at("device_config");
  } catch (std::out_of_range e1) {
    throw std::out_of_range("Attribute device_config not found");
  }
  return backend_name_ + ":" + additional_parameters.at("device_config");
}

unordered_map<string, string> BackendConfig::Split(
    const string& backend_config) {
  unordered_map<string, string> backend_parameters;

  int delimiter_index = backend_config.find(':');
  if (delimiter_index < 0) {
    // ":" not found
    backend_parameters["ngraph_backend"] = backend_config;
    backend_parameters["_ngraph_device_config"] = "";
  } else {
    backend_parameters["ngraph_backend"] =
        backend_config.substr(0, delimiter_index);
    backend_parameters["_ngraph_device_config"] =
        backend_config.substr(delimiter_index + 1);
  }

  NGRAPH_VLOG(3) << "Got Backend Name " << backend_parameters["ngraph_backend"];
  NGRAPH_VLOG(3) << "Got Device Config  "
                 << backend_parameters["_ngraph_device_config"];

  return backend_parameters;
}

vector<string> BackendConfig::GetAdditionalAttributes() {
  return BackendConfig::additional_attributes_;
}

BackendConfig::~BackendConfig() {
  NGRAPH_VLOG(2) << "BackendConfig::~BackendConfig() DONE";
};

// BackendNNPIConfig
BackendNNPIConfig::BackendNNPIConfig() : BackendConfig("NNPI") {
  additional_attributes_ = {"device_id", "ice_cores", "max_batch_size"};
}

string BackendNNPIConfig::Join(
    const unordered_map<string, string>& additional_parameters) {
  // If device_id is not found throw an error
  try {
    additional_parameters.at("device_id");
  } catch (std::out_of_range e1) {
    throw std::out_of_range("Attribute device_id not found");
  }
  return backend_name_ + ":" + additional_parameters.at("device_id");

  // Once the backend api for the other attributes like ice cores
  // and max batch size is fixed we change this
}

BackendNNPIConfig::~BackendNNPIConfig() {
  NGRAPH_VLOG(3) << "BackendNNPIConfig::~BackendNNPIConfig() DONE";
};

// BackendInterpreterConfig
BackendInterpreterConfig::BackendInterpreterConfig()
    : BackendConfig("INTERPRETER") {
  additional_attributes_ = {"test_echo"};
}

string BackendInterpreterConfig::Join(
    const unordered_map<string, string>& additional_parameters) {
  NGRAPH_VLOG(3) << "BackendInterpreterConfig::Join - return the backend name";
  return backend_name_;
}

unordered_map<string, string> BackendInterpreterConfig::Split(
    const string& backend_config) {
  unordered_map<string, string> backend_parameters;

  int delimiter_index = backend_config.find(':');
  if (delimiter_index < 0) {
    // ":" not found
    backend_parameters["ngraph_backend"] = backend_config;
    backend_parameters["_ngraph_test_echo"] = "";
  } else {
    backend_parameters["ngraph_backend"] =
        backend_config.substr(0, delimiter_index);
    backend_parameters["_ngraph_test_echo"] =
        backend_config.substr(delimiter_index + 1);
  }
  return backend_parameters;
}

BackendInterpreterConfig::~BackendInterpreterConfig() {
  NGRAPH_VLOG(3)
      << "BackendInterpreterConfig::~BackendInterpreterConfig() DONE";
};

}  // namespace ngraph_bridge
}  // namespace tensorflow