//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <sstream>

#include "ngraph/ngraph.hpp"

#include "ie_backend.h"
#include "ngraph_backend.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

std::mutex Backend::m_mtx;
std::string Backend::s_backend_shared_library_search_directory;

// This finds the full path of the containing shared library
static string find_my_pathname() { return ""; }

Backend::~Backend() {}

std::shared_ptr<ngraph::Node> Backend::get_backend_op(
    const std::string& /* op_name */, ...) {
  std::shared_ptr<ngraph::Node> dummy_node(nullptr);
  return dummy_node;
}

std::shared_ptr<Backend> Backend::create(const string& t,
                                         bool must_support_dynamic) {
  // Rewrite backend name BACKEND_OPTION to BACKEND:OPTION
  string type = t;
  auto pos = type.find('_');
  if (pos != string::npos) {
    type = type.replace(pos, 1, ":");
  }
  return make_shared<IE_Backend>(type);
}

vector<string> Backend::get_registered_devices() {
  return IE_Backend::get_registered_devices();
}

std::shared_ptr<ngraph::runtime::Tensor> Backend::create_dynamic_tensor(
    const ngraph::element::Type& /* element_type */,
    const PartialShape& /* shape */) {
  throw std::invalid_argument("This backend does not support dynamic tensors");
}

std::shared_ptr<Executable> Backend::compile(
    std::shared_ptr<Function> func, ngraph::pass::PassConfig& /* pass_config */,
    bool enable_performance_data) {
  return compile(func, enable_performance_data);
}

bool Backend::is_supported(const Node& /* node */) const {
  // The default behavior is that a backend does not support any ops. If this is
  // not the case
  // then override this method and enhance.
  return false;
}

bool Backend::is_supported_property(const Property /* prop */) const {
  return false;
}

void Backend::remove_compiled_function(std::shared_ptr<Executable> /* exec */) {
}

std::shared_ptr<Executable> Backend::load(istream& /* input_stream */) {
  throw runtime_error("load operation unimplemented.");
}

bool Backend::is_device_memory(void* /* ptr */) {
  // override this method for each supported backend to determine if the passed
  // pointer is in
  // device pinned memory or not
  return false;
}

void Backend::set_backend_shared_library_search_directory(const string& path) {
  std::lock_guard<std::mutex> lock(Backend::m_mtx);
  s_backend_shared_library_search_directory = path;
}

const string& Backend::get_backend_shared_library_search_directory() {
  if (s_backend_shared_library_search_directory.empty()) {
    s_backend_shared_library_search_directory = find_my_pathname();
  }
  return s_backend_shared_library_search_directory;
}

bool Backend::set_config(const map<string, string>& /* config */,
                         string& error) {
  error = "set_config not supported";
  return false;
}

}  // namespace ngraph_bridge
}  // namespace tensorflow