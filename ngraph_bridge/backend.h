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

#pragma once

#include <memory>
#include <string>

#include "ngraph/ngraph.hpp"

#include "ngraph_bridge/executable.h"
#include "ngraph_bridge/ie_tensor.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

class Backend {
 public:
  Backend(const string& configuration_string);
  ~Backend() {}

  shared_ptr<Executable> Compile(shared_ptr<ngraph::Function> func,
                                 bool enable_performance_data = false);
  bool IsSupported(const ngraph::Node& node) const;

 private:
  string m_device;
};
}
}
