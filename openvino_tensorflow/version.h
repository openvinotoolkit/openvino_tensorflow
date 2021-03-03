/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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
#pragma once

namespace tensorflow {
namespace ngraph_bridge {

extern "C" {
// Returns the ngraph-tensorflow bridge version
const char* version();

// Returns the nGraph version this bridge was compiled with
const char* ngraph_version();

// Returns the 0 if _GLIBCXX_USE_CXX11_ABI wasn't set by the
// compiler (e.g., clang or gcc pre 4.8) or the value of the
// _GLIBCXX_USE_CXX11_ABI set during the compilation time
int cxx11_abi_flag();

// Returns true when nGraph is using Grappler optimizer APIs for
// graph rewriting
bool is_grappler_enabled();

// Returns the tensorflow version
const char* tf_version();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
