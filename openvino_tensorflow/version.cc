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

#include <iostream>
#include <string>

#include "tensorflow/core/public/version.h"

#include "ngraph/ngraph.hpp"

#include "version.h"

// nGraph-TensorFlow bridge uses semantic versioning: see http://semver.org/

#define NG_TF_MAJOR_VERSION 0
#define NG_TF_MINOR_VERSION 23
#define NG_TF_PATCH_VERSION 0

// The version suffix is used for pre-release version numbers
// For example before v0.7.0 we may do a pre-release i.e., a release
// candidate such as v0.7.0-rc0
// The code in master will always have the last released version number
// with a suffix of '-master'
#define NG_TF_VERSION_SUFFIX "-rc0"

#define VERSION_STR_HELPER(x) #x
#define VERSION_STR(x) VERSION_STR_HELPER(x)

// e.g. "0.7.0" or "0.7.0-rc0".
#define NG_TF_VERSION_STRING                                    \
  (VERSION_STR(NG_TF_MAJOR_VERSION) "." VERSION_STR(            \
      NG_TF_MINOR_VERSION) "." VERSION_STR(NG_TF_PATCH_VERSION) \
       NG_TF_VERSION_SUFFIX)

extern "C" const char* get_ngraph_version_string();

namespace tensorflow {
namespace ngraph_bridge {

const char* version() { return (NG_TF_VERSION_STRING); }
const char* ngraph_version() { return get_ngraph_version_string(); }

int cxx11_abi_flag() {
#ifdef _GLIBCXX_USE_CXX11_ABI
  return _GLIBCXX_USE_CXX11_ABI;
#else
  return 0;
#endif
}

bool is_grappler_enabled() {
#if defined(NGRAPH_TF_USE_GRAPPLER_OPTIMIZER)
  return true;
#else
  return false;
#endif
}

const char* tf_version() { return (TF_VERSION_STRING); }

}  // namespace ngraph_bridge
}  // namespace tensorflow
