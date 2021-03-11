/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>
#include <string>

#include "tensorflow/core/public/version.h"

#include "ngraph/ngraph.hpp"

#include "version.h"

// openvino_tensorflow add-on uses semantic versioning: see http://semver.org/

#define NG_TF_MAJOR_VERSION 0
#define NG_TF_MINOR_VERSION 23
#define NG_TF_PATCH_VERSION 0

// The version suffix is used for pre-release version numbers
// For example before v0.7.0 we may do a pre-release i.e., a release
// candidate such as v0.7.0-rc0
// The code in master will always have the last released version number
// with a suffix of '-master'
#define OV_TF_VERSION_SUFFIX "-rc0"

#define VERSION_STR_HELPER(x) #x
#define VERSION_STR(x) VERSION_STR_HELPER(x)

// e.g. "0.7.0" or "0.7.0-rc0".
#define OV_TF_VERSION_STRING                                    \
  (VERSION_STR(OV_TF_MAJOR_VERSION) "." VERSION_STR(            \
      OV_TF_MINOR_VERSION) "." VERSION_STR(OV_TF_PATCH_VERSION) \
       OV_TF_VERSION_SUFFIX)

extern "C" const char* get_ngraph_version_string();

namespace tensorflow {
namespace openvino_tensorflow {

const char* version() { return (OV_TF_VERSION_STRING); }
const char* ngraph_version() { return get_ngraph_version_string(); }

int cxx11_abi_flag() {
#ifdef _GLIBCXX_USE_CXX11_ABI
  return _GLIBCXX_USE_CXX11_ABI;
#else
  return 0;
#endif
}

bool is_grappler_enabled() {
#if defined(OPENVINO_TF_USE_GRAPPLER_OPTIMIZER)
  return true;
#else
  return false;
#endif
}

const char* tf_version() { return (TF_VERSION_STRING); }

}  // namespace openvino_tensorflow
}  // namespace tensorflow
