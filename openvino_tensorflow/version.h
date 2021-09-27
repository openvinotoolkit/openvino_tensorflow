/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#pragma once

namespace tensorflow {
namespace openvino_tensorflow {

extern "C" {
// Returns the version of OpenVINO™ integration with TensorFlow
const char* version();

// Returns the nGraph version this bridge was compiled with
const char* openvino_version();

// Returns the 0 if _GLIBCXX_USE_CXX11_ABI wasn't set by the
// compiler (e.g., clang or gcc pre 4.8) or the value of the
// _GLIBCXX_USE_CXX11_ABI set during the compilation time
int cxx11_abi_flag();

// Returns true when openvino_tensorflow is using Grappler optimizer APIs for
// graph rewriting
bool is_grappler_enabled();

// Returns the tensorflow version
const char* tf_version();
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
