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

#include <dlfcn.h>
#include <chrono>
#include <iostream>

#include "gtest/gtest.h"

#include "tensorflow/core/platform/env.h"

#ifdef __APPLE__
#define EXT "dylib"
#else
#define EXT "so"
#endif

using namespace std;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Load the ngraph bridge
  //   void* handle;
  //   auto result = tensorflow::Env::Default()->LoadLibrary(
  //       "/home/avijitch/Projects-Mac/ngraph-tensorflow/bazel-bin/tensorflow/"
  //       "libtensorflow_framework.so",
  //       &handle);
  //   if (result != tensorflow::Status::OK()) {
  //     cout << "Cannot load Plugin library: " << result.error_message() <<
  //     endl; return -1;
  //   }

  //   handle = dlopen(
  //       "/home/avijitch/Projects-Mac/ngraph-tensorflow/bazel-bin/tensorflow/cc/"
  //       "libcc_ops.so",
  //       RTLD_LAZY | RTLD_LOCAL);
  //   if (handle == nullptr) {
  //     cout << "Library load failed: " << dlerror() << endl;
  //     return -1;
  //   }

  void* handle;
  auto result =
      tensorflow::Env::Default()->LoadLibrary("libngraph_device." EXT, &handle);
  if (result != tensorflow::Status::OK()) {
    cout << "Cannot load library: " << result.error_message() << endl;
    return -1;
  }

  int rc = RUN_ALL_TESTS();
  return rc;
}
