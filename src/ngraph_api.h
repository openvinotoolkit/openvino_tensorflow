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
#pragma once

#include <string.h>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace config {
extern "C" {
extern void ngraph_enable();
extern void ngraph_disable();
extern bool ngraph_is_enabled();

extern size_t ngraph_backends_len();
extern bool ngraph_list_backends(char** backends, int backends_len);
extern bool ngraph_set_backend(const char* backend);

extern void ngraph_start_logging_placement();
extern void ngraph_stop_logging_placement();
extern bool ngraph_is_logging_placement();
}

extern void Enable();
extern void Disable();
extern bool IsEnabled();

extern size_t BackendsLen();
// TODO: why is this not const?
extern vector<string> ListBackends();
extern tensorflow::Status SetBackend(const string& type);

extern void StartLoggingPlacement();
extern void StopLoggingPlacement();
extern bool IsLoggingPlacement();
}  // namespace config
}  // namespace ngraph_bridge
}  // namespace tensorflow
