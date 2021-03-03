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

#include <set>
#include <string>
#include <vector>

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace api {

extern "C" {
extern void enable();
extern void disable();
extern bool is_enabled();

extern size_t backends_len();
extern bool list_backends(char** backends);
extern bool set_backend(const char* backend);
extern bool is_supported_backend(const char* backend);
extern bool get_backend(char** backend);

extern void start_logging_placement();
extern void stop_logging_placement();
extern bool is_logging_placement();

extern void set_disabled_ops(const char* op_type_list);
extern const char* get_disabled_ops();
}

extern void Enable();
extern void Disable();
extern bool IsEnabled();

// TODO: why is this not const?
extern vector<string> ListBackends();
extern bool SetBackend(const string& type);
extern string GetBackend();

extern void StartLoggingPlacement();
extern void StopLoggingPlacement();
extern bool IsLoggingPlacement();

extern std::set<string> GetDisabledOps();
extern void SetDisabledOps(std::set<string>);
extern void SetDisabledOps(string);

}  // namespace api
}  // namespace ngraph_bridge
}  // namespace tensorflow
