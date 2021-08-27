/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#pragma once

#include <set>
#include <string>
#include <vector>

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {
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

extern void enable_dynamic_fallback();
extern void disable_dynamic_fallback();

extern bool export_ir(const char* output_dir, char** cluster_info,
                      bool confirm_before_overwrite = true);
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

extern void EnableDynamicFallback();
extern void DisableDynamicFallback();

extern bool ExportIR(const string& output_dir, string& cluster_info,
                     bool confirm_before_overwrite = true);
}  // namespace api
}  // namespace openvino_tensorflow
}  // namespace tensorflow
