/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#pragma once

#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
#ifdef BUILD_API
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __declspec(dllimport)
#endif
#else
#ifndef EXPORT_SYMBOL
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif
#endif

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {
namespace api {

extern "C" {
extern EXPORT_SYMBOL void enable();
extern EXPORT_SYMBOL void disable();
extern EXPORT_SYMBOL bool is_enabled();

extern EXPORT_SYMBOL size_t backends_len();
extern EXPORT_SYMBOL bool list_backends(char** backends);
extern EXPORT_SYMBOL bool set_backend(const char* backend);
extern EXPORT_SYMBOL bool is_supported_backend(const char* backend);
extern EXPORT_SYMBOL bool get_backend(char** backend);

extern EXPORT_SYMBOL void start_logging_placement();
extern EXPORT_SYMBOL void stop_logging_placement();
extern EXPORT_SYMBOL bool is_logging_placement();

extern EXPORT_SYMBOL void set_disabled_ops(const char* op_type_list);
extern EXPORT_SYMBOL const char* get_disabled_ops();

extern EXPORT_SYMBOL void enable_dynamic_fallback();
extern EXPORT_SYMBOL void disable_dynamic_fallback();

extern EXPORT_SYMBOL bool export_ir(const char* output_dir, char** cluster_info,
                                    char** err_msg);
extern EXPORT_SYMBOL void load_tf_conversion_extensions(
    const char* tf_conversion_extensions_so_path);
}

extern void Enable();
extern void Disable();
extern bool IsEnabled();

// TODO: why is this not const?
extern EXPORT_SYMBOL vector<string> ListBackends();
extern EXPORT_SYMBOL void SetBackend(const string& type);
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
                     string& err_msg);
extern void LoadTFConversionExtensions(
    const string& tf_conversion_extensions_so_path);
}  // namespace api
}  // namespace openvino_tensorflow
}  // namespace tensorflow
