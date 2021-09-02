/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
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
    #define EXPORT_SYMBOL __attribute__((visibility("default")))
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
}

extern EXPORT_SYMBOL void Enable();
extern EXPORT_SYMBOL void Disable();
extern EXPORT_SYMBOL bool IsEnabled();

// TODO: why is this not const?
extern EXPORT_SYMBOL vector<string> ListBackends();
extern EXPORT_SYMBOL bool SetBackend(const string& type);
extern EXPORT_SYMBOL string GetBackend();

extern EXPORT_SYMBOL void StartLoggingPlacement();
extern EXPORT_SYMBOL  void StopLoggingPlacement();
extern EXPORT_SYMBOL bool IsLoggingPlacement();

extern std::set<string> GetDisabledOps();
extern void SetDisabledOps(std::set<string>);
extern void SetDisabledOps(string);

}  // namespace api
}  // namespace openvino_tensorflow
}  // namespace tensorflow
