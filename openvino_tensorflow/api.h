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
    #define EXPORT_SYMBOL
#endif


using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {
namespace api {

extern "C" {
EXPORT_SYMBOL extern void enable();
EXPORT_SYMBOL extern void disable();
EXPORT_SYMBOL extern bool is_enabled();

EXPORT_SYMBOL extern size_t backends_len();
EXPORT_SYMBOL extern bool list_backends(char** backends);
EXPORT_SYMBOL extern bool set_backend(const char* backend);
EXPORT_SYMBOL extern bool is_supported_backend(const char* backend);
EXPORT_SYMBOL extern bool get_backend(char** backend);

EXPORT_SYMBOL extern void start_logging_placement();
EXPORT_SYMBOL extern void stop_logging_placement();
EXPORT_SYMBOL extern bool is_logging_placement();

EXPORT_SYMBOL extern void set_disabled_ops(const char* op_type_list);
EXPORT_SYMBOL extern const char* get_disabled_ops();
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
}  // namespace openvino_tensorflow
}  // namespace tensorflow
