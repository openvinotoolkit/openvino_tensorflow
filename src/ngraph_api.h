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
}
}
}
