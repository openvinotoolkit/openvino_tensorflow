/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
/*****************************************************************************/

#include "backend.h"

#include <ie_core.hpp>
#include "contexts.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace openvino_tensorflow {

static unique_ptr<GlobalContext> g_global_context;

Backend::Backend(const string& config) {
  string device = config.substr(0, config.find("_"));
  string prec = "";
  if (config.find("_") != string::npos)
    prec = config.substr(config.find("_") + 1);
  InferenceEngine::Core core;
  auto devices = core.GetAvailableDevices();
  // TODO: Handle multiple devices

  bool dev_found = false;
  if (find(devices.begin(), devices.end(), device) == devices.end()) {
    if (device == "MYRIAD") {
      for (auto dev : devices) {
        if (dev.find(device) != std::string::npos) dev_found = true;
      }
    }
  } else {
    dev_found = true;
  }

  if (!dev_found) {
    stringstream ss;
    ss << "Device '" << config << "' not found.";
    throw runtime_error(ss.str());
  }

  if (device == "GPU" && prec != "" && prec != "FP16") {
    stringstream ss;
    if (prec == "FP32") {
      ss << "'GPU_FP32' is not a supported device name."
         << " Please use 'GPU' device name for FP32 precision.";
      throw runtime_error(ss.str());
    } else {
      ss << "The precision '" << prec << "' is not supported on 'GPU'.";
      throw runtime_error(ss.str());
    }
  } else if (device != "GPU" && prec != "") {
    stringstream ss;
    ss << "Device '" << device << "' does not support custom precisions.";
    throw runtime_error(ss.str());
  }

  m_device_type = config;
  if (config.find("MYRIAD") != std::string::npos) {
    m_device = "MYRIAD";
  } else {
    m_device = device;
  }
}

shared_ptr<Executable> Backend::Compile(shared_ptr<ngraph::Function> func,
                                        bool) {
  return make_shared<Executable>(func, m_device, m_device_type);
}

GlobalContext& Backend::GetGlobalContext() {
  if (!g_global_context)
    g_global_context = unique_ptr<GlobalContext>(new GlobalContext);
  return *g_global_context;
}

void Backend::ReleaseGlobalContext() { g_global_context.reset(); }

std::string Backend::GetDeviceType() { return m_device_type; }

bool Backend::IsSupported(const Node& node) const {
  // TODO: check if the given backend/device supports the op. Right now we're
  // assuming
  // that the selected backend supports all opset5 ops
  const auto& opset = ngraph::get_opset5();
  return opset.contains_op_type(&node);
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow
