/*****************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
/*****************************************************************************/

#include "backend.h"

#include "contexts.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

static unique_ptr<GlobalContext> g_global_context;

Backend::Backend(const string& config) {
  string device = config.substr(0, config.find("_"));
  string prec = "";
  if (config.find("_") != string::npos)
    prec = config.substr(config.find("_") + 1);
  ov::Core core;
  auto devices = core.get_available_devices();
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
  } else if ((device.find("GPU") != std::string::npos) && prec != "") {
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

shared_ptr<Executable> Backend::Compile(shared_ptr<ov::Model> func, bool) {
  return make_shared<Executable>(func, m_device, m_device_type);
}

GlobalContext& Backend::GetGlobalContext() {
  if (!g_global_context)
    g_global_context = unique_ptr<GlobalContext>(new GlobalContext);
  return *g_global_context;
}

void Backend::ReleaseGlobalContext() { g_global_context.reset(); }

std::string Backend::GetDeviceType() { return m_device_type; }

}  // namespace openvino_tensorflow
}  // namespace tensorflow
