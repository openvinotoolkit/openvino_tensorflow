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
  std::cout << "OVTF_DEBUG - Backend::Backend - 1" << std::endl;
  string device = config.substr(0, config.find("_"));
  std::cout << "OVTF_DEBUG - Backend::Backend - 2" << std::endl;
  string prec = "";
  std::cout << "OVTF_DEBUG - Backend::Backend - 3" << std::endl;
  if (config.find("_") != string::npos)
    prec = config.substr(config.find("_") + 1);
  std::cout << "OVTF_DEBUG - Backend::Backend - 4" << std::endl;
  ov::Core core;
  std::cout << "OVTF_DEBUG - Backend::Backend - 5" << std::endl;
  auto devices = core.get_available_devices();
  std::cout << "OVTF_DEBUG - Backend::Backend - 6" << std::endl;
  // TODO: Handle multiple devices

  bool dev_found = false;
  std::cout << "OVTF_DEBUG - Backend::Backend - 7" << std::endl;
  if (find(devices.begin(), devices.end(), device) == devices.end()) {
  std::cout << "OVTF_DEBUG - Backend::Backend - 8" << std::endl;
    if (device == "MYRIAD") {
  std::cout << "OVTF_DEBUG - Backend::Backend - 9" << std::endl;
      for (auto dev : devices) {
  std::cout << "OVTF_DEBUG - Backend::Backend - 10" << std::endl;
        if (dev.find(device) != std::string::npos) dev_found = true;
  std::cout << "OVTF_DEBUG - Backend::Backend - 11" << std::endl;
      }
    }
  } else {
  std::cout << "OVTF_DEBUG - Backend::Backend - 12" << std::endl;
    dev_found = true;
  std::cout << "OVTF_DEBUG - Backend::Backend - 13" << std::endl;
  }

  std::cout << "OVTF_DEBUG - Backend::Backend - 14" << std::endl;
  if (!dev_found) {
  std::cout << "OVTF_DEBUG - Backend::Backend - 15" << std::endl;
    stringstream ss;
    ss << "Device '" << config << "' not found.";
    throw runtime_error(ss.str());
  }

  std::cout << "OVTF_DEBUG - Backend::Backend - 16" << std::endl;
  if ((device.find("GPU") != std::string::npos) && prec != "" &&
      prec != "FP16") {  // device == GPU
  std::cout << "OVTF_DEBUG - Backend::Backend - 17" << std::endl;
    stringstream ss;
  std::cout << "OVTF_DEBUG - Backend::Backend - 18" << std::endl;
    if (prec == "FP32") {
  std::cout << "OVTF_DEBUG - Backend::Backend - 19" << std::endl;
      ss << "'GPU_FP32' is not a supported device name."
         << " Please use 'GPU' device name for FP32 precision.";
      throw runtime_error(ss.str());
    } else {
  std::cout << "OVTF_DEBUG - Backend::Backend - 20" << std::endl;
      ss << "The precision '" << prec << "' is not supported on 'GPU'.";
      throw runtime_error(ss.str());
    }
  std::cout << "OVTF_DEBUG - Backend::Backend - 21" << std::endl;
  } else if ((device.find("GPU") == std::string::npos) &&
             prec != "") {  // device != GPU
  std::cout << "OVTF_DEBUG - Backend::Backend - 22" << std::endl;
    stringstream ss;
    ss << "Device '" << device << "' does not support custom precisions.";
    throw runtime_error(ss.str());
  }

  std::cout << "OVTF_DEBUG - Backend::Backend - 23" << std::endl;
  m_device_type = config;
  std::cout << "OVTF_DEBUG - Backend::Backend - 24" << std::endl;
  if (config.find("MYRIAD") != std::string::npos) {
  std::cout << "OVTF_DEBUG - Backend::Backend - 25" << std::endl;
    m_device = "MYRIAD";
  } else {
  std::cout << "OVTF_DEBUG - Backend::Backend - 26" << std::endl;
    m_device = device;
  }
  std::cout << "OVTF_DEBUG - Backend::Backend - 27" << std::endl;
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
