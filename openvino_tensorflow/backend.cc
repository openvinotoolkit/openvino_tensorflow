/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
/*****************************************************************************/

#include "backend.h"

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"
#include "contexts.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace openvino_tensorflow {


static unique_ptr<GlobalContext> g_global_context;

Backend::Backend(const string& config) {
  string device = config.substr(0, config.find(":"));
  InferenceEngine::Core core;
  auto devices = core.GetAvailableDevices();
  // TODO: Handle multiple devices

  bool dev_found = false;
  if (find(devices.begin(), devices.end(), device) == devices.end()) {

    if(device == "MYRIAD"){
      for(auto dev : devices){
        if(dev.find(device) != std::string::npos)
          dev_found = true;
      }
    }
  }
  else{
    dev_found = true;
  }

  if(!dev_found){

    stringstream ss;
    ss << "Device '" << config << "' not found.";
    throw runtime_error(ss.str());

  }
  Backend::GetGlobalContext().device_type = config;
  if(config.find("MYRIAD") != std::string::npos){
    m_device = "MYRIAD";
  }
  else{
    m_device = config;
  }
}

shared_ptr<Executable> Backend::Compile(shared_ptr<ngraph::Function> func,
                                        bool) {
  return make_shared<Executable>(func, m_device);
}

GlobalContext& Backend::GetGlobalContext() {

  if(!g_global_context)
    g_global_context = unique_ptr<GlobalContext>(new GlobalContext);
  return *g_global_context;
}

void Backend::ReleaseGlobalContext() {
  g_global_context.reset();
}

bool Backend::IsSupported(const Node& node) const {
  // TODO: check if the given backend/device supports the op. Right now we're
  // assuming
  // that the selected backend supports all opset5 ops
  const auto& opset = ngraph::get_opset5();
  return opset.contains_op_type(&node);
}

}  // namespace openvino_tensorflow
}  // namespace tensorflow