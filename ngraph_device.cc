/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>
#include <sstream>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session_options.h"

#include "tensorflow/core/platform/default/logging.h"

#include "ngraph_utils.h"

namespace tensorflow {
// Return a fake device with the specified type and name.
class NGraphDevice : public Device {
 public:
  explicit NGraphDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
  Status Sync() override { return Status::OK(); }
  Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }

  // Overwrite MaybeRewriteGraph
  Status MaybeRewriteGraph(std::unique_ptr<Graph>* graph) override {
    VLOG(0) << "NGraphDevice::MaybeRewriteGraph() called";

    std::string dot =
        GraphToDot(graph->get(), "NGraphDevice::MaybeRewriteGraph", true);

    static int count = 1;
    std::ostringstream filename;
    filename << "./ngraph_MaybeRewriteGraph_" << count << ".dot";
    count++;

    std::ofstream ostrm(filename.str(), std::ios_base::trunc);
    ostrm << dot;

    // auto graph_def = graph->get()->ToGraphDefDebug();
    // std::cout << "-----------------------------------------------------\n"
    //           << graph_def.DebugString()
    //           << "-----------------------------------------------------\n"
    //           << std::endl;

    // for (Edge const* edge : graph->get()->edges()) {
    //   std::cout << "Edge: " << edge->DebugString() << std::endl;
    // }
    return Status::OK();
  }
};

class NGraphDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    std::cout << "NGraphDeviceFactory::CreateDevices() called: Name: "
              << name_prefix << std::endl;
    DeviceAttributes attr;
    attr.set_name(strings::StrCat(name_prefix, "/device:NGRAPH_CPU:0"));
    attr.set_device_type("NGRAPH_CPU");

    devices->push_back(new NGraphDevice(attr));
    return Status::OK();
  }
};

// Assumes the default priority is '50'.
REGISTER_LOCAL_DEVICE_FACTORY("NGRAPH_CPU", NGraphDeviceFactory, 50);

static bool InitModule() {
  std::cout << "InitModule called" << std::endl;

  return true;
}

volatile bool not_used = InitModule();

}  // namespace tensorflow
