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

#include <vector>
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
// Return a fake device with the specified type and name.
class FakeDevice : public Device {
 public:
  explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
  Status Sync() override { return Status::OK(); }
  Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
};

// CPU device implementation.
// class GraphSnifferDevice : public LocalDevice {
//  public:
//   GraphSnifferDevice(const SessionOptions& options, const string& name,
//                    Bytes memory_limit, const DeviceLocality& locality,
//                    Allocator* allocator);
//   ~GraphSnifferDevice() override;

//   void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
//   Allocator* GetAllocator(AllocatorAttributes attr) override;
//   Status MakeTensorFromProto(const TensorProto& tensor_proto,
//                              const AllocatorAttributes alloc_attrs,
//                              Tensor* tensor) override;

//   Status Sync() override { return Status::OK(); }

//  private:
//   Allocator* allocator_;  // Not owned
// };

class DummyFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    std::cout << "DummyFactory::CreateDevices() called: Name: " << name_prefix
              << std::endl;
    DeviceAttributes attr;
    attr.set_name(strings::StrCat(name_prefix, "/device:NGRAPH_CPU:0"));
    attr.set_device_type("NGRAPH_CPU");

    devices->push_back(new FakeDevice(attr));
    return Status::OK();
  }
};

// Assumes the default priority is '50'.
REGISTER_LOCAL_DEVICE_FACTORY("NGRAPH_CPU", DummyFactory, 200);

static bool InitModule() {
  std::cout << "InitModule called" << std::endl;

  // Device* device =
  //     DeviceFactory::NewDevice("NGRAPH_CPU", {}, "/job:a/replica:0/task:0");
  // static DeviceMgr device_mgr({device});

  // tensorflow::SessionOptions options;
  // auto* device_count = options.config.mutable_device_count();
  // device_count->insert({"d4", 1});
  // std::vector<tensorflow::Device*> devices;
  // tensorflow::DeviceFactory::AddDevices(
  //     options, "/job:localhost/replica:0/task:0", &devices);

  // static std::unique_ptr<Device> device_(DeviceFactory::NewDevice(
  //     "d2", {}, "/job:a/replica:0/task:0/device:d2:0"));

  // DeviceFactory::Register(
  //     device_info.XLA_DEVICE_NAME,
  //     new DeviceFactoryAdapter(device_info.PLATFORM_NAME,
  //                              device_info.XLA_DEVICE_NAME,
  //                              device_info.XLA_DEVICE_JIT_NAME),
  //     device_info.device_priority);

  return true;
}
bool not_used = InitModule();

}  // namespace tensorflow
