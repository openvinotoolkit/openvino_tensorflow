/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <iostream>
#include <string>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class NGraphDeviceContext : public DeviceContext {
 public:
  // Does not take ownership of streams.
  ~NGraphDeviceContext() override {}

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override {
    std::cout << "CopyCPUTensorToDevice: DEVICE: " << device->name()
              << std::endl;

    *device_tensor = *cpu_tensor;
    done(Status::OK());
  }

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override {
    std::cout << "CopyDeviceTensorToCPU: DEVICE: " << device->name()
              << " Edge: " << edge_name << std::endl;

    *cpu_tensor = *device_tensor;
    done(Status::OK());
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override {
    std::cout << "CopyTensorInSameDevice: DEVICE: " << device->name()
              << std::endl;
    *output_tensor = *input_tensor;
    done(Status::OK());
  }

  // Not used.
  // void MaintainLifetimeOnStream(const Tensor* t,
  //                              se::Stream* stream) const override {}

  // Status ThenExecute(Device* device, se::Stream* stream,
  //                   std::function<void()> func) override;
};

class NGraphDevice : public LocalDevice {
 public:
  NGraphDevice(const SessionOptions& options,
               const DeviceAttributes& attributes)
      : LocalDevice(options, attributes) {
    std::cout << "NGraphDevice::ctor CALLED" << std::endl;
  }

  Status Sync() override { return Status::OK(); }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    // QCHECK(false) << "xpu device allocator not implemented.";
    return ProcessState::singleton()->GetCPUAllocator(0);
  }

  Status TryGetDeviceContext(DeviceContext** out_context) override {
    static NGraphDeviceContext* ctx = new NGraphDeviceContext;
    ctx->Ref();
    *out_context = ctx;
    return Status::OK();
  }

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override {
    Tensor parsed(tensor_proto.dtype());
    if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
      return errors::InvalidArgument("Cannot parse tensor from tensor_proto.");
    }
    *tensor = parsed;
    return Status::OK();
  }
};

class NGraphDeviceFactory : public DeviceFactory {
 private:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    devices->emplace_back(new NGraphDevice(
        options,
        Device::BuildDeviceAttributes(name_prefix + "/device:NGRAPH:0",
                                      "NGRAPH", static_cast<Bytes>(2 << 30),
                                      DeviceLocality{}, "NGRAPH Device")));
    return Status::OK();
  }
  // For a specific device factory list all possible physical devices.
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    devices->push_back("/physical_device:NGRAPH:0");
    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("NGRAPH", NGraphDeviceFactory, 210);

}  // namespace tensorflow
