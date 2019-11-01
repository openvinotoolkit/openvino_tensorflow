/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

class NGraphWriteToDeviceOp : public OpKernel {
 public:
  explicit NGraphWriteToDeviceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      std::cout << "INPUT TENSOR " << context->input(0).DebugString()
                << std::endl;
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("NGraphWriteToDevice").Device(DEVICE_CPU),
                        NGraphWriteToDeviceOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
