/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/sendrecv_ops.h"

namespace tensorflow{

class XPUAddOp : public OpKernel {
 public:
  explicit XPUAddOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    LOG(ERROR) << "-------> XPUAddOp::Compute()";
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, ctx->input(0).shape(), &output));
    output->flat<float>().data()[0] = 12345;
  }
};
REGISTER_KERNEL_BUILDER(Name("Add").Device("XPU"), XPUAddOp);

class XPUNoOp : public OpKernel {
 public:
  explicit XPUNoOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    LOG(ERROR) << "-------> XPUNoOp::Compute()";
  }
};
REGISTER_KERNEL_BUILDER(Name("NoOp").Device("XPU"), XPUNoOp);

}  // namespace tensorflow
