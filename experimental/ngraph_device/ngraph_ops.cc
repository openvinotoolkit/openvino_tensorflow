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

namespace tensorflow {

class NGraphAddOp : public OpKernel {
 public:
  explicit NGraphAddOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    LOG(ERROR) << "-------> NGraphAddOp::Compute()";
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, ctx->input(0).shape(), &output));
    output->flat<float>().data()[0] = 12345;
  }
};
REGISTER_KERNEL_BUILDER(Name("Add").Device("NGRAPH"), NGraphAddOp);

class NGraphMulOp : public OpKernel {
 public:
  explicit NGraphMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    LOG(ERROR) << "-------> NGraphMulOp::Compute()";
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, ctx->input(0).shape(), &output));
    // output->flat<float>().data()[0] = 9999;
  }
};
REGISTER_KERNEL_BUILDER(Name("Mul").Device("NGRAPH"), NGraphMulOp);

class NGraphConstOp : public OpKernel {
 public:
  explicit NGraphConstOp(OpKernelConstruction* ctx) : OpKernel(ctx) ,
  tensor_(ctx->output_type(0)) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
    OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(
                            *proto, AllocatorAttributes(), &tensor_));
    OP_REQUIRES(
        ctx, ctx->output_type(0) == tensor_.dtype(),
        errors::InvalidArgument("Type mismatch between value (",
                                DataTypeString(tensor_.dtype()), ") and dtype (",
                                DataTypeString(ctx->output_type(0)), ")"));

  }
  void Compute(OpKernelContext* ctx) override {
    LOG(ERROR) << "-------> NGraphConstOp::Compute()";

    ctx->set_output(0, tensor_);
    //OP_REQUIRES_OK(ctx,
    //               ctx->allocate_output(0, ctx->input(0).shape(), &output));
    // output->flat<float>().data()[0] = 21212121;
  }
private:
  Tensor tensor_;
};
REGISTER_KERNEL_BUILDER(Name("Const").Device("NGRAPH"), NGraphConstOp);

class NGraphNoOp : public OpKernel {
 public:
  explicit NGraphNoOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    LOG(ERROR) << "-------> NGraphNoOp::Compute()";
  }
};
REGISTER_KERNEL_BUILDER(Name("NoOp").Device("NGRAPH"), NGraphNoOp);

class NGraphIdentityOp : public OpKernel {
 public:
  explicit NGraphIdentityOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("Identity").Device("NGRAPH"), NGraphIdentityOp);

}  // namespace tensorflow
