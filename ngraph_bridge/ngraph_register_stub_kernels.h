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
#ifndef NGRAPH_TF_BRIDGE_REGISTER_STUB_KERNELS_H_
#define NGRAPH_TF_BRIDGE_REGISTER_STUB_KERNELS_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

/* -------------------------------------------------
//
// NGStubOp
//
---------------------------------------------------*/

class NGStubOp : public OpKernel {
 public:
  explicit NGStubOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* context) override;

 private:
  ~NGStubOp() override;
};

#define REGISTER_NGRAPH_STUB_KERNEL(optype) \
  REGISTER_KERNEL_BUILDER(Name(optype).Device(DEVICE_CPU), NGStubOp);

#define REGISTER_NGRAPH_STUB_BFLOAT_KERNEL(optype)                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name(optype).Device(DEVICE_CPU).TypeConstraint<bfloat16>("T"), \
      NGStubOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_REGISTER_STUB_KERNELS_H_
