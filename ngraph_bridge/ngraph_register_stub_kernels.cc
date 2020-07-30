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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "ngraph_bridge/ngraph_register_stub_kernels.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

/* -------------------------------------------------
//
// NGraphStubOp
//
---------------------------------------------------*/
// Constructor
NGStubOp::NGStubOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES(
      context, false,
      errors::Internal("The constructor for OpType ", type_string(),
                       " should not get called. This Op is expected to have "
                       "been encapsulated or replaced by other ops. Op Name: ",
                       name(), "\n"));
}
// Compute
void NGStubOp::Compute(OpKernelContext* context) {
  OP_REQUIRES(
      context, false,
      errors::Internal("This kernel for OpType ", type_string(),
                       " should not get called. This Op is expected to have "
                       "been encapsulated or replaced by other ops. Op Name: ",
                       name(), "\n"));
}
// Destructor
NGStubOp::~NGStubOp() {}

/* ------------------------------------------------- */

// Register Bfloat Stub Kernels

// TF Ops that work on bfloat DataType get assigned Device XLA_CPU
// Since nGraph-bridge OPs work on TF DEVICE_CPU we are registering stub
// bfloat16 kernels here. The expectation is when we register the stub kernels
// for bfloat16 TF is going to assign DEVICE_CPU to the respective Ops and
// we will encapsulate them
// These Stub Kernels/Op will never get called

// Keep them in alphabetical order
REGISTER_NGRAPH_STUB_BFLOAT_KERNEL("Conv2D")

}  // namespace ngraph_bridge

}  // namespace tensorflow
