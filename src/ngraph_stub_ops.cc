/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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

#include "ngraph_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/platform/default/logging.h"

//
// Any nGraph op that is supported when placed inside a cluster, but _not_
// supported when placed outside of a cluster, should be registered here with
// a stub OpKernel.
//

namespace ngraph_bridge {
extern const char* const DEVICE_NGRAPH;
}

using namespace tensorflow;

class NGraphStubOp : public OpKernel {
 public:
  explicit NGraphStubOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx, false,
                errors::Internal("NGraphStubOp compute kernel called"));
  }
};

#define REGISTER_NGRAPH_STUB(builder) \
  REGISTER_KERNEL_BUILDER(builder.Label("ngraph"), NGraphStubOp);

//
// Please keep these in alphabetical order.
//

REGISTER_NGRAPH_STUB(Name("Abs")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_FLOAT, DT_INT32}));
REGISTER_NGRAPH_STUB(Name("Add")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_FLOAT, DT_INT32, DT_INT64}));
REGISTER_NGRAPH_STUB(Name("AvgPool")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T"));
REGISTER_NGRAPH_STUB(Name("BiasAdd")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T"));
REGISTER_NGRAPH_STUB(Name("ConcatV2")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_INT32, DT_FLOAT})
                         .TypeConstraint("Tidx", {DT_INT32, DT_INT64}));
// "Const" can occur outside of clusters and is registered elsewhere.
REGISTER_NGRAPH_STUB(Name("Conv2D")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T"));
// REGISTER_NGRAPH_STUB(Name("DepthwiseConv2dNative")
//                         .Device(ngraph_bridge::DEVICE_NGRAPH)
//                         .TypeConstraint<float>("T"));
REGISTER_NGRAPH_STUB(Name("Equal")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_FLOAT, DT_INT32, DT_INT64}));
REGISTER_NGRAPH_STUB(Name("Floor")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T"));
REGISTER_NGRAPH_STUB(Name("FusedBatchNorm")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T"));
// "Identity" can occur outside of clusters and is registered elsewhere.
REGISTER_NGRAPH_STUB(Name("MatMul")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T"));
REGISTER_NGRAPH_STUB(Name("MaxPool")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T"));
REGISTER_NGRAPH_STUB(Name("Mean")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T")
                         .TypeConstraint("Tidx", {DT_INT32, DT_INT64}));
REGISTER_NGRAPH_STUB(Name("Mul")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_FLOAT, DT_INT32}));
// "NoOp" can occur outside of clusters and is registered elsewhere.
REGISTER_NGRAPH_STUB(Name("Pad")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T")
                         .TypeConstraint("Tpaddings", {DT_INT32, DT_INT64}));
REGISTER_NGRAPH_STUB(Name("Relu")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T"));
REGISTER_NGRAPH_STUB(Name("Relu6")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T"));
REGISTER_NGRAPH_STUB(Name("Reshape")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_FLOAT, DT_INT32, DT_INT64})
                         .TypeConstraint("Tshape", {DT_INT32, DT_INT64}));
REGISTER_NGRAPH_STUB(Name("Sign")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_FLOAT, DT_INT32}));
REGISTER_NGRAPH_STUB(Name("Snapshot")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_FLOAT, DT_INT32}));
REGISTER_NGRAPH_STUB(Name("Squeeze")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_FLOAT, DT_INT32}));
REGISTER_NGRAPH_STUB(Name("Sum")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint("T", {DT_FLOAT, DT_INT32})
                         .TypeConstraint("Tidx", {DT_INT32, DT_INT64}));
REGISTER_NGRAPH_STUB(Name("Transpose")
                         .Device(ngraph_bridge::DEVICE_NGRAPH)
                         .TypeConstraint<float>("T")
                         .TypeConstraint("Tperm", {DT_INT32, DT_INT64}));
