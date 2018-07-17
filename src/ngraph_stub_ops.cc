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

#define REGISTER_NGRAPH_STUB(name)                                     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(name).Device(ngraph_bridge::DEVICE_NGRAPH).Label("ngraph"), \
      NGraphStubOp);

//
// Please keep these in alphabetical order.
//

REGISTER_NGRAPH_STUB("Abs");
REGISTER_NGRAPH_STUB("Add");
REGISTER_NGRAPH_STUB("AvgPool");
REGISTER_NGRAPH_STUB("BiasAdd");
REGISTER_NGRAPH_STUB("Cast");
REGISTER_NGRAPH_STUB("ConcatV2");
// "Const" can occur outside of clusters and is registered elsewhere.
REGISTER_NGRAPH_STUB("Conv2D");
REGISTER_NGRAPH_STUB("DepthwiseConv2dNative");
REGISTER_NGRAPH_STUB("Equal");
REGISTER_NGRAPH_STUB("Exp");
REGISTER_NGRAPH_STUB("ExpandDims");
REGISTER_NGRAPH_STUB("Fill")
REGISTER_NGRAPH_STUB("Floor");
REGISTER_NGRAPH_STUB("FusedBatchNorm");
REGISTER_NGRAPH_STUB("Greater");
REGISTER_NGRAPH_STUB("GreaterEqual");
REGISTER_NGRAPH_STUB("Less");
REGISTER_NGRAPH_STUB("LessEqual");
REGISTER_NGRAPH_STUB("Log");
REGISTER_NGRAPH_STUB("LogicalAnd");
REGISTER_NGRAPH_STUB("MatMul");
REGISTER_NGRAPH_STUB("Maximum");
REGISTER_NGRAPH_STUB("MaxPool");
REGISTER_NGRAPH_STUB("Mean");
REGISTER_NGRAPH_STUB("Mul");
// "NoOp" can occur outside of clusters and is registered elsewhere.
REGISTER_NGRAPH_STUB("Pad");
REGISTER_NGRAPH_STUB("Pow");
REGISTER_NGRAPH_STUB("Prod");
REGISTER_NGRAPH_STUB("RealDiv");
REGISTER_NGRAPH_STUB("Relu");
REGISTER_NGRAPH_STUB("Relu6");
REGISTER_NGRAPH_STUB("Reshape");
REGISTER_NGRAPH_STUB("Slice");
REGISTER_NGRAPH_STUB("Sign");
REGISTER_NGRAPH_STUB("Sigmoid");
REGISTER_NGRAPH_STUB("Softmax");
REGISTER_NGRAPH_STUB("Snapshot");
REGISTER_NGRAPH_STUB("Squeeze");
REGISTER_NGRAPH_STUB("StridedSlice");
REGISTER_NGRAPH_STUB("Sub");
REGISTER_NGRAPH_STUB("Sum");
REGISTER_NGRAPH_STUB("Tanh");
REGISTER_NGRAPH_STUB("Transpose");
