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
// NOTE: Check out the Optimizer registration phase
// The macro is: REGISTER_OPTIMIZATION
//
// This is one way to get the graph and do the necessary analysis, fusion etc
// Check out:
// tensorflow/compiler/jit/build_xla_launch_ops_pass.cc
// to see how XLALaunchOps are used. Can we do something similar?

namespace ngraph_bridge {
extern const char* const DEVICE_NGRAPH_CPU;
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

class NGraphNoOp : public OpKernel {
 public:
  explicit NGraphNoOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {}
};

//----------------------------- Taken from TF implementation
//------------------------
namespace {

std::unique_ptr<const tf::NodeDef> StripTensorDataFromNodeDef(
    tf::OpKernelConstruction* ctx) {
  const tf::NodeDef& original = ctx->def();
  NodeDef* ret = new NodeDef;
  ret->set_name(original.name());
  ret->set_op(original.op());
  ret->set_device(original.device());
  // Strip the "value" attr from the returned NodeDef.
  // NOTE(mrry): The present implementation of `OpKernel::OpKernel()` only uses
  // attrs that affect the cardinality of list-typed inputs and outputs, so it
  // is safe to drop other attrs from the NodeDef.
  tf::AddNodeAttr("dtype", ctx->output_type(0), ret);
  return std::unique_ptr<const tf::NodeDef>(ret);
}
}  // namespace
//-----------------------------------------------------------------------------

// ConstantOp returns a tensor specified by ConstantOpDef.
class NGraphConstOp : public tf::OpKernel {
 public:
  explicit NGraphConstOp(tf::OpKernelConstruction* ctx)
      : tf::OpKernel(ctx, StripTensorDataFromNodeDef(ctx)),
        m_tensor(ctx->output_type(0)) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
    OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(
                            *proto, AllocatorAttributes(), &m_tensor));
    OP_REQUIRES(
        ctx, ctx->output_type(0) == m_tensor.dtype(),
        errors::InvalidArgument(
            "Type mismatch between value (", DataTypeString(m_tensor.dtype()),
            ") and dtype (", DataTypeString(ctx->output_type(0)), ")"));
  }
  void Compute(tf::OpKernelContext* ctx) override {
    ctx->set_output(0, m_tensor);
    if (TF_PREDICT_FALSE(ctx->track_allocations())) {
      ctx->record_persistent_memory_allocation(m_tensor.AllocatedBytes());
    }
  }
  bool IsExpensive() override { return false; }
  ~NGraphConstOp() override{};

 private:
  Tensor m_tensor;
};

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

// This form allows you to specify a list of types as the constraint.
REGISTER_KERNEL_BUILDER(Name("Add")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint("T", {DT_FLOAT}),
                        NGraphStubOp);
REGISTER_KERNEL_BUILDER(Name("Mul")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint("T", {DT_FLOAT}),
                        NGraphStubOp);

// REGISTER_KERNEL_BUILDER(
//    Name("Assign").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint("T",
//    {DT_FLOAT}),
//    NGraphStubOp);
// REGISTER_KERNEL_BUILDER(
//    Name("ApplyAdam").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint("T",
//    {DT_FLOAT}),
//    NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Fill")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint("T", {DT_FLOAT}),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("NoOp").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
                        NGraphNoOp);
REGISTER_KERNEL_BUILDER(
    Name("Placeholder").Device(ngraph_bridge::DEVICE_NGRAPH_CPU), NGraphStubOp);
REGISTER_KERNEL_BUILDER(
    Name("PlaceholderV2").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
    NGraphStubOp);

#define REGISTER_CONST(T)                                               \
  REGISTER_KERNEL_BUILDER(Name("Const")                                 \
                              .Device(ngraph_bridge::DEVICE_NGRAPH_CPU) \
                              .TypeConstraint<T>("dtype"),              \
                          NGraphConstOp);

REGISTER_CONST(float);
REGISTER_CONST(int32);

// REGISTER_KERNEL_BUILDER(Name("IsVariableInitialized").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
//                         NGraphNoOp);

REGISTER_KERNEL_BUILDER(Name("Prod")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphStubOp);

/*REGISTER_KERNEL_BUILDER(Name("ArgMax")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphStubOp);*/

REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tshape"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Mean")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Slice")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Index"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Slice")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Index"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Maximum")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Sub")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Sum")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Pack")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphStubOp);

/*REGISTER_KERNEL_BUILDER(
    Name("RefSwitch").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<float>("T"),
    NGraphStubOp);

REGISTER_KERNEL_BUILDER(
    Name("Switch").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<bool>("T"),
    NGraphStubOp);

REGISTER_KERNEL_BUILDER(
    Name("Switch").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<float>("T"),
    NGraphStubOp);

REGISTER_KERNEL_BUILDER(
    Name("Merge").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<float>("T"),
    NGraphStubOp);*/

#define REGISTER_IDENTITY(T)                                            \
  REGISTER_KERNEL_BUILDER(Name("Identity")                              \
                              .Device(ngraph_bridge::DEVICE_NGRAPH_CPU) \
                              .TypeConstraint<float>("T"),              \
                          NGraphIdentityOp);

REGISTER_IDENTITY(float);

REGISTER_KERNEL_BUILDER(Name("Snapshot")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("StopGradient")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Conv2D")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("BroadcastGradientArgs")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphStubOp);
REGISTER_KERNEL_BUILDER(Name("Relu")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("MaxPool")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("MatMul")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

/*REGISTER_KERNEL_BUILDER(Name("Equal")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int64>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Equal")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);*/

REGISTER_KERNEL_BUILDER(Name("Greater")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Neg")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("LogSoftmax")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

/*REGISTER_KERNEL_BUILDER(
    Name("ScalarSummary").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<float>("T"),
    NGraphStubOp);*/

REGISTER_KERNEL_BUILDER(Name("ReluGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("MaxPoolGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("ZerosLike")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("FloorDiv")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("RealDiv")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

/*REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<bool>("SrcT")
                            .TypeConstraint<float>("DstT"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("SrcT")
                            .TypeConstraint<float>("DstT"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int64>("SrcT")
                            .TypeConstraint<int32>("DstT"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<bool>("SrcT")
                            .TypeConstraint<int32>("DstT"),
                        NGraphStubOp);*/

REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tmultiples"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tdim"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNorm")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("L2Loss")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("AddN")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("AvgPool")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Sub")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Pad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tpaddings"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Greater")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("LessEqual")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(
    Name("LogicalAnd").Device(ngraph_bridge::DEVICE_NGRAPH_CPU), NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Select")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("BiasAdd")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("BiasAddGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("OneHot")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int64>("TI"),
                        NGraphStubOp);

REGISTER_KERNEL_BUILDER(Name("Transpose")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tperm"),
                        NGraphStubOp);

#define REGISTER_NG_KERNEL(NAME, TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name((NAME))                                  \
                              .Device(ngraph_bridge::DEVICE_NGRAPH_CPU) \
                              .TypeConstraint<TYPE>("dtype"),           \
                          NGraphNoOp);
// REGISTER_NG_KERNEL("Const", float);
// REGISTER_NG_KERNEL("VariableV2", float);
REGISTER_NG_KERNEL("RandomUniform", float);

// REGISTER_KERNEL_BUILDER(Name("MergeSummary").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
// NGraphNoOp);
