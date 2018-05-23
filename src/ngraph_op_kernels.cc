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

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename T>
class NGraphOp : public OpKernel {
 public:
  explicit NGraphOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // SummarizeOp(ctx, std::cout);
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    // std::cout << "Step: " << ctx->step_id()
    //           << " Op: " << ctx->op_kernel().name() << std::endl;
    // std::cout << "NGraphOp::Compute" << std::endl;
  }
};
template <typename T>
class NGraphAddOp : public OpKernel {
 public:
  explicit NGraphAddOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphAddOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // Verify that the tensor shapes match
    // TODO
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphAddOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    VLOG(0) << "Inputs: " << ctx->num_inputs()
            << " Outputs: " << ctx->num_outputs();
    // Get the inputs
    const tf::Tensor& input_tensor_1 = ctx->input(0);
    // const tf::Tensor& input_tensor_2 = ctx->input(1);

    // DO the Math

    // Save the output
    // Create an output tensor
    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor_1.shape(), &output_tensor));
  }
};

template <typename T>
class NGraphMulOp : public OpKernel {
 public:
  explicit NGraphMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphMulOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphMulOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    VLOG(0) << "Inputs: " << ctx->num_inputs()
            << " Outputs: " << ctx->num_outputs();
    // Get the inputs
    const tf::Tensor& input_tensor_1 = ctx->input(0);
    // const tf::Tensor& input_tensor_2 = ctx->input(1);

    // DO the Math

    // Save the output
    // Create an output tensor
    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor_1.shape(), &output_tensor));
  }
};

class NGraphNoOp : public OpKernel {
 public:
  explicit NGraphNoOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NGraphNoOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // SummarizeOp(ctx, std::cout);
  }

  void Compute(OpKernelContext* ctx) override {}
};

class NgPlaceholderOp : public OpKernel {
 public:
  explicit NgPlaceholderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NgPlaceholderOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // SummarizeOp(ctx, std::cout);
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphNoOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    // std::cout << "NgPlaceholderOp: Step: " << ctx->step_id()
    //           << " Op: " << ctx->op_kernel().name() << std::endl;
  }
};

class NGraphConstantOp : public OpKernel {
 public:
  explicit NGraphConstantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto node_def = ctx->def();
    VLOG(0) << "NgPlaceholderOp::ctor(): Node: " << node_def.name()
            << " Op: " << node_def.op();
    // SummarizeNodeDef(def());
  }
  void Compute(OpKernelContext* ctx) override {
    VLOG(0) << "NGraphNoOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
  }
  ~NGraphConstantOp() override {}

 private:
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
      : tf::OpKernel(ctx, StripTensorDataFromNodeDef(ctx)) {}
  void Compute(tf::OpKernelContext* ctx) override {
    // Get the input tensor
    // const tf::Tensor& input_tensor = ctx->input(0);

    // Create an output tensor
    tf::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, ctx->input(0).shape(), &output_tensor));
  }
  bool IsExpensive() override { return false; }
  ~NGraphConstOp() override{};

 private:
  // Tensor tensor_;
};

// This form allows you to specify a list of types as the constraint.
REGISTER_KERNEL_BUILDER(Name("Add")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint("T", {DT_FLOAT}),
                        NGraphAddOp<float>);
REGISTER_KERNEL_BUILDER(Name("Mul")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint("T", {DT_FLOAT}),
                        NGraphMulOp<float>);

// REGISTER_KERNEL_BUILDER(
//    Name("Assign").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint("T",
//    {DT_FLOAT}),
//    NGraphOp<float>);
// REGISTER_KERNEL_BUILDER(
//    Name("ApplyAdam").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint("T",
//    {DT_FLOAT}),
//    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Fill")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint("T", {DT_FLOAT}),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("NoOp").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
                        NGraphNoOp);
REGISTER_KERNEL_BUILDER(
    Name("Placeholder").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
    NgPlaceholderOp);
REGISTER_KERNEL_BUILDER(
    Name("PlaceholderV2").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
    NgPlaceholderOp);

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
                        NGraphOp<int32>);

/*REGISTER_KERNEL_BUILDER(Name("ArgMax")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<float>);*/

REGISTER_KERNEL_BUILDER(Name("Reshape")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tshape"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("ConcatV2")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Mean")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Slice")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Index"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("Slice")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Index"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Maximum")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("Sub")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("Sum")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tidx"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Pack")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphOp<int32>);

/*REGISTER_KERNEL_BUILDER(
    Name("RefSwitch").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Switch").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<bool>("T"),
    NGraphOp<bool>);

REGISTER_KERNEL_BUILDER(
    Name("Switch").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<float>("T"),
    NGraphOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("Merge").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<float>("T"),
    NGraphOp<float>);*/

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

#define REGISTER_IDENTITY(T)                                            \
  REGISTER_KERNEL_BUILDER(Name("Identity")                              \
                              .Device(ngraph_bridge::DEVICE_NGRAPH_CPU) \
                              .TypeConstraint<float>("T"),              \
                          NGraphIdentityOp);

REGISTER_IDENTITY(float);

REGISTER_KERNEL_BUILDER(Name("Snapshot")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("StopGradient")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Conv2D")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("BroadcastGradientArgs")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Relu")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("MaxPool")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("MatMul")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

/*REGISTER_KERNEL_BUILDER(Name("Equal")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int64>("T"),
                        NGraphOp<int64>);

REGISTER_KERNEL_BUILDER(Name("Equal")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<bool>);*/

REGISTER_KERNEL_BUILDER(Name("Greater")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<bool>);

REGISTER_KERNEL_BUILDER(Name("Neg")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("LogSoftmax")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

/*REGISTER_KERNEL_BUILDER(
    Name("ScalarSummary").Device(ngraph_bridge::DEVICE_NGRAPH_CPU).TypeConstraint<float>("T"),
    NGraphOp<float>);*/

REGISTER_KERNEL_BUILDER(Name("ReluGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("MaxPoolGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("ZerosLike")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("FloorDiv")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("RealDiv")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

/*REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<bool>("SrcT")
                            .TypeConstraint<float>("DstT"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("SrcT")
                            .TypeConstraint<float>("DstT"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int64>("SrcT")
                            .TypeConstraint<int32>("DstT"),
                        NGraphOp<int32>);

REGISTER_KERNEL_BUILDER(Name("Cast")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<bool>("SrcT")
                            .TypeConstraint<int32>("DstT"),
                        NGraphOp<int32>);*/

REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tmultiples"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tdim"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNorm")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("L2Loss")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("AddN")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("AvgPool")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Sub")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Pad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tpaddings"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("Greater")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphOp<bool>);

REGISTER_KERNEL_BUILDER(Name("LessEqual")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<int32>("T"),
                        NGraphOp<bool>);

REGISTER_KERNEL_BUILDER(
    Name("LogicalAnd").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
    NGraphOp<bool>);

REGISTER_KERNEL_BUILDER(Name("Select")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("BiasAdd")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("BiasAddGrad")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T"),
                        NGraphOp<float>);

REGISTER_KERNEL_BUILDER(Name("OneHot")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int64>("TI"),
                        NGraphOp<int64>);

REGISTER_KERNEL_BUILDER(Name("Transpose")
                            .Device(ngraph_bridge::DEVICE_NGRAPH_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<int32>("Tperm"),
                        NGraphOp<int32>);

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
