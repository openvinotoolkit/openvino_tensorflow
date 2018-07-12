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
// Kernels for some ops that should work even when they are placed on
// NGRAPH but _not_ assigned to any cluster. This can happen for a few
// reasons, one of which is that the partitioning pass seems to insert things
// like Identity (and perhaps sometimes Const?), _after_ we've done
// encapsulation. Sometimes it also makes sense not to cluster these ops,
// e.g. when doing so would result in a trivial cluster that contains only a
// single Const.
//

namespace ngraph_bridge {
extern const char* const DEVICE_NGRAPH;
}

using namespace tensorflow;

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

class NGraphEnterOp : public OpKernel {
 public:
  explicit NGraphEnterOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};

class NGraphExitOp : public OpKernel {
 public:
  explicit NGraphExitOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};

class NGraphNextIterationOp : public OpKernel {
 public:
  explicit NGraphNextIterationOp(OpKernelConstruction* context)
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

class NGraphMergeOp : public OpKernel {
 public:
  explicit NGraphMergeOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    bool input_seen = false;
    for (int i = 0; i < context->num_inputs(); ++i) {
      if (context->has_input(i)) {
        if (input_seen) {
          context->SetStatus(errors::Internal(
              "Merge can not have more than one valid input."));
          return;
        }
        input_seen = true;

        if (IsRefType(context->input_dtype(i))) {
          context->forward_ref_input_to_ref_output(i, 0);
        } else {
          context->set_output(0, context->input(i));
        }
        Tensor* value_index = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                         &value_index));
        value_index->scalar<int32>()() = i;
      }
    }
  }
  bool IsExpensive() override { return false; }
  ~NGraphMergeOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphMergeOp);
};

class NGraphSwitchOp : public OpKernel {
 public:
  explicit NGraphSwitchOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) {
    const Tensor& outputPorts = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(outputPorts.shape()),
                errors::InvalidArgument("The second input must be a scalar, "
                                        "but it has shape ",
                                        outputPorts.shape().DebugString()));

    bool pred = outputPorts.scalar<bool>()();
    int port = (pred) ? 1 : 0;
    if (context->input_is_ref(0)) {
      context->forward_ref_input_to_ref_output(0, port);
    } else {
      context->set_output(port, context->input(0));
    }
  }
  bool IsExpensive() override { return false; }
  ~NGraphSwitchOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphSwitchOp);
};

REGISTER_KERNEL_BUILDER(Name("NoOp").Device(ngraph_bridge::DEVICE_NGRAPH),
                        NGraphNoOp);

REGISTER_KERNEL_BUILDER(Name("Const")
                            .Device(ngraph_bridge::DEVICE_NGRAPH)
                            .TypeConstraint("dtype",
                                            ngraph_bridge::NGraphDTypes()),
                        NGraphConstOp);

REGISTER_KERNEL_BUILDER(Name("Identity")
                            .Device(ngraph_bridge::DEVICE_NGRAPH)
                            .TypeConstraint("T", ngraph_bridge::NGraphDTypes()),
                        NGraphIdentityOp);

REGISTER_KERNEL_BUILDER(Name("Enter").Device(ngraph_bridge::DEVICE_NGRAPH),
                        NGraphEnterOp);
REGISTER_KERNEL_BUILDER(Name("Exit").Device(ngraph_bridge::DEVICE_NGRAPH),
                        NGraphExitOp);
REGISTER_KERNEL_BUILDER(
    Name("NextIteration").Device(ngraph_bridge::DEVICE_NGRAPH),
    NGraphNextIterationOp);
REGISTER_KERNEL_BUILDER(Name("Merge").Device(ngraph_bridge::DEVICE_NGRAPH),
                        NGraphMergeOp);
REGISTER_KERNEL_BUILDER(Name("Switch").Device(ngraph_bridge::DEVICE_NGRAPH),
                        NGraphSwitchOp);
