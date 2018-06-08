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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/platform/default/logging.h"

#include "ngraph_utils.h"

namespace ngraph_bridge {
extern const char* const DEVICE_NGRAPH;
}

using namespace tensorflow;

//
// BEGIN VARIABLE OP STUFF COPIED, WITH MODIFICATION, FROM TF CODE BASE.
//
// Copied from:
//
//   tensorflow/core/kernels/variable_ops.{cc,h}
//   tensorflow/core/kernels/assign_op.h
//   tensorflow/core/kernels/dense_update_ops.cc
//
// The general pattern is that we've taken classes and stuck "NGraph" on the
// front of their names, just so we can get variables placed on NGRAPH. As
// we go, we will be adding nGraph-specific features to these ops.
//
// For the time being, everything here conforms to TF-style naming conventions
// (in particular, member variables are suffixed with "_", not prefixed with
// "m_").
//

// Resource stored by variables in the resource manager
// (legacy, ref-style version).
class NGraphVar : public ResourceBase {
 public:
  explicit NGraphVar(DataType dtype,const TensorShape& shape) : tensor_(dtype,shape) {}
  // Not copyable or movable.
  NGraphVar(const NGraphVar&) = delete;
  NGraphVar& operator=(const NGraphVar&) = delete;

  tf::mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tensor_; }

  string DebugString() override {
    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }

 private:
  tf::mutex mu_;
  Tensor tensor_;

  ~NGraphVar() override {}
};

class NGraphVariableOp : public OpKernel {
 public:
  explicit NGraphVariableOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* ctx) override;

 private:
  DataType dtype_;
  TensorShape shape_;

  tf::mutex init_mu_;
  ContainerInfo cinfo_ GUARDED_BY(init_mu_);
  bool initialized_ GUARDED_BY(init_mu_){false};

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphVariableOp);
};

NGraphVariableOp::NGraphVariableOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
  dtype_ = RemoveRefType(context->output_type(0));
}

void NGraphVariableOp::Compute(OpKernelContext* ctx) {
  mutex_lock l(init_mu_);
  if (!initialized_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                    true /* use name() */));
    initialized_ = true;
  }
  auto creator = [this](NGraphVar** var) {
    *var = new NGraphVar(dtype_,shape_);
    // tf::Variable has friend access to the following method, but we don't.
    // Fortunately we can just tweak NGraphVar to do this in the constructor.
    //(*var)->tensor()->set_shape(shape_);
    return Status::OK();
  };
  NGraphVar* var;
  OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<NGraphVar>(
                          cinfo_.container(), cinfo_.name(), &var, creator));
  // Output a reference to our tensor, so it may be updated.
  //
  // As long as the resource manager hasn't been cleared the ref we return
  // here is valid because it owns a ref on var.
  ctx->set_output_ref(0, var->mu(), var->tensor());
  if (ctx->track_allocations() && var->tensor()->IsInitialized()) {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    ctx->record_persistent_memory_allocation(var->tensor()->AllocatedBytes());
  }
  var->Unref();
}

REGISTER_KERNEL_BUILDER(
   Name("VariableV2").Device(ngraph_bridge::DEVICE_NGRAPH),
   NGraphVariableOp);

class NGraphAssignOp : public OpKernel {
 public:
  explicit NGraphAssignOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &m_use_exclusive_lock));
    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_shape", &m_validate_shape));
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& rhs = context->input(1);

    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    // We can't always know how this value will be used downstream,
    // so make conservative assumptions in specifying constraints on
    // the memory allocation attributes.
    // TODO(rmlarsen): These conservative constraints make buffer
    // forwarding unlikely to happen very often. Try to use graph analysis
    // (possibly the InferAllocAttr pass in the executer) to improve the
    // situation.
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);

    {
      mutex_lock l(*context->input_ref_mutex(0));
      const Tensor& old_lhs = context->mutable_input(0, /* lock_held */ true);
      const bool same_shape = old_lhs.shape().IsSameSize(rhs.shape());
      if (m_validate_shape) {
        OP_REQUIRES(
            context, same_shape,
            errors::InvalidArgument(
                "Assign requires shapes of both tensors to match. lhs shape= ",
                old_lhs.shape().DebugString(),
                " rhs shape= ", rhs.shape().DebugString()));
      }

      // In the code below we try to minimize the amount of memory allocation
      // and copying by trying the following two shortcuts:
      // 1. If we can reuse the rhs buffer we avoid both a memory allocation
      //   and copying.
      // 2. If the lhs is initialized and has the same number of elements as the
      //    rhs we can avoid a memory allocation.

      // 1. Try to reuse the rhs.
      std::unique_ptr<Tensor> input_alias = context->forward_input(
          1, OpKernelContext::Params::kNoReservation /*output_index*/,
          old_lhs.dtype(), old_lhs.shape(), DEVICE_MEMORY, attr);
      if (input_alias != nullptr) {
        // Transfer ownership to the ref.
        context->replace_ref_input(0, *input_alias.release(),
                                   /* lock_held */ true);
        return;
      }

      // 2. Try to copy into an existing buffer.
      if (old_lhs.IsInitialized() &&
          old_lhs.shape().num_elements() == rhs.shape().num_elements()) {
        // The existing lhs tensor has already been initialized and the right
        // hand side can fit in the underlying buffer.
        Tensor reshaped_old_lhs;
        if (same_shape) {
          reshaped_old_lhs = old_lhs;
        } else {
          CHECK(reshaped_old_lhs.CopyFrom(old_lhs, rhs.shape()));
          context->replace_ref_input(0, reshaped_old_lhs, /* lock_held */ true);
        }
        if (m_use_exclusive_lock) {
          Copy(context, &reshaped_old_lhs, rhs);
          return;
        }
      } else {
        // Create a new persistent tensor whose shape matches the right hand
        // side, hand off to lhs and copy the rhs into it.
        PersistentTensor copy;
        Tensor* copyTensor = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_persistent(old_lhs.dtype(), rhs.shape(),
                                                  &copy, &copyTensor, attr));
        // We track memory of variables in variable ops instead of in this
        // assign op.
        context->clear_recorded_memory();
        context->replace_ref_input(0, *copyTensor, /* lock_held */ true);
        if (m_use_exclusive_lock) {
          Copy(context, copyTensor, rhs);
          return;
        }
      }
    }

    // The tensor has already been initialized and the right hand side
    // matches the left hand side's shape. We have been told to do the
    // copy outside the lock.
    Tensor old_unlocked_lhs = context->mutable_input(0, /* lock_held */ false);
    Copy(context, &old_unlocked_lhs, rhs);
  }

  void Copy(OpKernelContext* context, Tensor* lhs, const Tensor& rhs) {
    void* src_ptr = const_cast<void*>(DMAHelper::base(&rhs));
    const int64 total_bytes = rhs.TotalBytes();
    void* dst_ptr = DMAHelper::base(lhs);
    memcpy(dst_ptr, src_ptr, total_bytes);
  }

  bool m_use_exclusive_lock;
  bool m_validate_shape;
};

REGISTER_KERNEL_BUILDER(
   Name("Assign").Device(ngraph_bridge::DEVICE_NGRAPH),
   NGraphAssignOp);

class NGraphIsVariableInitializedOp : public OpKernel {
 public:
  explicit NGraphIsVariableInitializedOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get a mutable input tensor of the Ref input.
    const Tensor& input_tensor = context->mutable_input(0, false);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<bool, 0>();
    bool result = input_tensor.IsInitialized();
    output_tensor() = result;
  }
};

REGISTER_KERNEL_BUILDER(
   Name("IsVariableInitialized").Device(ngraph_bridge::DEVICE_NGRAPH),
   NGraphIsVariableInitializedOp);
//
// END VARIABLE OP STUFF COPIED, WITH MODIFICATION, FROM TF CODE BASE
//
