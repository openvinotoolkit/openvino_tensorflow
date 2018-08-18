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

#include "ngraph_freshness_tracker.h"
#include "ngraph_utils.h"

namespace ngraph_bridge {
extern const char* const DEVICE_NGRAPH;
}

using namespace tensorflow;
namespace ngb = ngraph_bridge;

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
  explicit NGraphVar(DataType dtype, const TensorShape& shape)
      : tensor_(dtype, shape) {}
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
  ~NGraphVariableOp();

 private:
  DataType dtype_;
  TensorShape shape_;

  tf::mutex init_mu_;
  ContainerInfo cinfo_ GUARDED_BY(init_mu_);
  NGraphVar* var_ = nullptr;
  bool initialized_ GUARDED_BY(init_mu_){false};

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphVariableOp);
};

NGraphVariableOp::NGraphVariableOp(OpKernelConstruction* context)
    : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
  dtype_ = RemoveRefType(context->output_type(0));
  var_ = nullptr;
}

void NGraphVariableOp::Compute(OpKernelContext* ctx) {
  mutex_lock l(init_mu_);
  if (!initialized_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                    true /* use name() */));
    initialized_ = true;
  }
  auto creator = [this](NGraphVar** var) {
    // This part is modified slightly. The TF implementation does this:
    //
    //   *var = new LocalVar(dtype_);
    //   (*var)->tensor()->set_shape(shape_);
    //
    // but VariableOp is a friend of tensor and set_shape is private. I believe
    // what this is doing is setting the shape of the tensor without allocating
    // any memory for it, since Assign will do that later if necessary. For us,
    // pre-allocation is okay for now.
    *var = new NGraphVar(dtype_, shape_);
    return Status::OK();
  };
  // NGraphVar* var;
  if (var_ == nullptr) {
    OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<NGraphVar>(
                            cinfo_.container(), cinfo_.name(), &var_, creator));
  }
  // Output a reference to our tensor, so it may be updated.
  //
  // As long as the resource manager hasn't been cleared the ref we return
  // here is valid because it owns a ref on var.
  ctx->set_output_ref(0, var_->mu(), var_->tensor());
  if (ctx->track_allocations() && var_->tensor()->IsInitialized()) {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    ctx->record_persistent_memory_allocation(var_->tensor()->AllocatedBytes());
  }
  // var->Unref();
}

NGraphVariableOp::~NGraphVariableOp() { var_->Unref(); }

// REGISTER_KERNEL_BUILDER(Name("VariableV2").Device(ngraph_bridge::DEVICE_NGRAPH),
//                        NGraphVariableOp);

class NGraphAssignOp : public OpKernel {
 public:
  explicit NGraphAssignOp(OpKernelConstruction* context)
      : OpKernel(context), m_tracker(nullptr) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &m_use_exclusive_lock));
    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_shape", &m_validate_shape));
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
  }

  ~NGraphAssignOp() { m_tracker->Unref(); }

  void Compute(OpKernelContext* context) override {
    const Tensor& rhs = context->input(1);

    // Mark the underlying tensor as stale.
    auto creator = [this](ngb::NGraphFreshnessTracker** tracker) {
      *tracker = new ngb::NGraphFreshnessTracker();
      return Status::OK();
    };
    if (m_tracker == nullptr) {
      OP_REQUIRES_OK(context,
                     context->resource_manager()
                         ->LookupOrCreate<ngb::NGraphFreshnessTracker>(
                             context->resource_manager()->default_container(),
                             "ngraph_freshness_tracker", &m_tracker, creator));
      m_tracker->AddTensor(DMAHelper::base(&context->input(0)));
      m_tracker->MarkStale(DMAHelper::base(&context->input(0)));
    }

    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    {
      mutex_lock l(*context->input_ref_mutex(0));
      const Tensor& old_lhs = context->mutable_input(0, /* lock_held */ true);
      const bool same_shape = old_lhs.shape().IsSameSize(rhs.shape());

      // For now, nGraph bridge only works if the rhs shape matches. This means
      // that the tensor we are assigning to must previously have been
      // allocated with a matching shape, and we cannot change its shape here
      // at assignment time.
      //
      // It should be possible to remove this restriction, but it may require
      // some tricky interactions with the freshness tracker, since we will
      // need to allocate new space. Since we don't yet have any use cases that
      // will exercise this functionality, it's best not try to implement it
      // yet.
      OP_REQUIRES(
          context, same_shape,
          errors::InvalidArgument("Assign for nGraph requires shapes of both "
                                  "tensors to match. lhs shape= ",
                                  old_lhs.shape().DebugString(), " rhs shape= ",
                                  rhs.shape().DebugString()));

      if (m_use_exclusive_lock) {
        Tensor lhs = context->mutable_input(0, /* lock_held */ true);
        Copy(context, &lhs, rhs);
        return;
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
  ngb::NGraphFreshnessTracker* m_tracker;
};

// REGISTER_KERNEL_BUILDER(Name("Assign").Device(ngraph_bridge::DEVICE_NGRAPH),
//                        NGraphAssignOp);

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

// REGISTER_KERNEL_BUILDER(
//    Name("IsVariableInitialized").Device(ngraph_bridge::DEVICE_NGRAPH),
//    NGraphIsVariableInitializedOp);
//
// END VARIABLE OP STUFF COPIED, WITH MODIFICATION, FROM TF CODE BASE
//
