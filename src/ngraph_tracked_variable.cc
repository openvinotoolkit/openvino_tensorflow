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

#include "ngraph/event_tracing.hpp"
#include "ngraph_freshness_tracker.h"
#include "ngraph_utils.h"

namespace tensorflow {

namespace ngraph_bridge {

//
// Forked from tensorflow:tensorflow/core/kernels/variable_ops.{cc,h}
// and tensorflow:tensorflow/core/ops/state_ops.cc.
//

// Resource stored by variables in the resource manager
// (legacy, ref-style version).
//
// (Changes: Renamed from LegacyVar, modified to take a TensorShape in
// constructor.)

// THIS CLASS IS NOT BEING USED ANYWHERE
class NGraphVar : public ResourceBase {
 public:
  explicit NGraphVar(DataType dtype, TensorShape shape)
      : tensor_(dtype, shape) {}
  // Not copyable or movable.
  NGraphVar(const NGraphVar&) = delete;
  NGraphVar& operator=(const NGraphVar&) = delete;

  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tensor_; }

  string DebugString() override {
    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }

 private:
  mutex mu_;
  Tensor tensor_;

  ~NGraphVar() override {}
};

class NGraphVariableOp : public OpKernel {
 public:
  explicit NGraphVariableOp(OpKernelConstruction* context);
  ~NGraphVariableOp() override;
  void Compute(OpKernelContext* ctx) override;

 private:
  TensorShape shape_;
  bool just_looking_;
  NGraphFreshnessTracker* tracker_;
  DataType dtype_;

  mutex init_mu_;
  ContainerInfo cinfo_ GUARDED_BY(init_mu_);
  bool initialized_ GUARDED_BY(init_mu_){false};

  static int s_instance_count;
  int my_instance_id{0};

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphVariableOp);
};

int NGraphVariableOp::s_instance_count = 0;

NGraphVariableOp::NGraphVariableOp(OpKernelConstruction* context)
    : OpKernel(context),
      just_looking_(false),
      tracker_(nullptr),
      dtype_(RemoveRefType(context->output_type(0))) {
  my_instance_id = s_instance_count;
  s_instance_count++;

  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
  OP_REQUIRES_OK(context, context->GetAttr("just_looking", &just_looking_));
  NGRAPH_VLOG(5) << def().name() << ": just looking? " << just_looking_;
}

NGraphVariableOp::~NGraphVariableOp() { tracker_->Unref(); }

// (Changes: Renamed from VariableOp, modified to pass TensorShape to NGraphVar
// constructor.)
void NGraphVariableOp::Compute(OpKernelContext* ctx) {
  mutex_lock l(init_mu_);
  std::ostringstream oss;
  oss << "NGraphVariable: " << my_instance_id << ": " << name();
  ngraph::Event event_compute(oss.str(), name(), "");

  if (!initialized_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                    true /* use name() */));
    initialized_ = true;
  }
  auto creator = [this](NGraphVar** var) {
    *var = new NGraphVar(dtype_, shape_);
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

  // Mark the underlying tensor as stale. TODO(amprocte): Make this
  // conditional on whether any reader is taking in a reference. More
  // conservative condition that would work for now: invalidate if any
  // reader is not NGraphEncapsulateOp.
  auto t_creator = [this](NGraphFreshnessTracker** tracker) {
    *tracker = new NGraphFreshnessTracker();
    return Status::OK();
  };
  if (tracker_ == nullptr) {
    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name()
                     << ": getting tracker";
    }
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->LookupOrCreate<NGraphFreshnessTracker>(
                 ctx->resource_manager()->default_container(),
                 "ngraph_freshness_tracker", &tracker_, t_creator));
    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name()
                     << ": got tracker";
    }
  }

  if (NGRAPH_VLOG_IS_ON(5)) {
    NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name() << ": adding "
                   << DMAHelper::base(var->tensor());
  }
  tracker_->AddTensor(DMAHelper::base(var->tensor()));
  if (NGRAPH_VLOG_IS_ON(5)) {
    NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name() << ": added "
                   << DMAHelper::base(var->tensor());
  }

  if (!just_looking_) {
    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name() << ": marking "
                     << DMAHelper::base(var->tensor());
    }
    tracker_->MarkStale(DMAHelper::base(var->tensor()));
    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name() << ": marked "
                     << DMAHelper::base(var->tensor());
    }
  }

  ctx->set_output_ref(0, var->mu(), var->tensor());
  if (ctx->track_allocations() && var->tensor()->IsInitialized()) {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    ctx->record_persistent_memory_allocation(var->tensor()->AllocatedBytes());
  }
  var->Unref();
  ngraph::Event::write_trace(event_compute);
}

REGISTER_OP("NGraphVariable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("just_looking: bool = false")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_KERNEL_BUILDER(Name("NGraphVariable").Device(DEVICE_CPU),
                        NGraphVariableOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
