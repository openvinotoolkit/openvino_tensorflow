/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/default/logging.h"

#include "ngraph/event_tracing.hpp"
#include "ngraph/runtime/backend.hpp"

#include "ngraph_bridge/enable_variable_ops/ngraph_catalog.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_var.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

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

/* -------------------------------------------------
//
// NGraphVariableOp
//
---------------------------------------------------*/
class NGraphVariableOp : public OpKernel {
 public:
  explicit NGraphVariableOp(OpKernelConstruction* context);
  ~NGraphVariableOp() override;
  void Compute(OpKernelContext* ctx) override;

 private:
  int ng_graph_id_;
  TensorShape shape_;
  NGraphFreshnessTracker* tracker_;
  // If the variable is not going to be modified
  // then just_looking = true
  // This is added to keep it consistent with the path where
  // enable_variable_and_optimizers flag is not set and to add the
  // freshness tracking required for the CPU backend.
  bool just_looking_;
  // If TF is not going to modify the variable
  // then is_tf_just_looking = true
  bool is_tf_just_looking_;
  bool copy_to_tf_;
  DataType dtype_;
  string ng_backend_name_;
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
      tracker_(nullptr),
      just_looking_(false),
      is_tf_just_looking_(false),
      copy_to_tf_(false),
      dtype_(RemoveRefType(context->output_type(0))) {
  my_instance_id = s_instance_count;
  s_instance_count++;

  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
  OP_REQUIRES_OK(context, context->GetAttr("just_looking", &just_looking_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("is_tf_just_looking", &is_tf_just_looking_));
  OP_REQUIRES_OK(context, context->GetAttr("copy_to_tf", &copy_to_tf_));
  OP_REQUIRES_OK(context, context->GetAttr("ngraph_graph_id", &ng_graph_id_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("_ngraph_backend", &ng_backend_name_));
  NGRAPH_VLOG(4) << "NGraphVariable:: Constructor called for: " << def().name()
                 << " ,just looking " << just_looking_ << "is_tf_just_looking "
                 << is_tf_just_looking_ << " ,copy-to-tf " << copy_to_tf_
                 << " ,Graph ID " << ng_graph_id_ << " ,backend_name "
                 << ng_backend_name_;
}

NGraphVariableOp::~NGraphVariableOp() {
  NGRAPH_VLOG(4) << "~NGraphVariableOp:: " << name() << endl;
  string node_key = NGraphCatalog::CreateNodeKey(ng_graph_id_, name(), 0);
  NGraphCatalog::DeleteFromInputVariableSharedNameMap(node_key);
  tracker_->Unref();
}

// (Changes: Renamed from VariableOp, modified to pass TensorShape to NGraphVar
// constructor.)
void NGraphVariableOp::Compute(OpKernelContext* ctx) {
  NGRAPH_VLOG(4) << "NGraphVariable:: Compute called for: " << def().name()
                 << " ,just looking " << just_looking_ << "is_tf_just_looking "
                 << is_tf_just_looking_ << " ,copy-to-tf " << copy_to_tf_
                 << " ,Graph ID " << ng_graph_id_ << " ,backend_name "
                 << ng_backend_name_;

  std::ostringstream oss;
  oss << "NGraphVariable: " << my_instance_id << ": " << name();
  ngraph::Event event_compute(oss.str(), name(), "");

  bool log_copies = false;
  OP_REQUIRES_OK(ctx,
                 IsNgraphTFLogTensorCopiesEnabled(ng_graph_id_, log_copies));
  std::stringstream copy_log_str;
  copy_log_str << "KERNEL[" << type_string() << "]: " << name()
               << " ,copy-to-tf " << PrintBool(copy_to_tf_)
               << " ,is_tf_just_looking " << PrintBool(is_tf_just_looking_)
               << "\n";
  int number_of_copies = 0;

  mutex_lock l(init_mu_);
  if (!initialized_) {
    // Analyze the node attribute of 'ndef' and decides the container and
    // resource name the kernel should use for accessing the shared
    // resource.
    //
    // 'ndef' is expected to have node attribute "container" and
    // "shared_name". Returns non-OK if they are not provided or they are
    // invalid.
    //
    // The policy is as following:
    // * If the attribute "container" is non-empty, it is used as is.
    //   Otherwise, uses the resource manager's default container.
    // * If the attribute "shared_name" is non-empty, it is used as is.
    //   Otherwise, if "use_node_name_as_default" is true, the kernel's
    //   node name is used as the resource name. Otherwise, a string
    //   unique to this process is used.

    // API: Status Init(ResourceMgr* rmgr, const NodeDef& ndef,
    //          bool use_node_name_as_default);
    //
    //
    // We Use context's resource manager's default container
    // And shared name is same as node_name
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                    true /* use name() */));
    initialized_ = true;
  }

  auto creator = [this](NGraphVar** var) {
    NGRAPH_VLOG(4) << "Create NGraphVar Tensors for " << name() << endl;
    *var = new NGraphVar(dtype_, shape_, ng_backend_name_);
    return Status::OK();
  };

  // If "container" has a resource "name", returns it in
  // "*resource". Otherwise, invokes creator() to create the resource.
  // The caller takes the ownership of one ref on "*resource".
  //

  // Here uses the Resource Manager's default container
  NGraphVar* var;
  OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<NGraphVar>(
                          cinfo_.container(), cinfo_.name(), &var, creator));

  bool just_synced = false;
  // If NG-Tensor was modified by TF sync it
  if (var->sync_ng_tensor()) {
    number_of_copies++;
    copy_log_str << "Var_Sync ";
    NGRAPH_VLOG(4) << "in tracked variable, ng tensor behind, needs to sync "
                      "with tf-tensor";
    var->set_sync_ng_tensor(false);
    just_synced = true;
  }

  // If TF needs the tensor
  if (copy_to_tf_) {
    // If it was just synced from TF
    // We dont need to sync again
    if (!just_synced) {
      if (var->copy_ng_to_tf()) {
        number_of_copies++;
        copy_log_str << " COPY_TO_TF ";
      }
      NGRAPH_VLOG(4) << "Copying to TF Tensor";
    }

    if (!is_tf_just_looking_) {
      // Some tf op might update the tf-tensor
      // So we need to sync_it_later
      var->set_sync_ng_tensor(true);
      copy_log_str << " SET_SYNC ";
    }
  }

  copy_log_str << " Number of copies " << number_of_copies << "\n";
  if (log_copies) {
    cout << copy_log_str.str();
  }

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
  // To output a reference.  Caller retains ownership of mu and tensor_for_ref,
  // and they must outlive all uses within the step. See comment above.
  // REQUIRES: IsRefType(expected_output_dtype(index))
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

REGISTER_KERNEL_BUILDER(Name("NGraphVariable").Device(DEVICE_CPU),
                        NGraphVariableOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
