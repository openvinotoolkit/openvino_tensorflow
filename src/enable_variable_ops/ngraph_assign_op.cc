/*******************************************************************************
 * Copyright 2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use thi0s file except in compliance with the License.
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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/default/logging.h"

#include "ngraph/event_tracing.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph_catalog.h"
#include "ngraph_freshness_tracker.h"
#include "ngraph_timer.h"
#include "ngraph_utils.h"
#include "ngraph_var.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

/* -------------------------------------------------
//
// NGraphAssignOp
//
---------------------------------------------------*/

// Computes *input[0] = input[1]
class NGraphAssignOp : public OpKernel {
 private:
  bool just_looking_;
  bool copy_to_tf_;
  int ng_graph_id_;
  static int s_instance_count;
  int my_instance_id{0};

  // TODO(malikshr): Do we need these attributes, exist in TF Assign ops
  // use_exclusive_lock_, validate_shape_, relax_constraints_;

 public:
  ~NGraphAssignOp() { NGRAPH_VLOG(4) << "~NGraphAssignOp::" << name() << endl; }
  explicit NGraphAssignOp(OpKernelConstruction* context)
      : OpKernel(context), just_looking_(false), copy_to_tf_(false) {
    OP_REQUIRES_OK(context, context->GetAttr("just_looking", &just_looking_));
    OP_REQUIRES_OK(context, context->GetAttr("copy_to_tf", &copy_to_tf_));
    OP_REQUIRES_OK(context, context->GetAttr("ngraph_graph_id", &ng_graph_id_));

    NGRAPH_VLOG(4) << "NGraphAssign:: Constructor called for: " << def().name()
                   << ",just looking " << PrintBool(just_looking_)
                   << ",copy-to-tf " << PrintBool(copy_to_tf_) << " ,Graph ID "
                   << ng_graph_id_;

    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
    my_instance_id = s_instance_count;
    s_instance_count++;
  }

  void Compute(OpKernelContext* context) override {
    std::ostringstream oss;
    oss << "Execute: Assign_" << my_instance_id << ": " << name();
    ngraph::Event event_compute(oss.str(), name(), "");

    NGRAPH_VLOG(4) << "NGraphAssign:: Compute called for: " << def().name()
                   << " ,just looking " << PrintBool(just_looking_)
                   << " ,copy-to-tf " << PrintBool(copy_to_tf_) << " ,Graph ID "
                   << ng_graph_id_;

    bool log_copies = false;
    OP_REQUIRES_OK(context, IsCopyLogEnabled(ng_graph_id_, log_copies));
    std::stringstream copy_log_str;
    copy_log_str << "KERNEL[" << type_string() << "]: " << name()
                 << " ,Copy_TF " << PrintBool(copy_to_tf_) << " ,Just_Looking "
                 << PrintBool(just_looking_) << "\n";
    int number_of_copies = 0;

    bool ref_exists = NGraphCatalog::ExistsInInputVariableSharedNameMap(
        ng_graph_id_, def().name(), 0);
    if (!ref_exists) {
      OP_REQUIRES(context, ref_exists,
                  errors::Internal(
                      "Caught exception : RefInput to NGAssign not found \n"));
    }
    string get_ref_var_name = NGraphCatalog::GetInputVariableSharedName(
        ng_graph_id_, def().name(), 0);

    NGraphVar* var;
    OP_REQUIRES_OK(context,
                   context->resource_manager()->Lookup<NGraphVar>(
                       context->resource_manager()->default_container(),
                       get_ref_var_name, &var));

    const Tensor& rhs = context->input(1);

    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    // get the nGraphTensor
    shared_ptr<ngraph::runtime::Tensor> ng_tensor_to_assign = var->ng_tensor();

    // DO NOT CARE ABOUT SYNCING AS WE ARE ALWAYS SETTING THE NGTENSOR

    // Get input[1]
    string valkey = to_string(ng_graph_id_) + "_" + def().input(1);
    bool valref_exists = NGraphCatalog::ExistsInEncapOutputTensorMap(valkey);
    if (valref_exists) {
      // Value is from encap
      NGRAPH_VLOG(4) << "NGraphAssign::Getting from catalog: " << valkey;
      auto ng_val = NGraphCatalog::GetTensorFromEncapOutputTensorMap(valkey);
      ng_tensor_to_assign->copy_from(*ng_val);
    } else {
      number_of_copies++;
      copy_log_str << " COPY_INP_VAL[0]";
      NGRAPH_VLOG(4) << "NGraphAssign::Getting from TF : " << valkey;
      void* tf_src_ptr = (void*)DMAHelper::base(&rhs);
      ng_tensor_to_assign->write(
          tf_src_ptr, 0, ng_tensor_to_assign->get_element_count() *
                             ng_tensor_to_assign->get_element_type().size());
    }

    mutex_lock l(*context->input_ref_mutex(0));
    Tensor old_lhs = context->mutable_input(0, /* lock_held */ true);

    if (copy_to_tf_) {
      number_of_copies++;
      copy_log_str << " COPY_TF ";
      ReadNGTensor(ng_tensor_to_assign, &old_lhs);

      if (!just_looking_) {
        // Some tf op might update the ng-tensor value so mark it stale
        copy_log_str << " SET_SYNC ";
        var->sync_ng_tensor(true);
      }
    }

    copy_log_str << " Number of copies " << number_of_copies << "\n";
    if (log_copies) {
      cout << copy_log_str.str();
    }

    // Unref Var
    var->Unref();
    event_compute.Stop();
    ngraph::Event::write_trace(event_compute);
  }
};

int NGraphAssignOp::s_instance_count = 0;

REGISTER_OP("NGraphAssign")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .Attr("just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

REGISTER_KERNEL_BUILDER(Name("NGraphAssign").Device(DEVICE_CPU),
                        NGraphAssignOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
