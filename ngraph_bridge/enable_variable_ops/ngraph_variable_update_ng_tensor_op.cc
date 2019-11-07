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
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/default/logging.h"

#include "ngraph/event_tracing.hpp"

#include "ngraph_bridge/enable_variable_ops/ngraph_var.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_variable_update_ng_tensor_op.h"
#include "ngraph_bridge/ngraph_timer.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

//---------------------------------------------------------------------------
//  NGraphVariableUpdateNGTensorOp::ctor
//---------------------------------------------------------------------------
NGraphVariableUpdateNGTensorOp::NGraphVariableUpdateNGTensorOp(
    OpKernelConstruction* context)
    : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("ngraph_graph_id", &ng_graph_id_));
  OP_REQUIRES_OK(context, context->GetAttr("ngraph_variable_shared_name",
                                           &ng_variable_shared_name_));

  NGRAPH_VLOG(4) << "NGraphVariableUpdateNGTensorOp:: Constructor called for: "
                 << def().name() << " ,Graph ID " << ng_graph_id_
                 << " ngraph_variable_shared_name " << ng_variable_shared_name_;

  OP_REQUIRES(context, IsRefType(context->input_type(0)),
              errors::InvalidArgument("input[0] needs to be a ref type"));
}

//---------------------------------------------------------------------------
//  ~NGraphVariableUpdateNGTensorOp
//---------------------------------------------------------------------------
NGraphVariableUpdateNGTensorOp::~NGraphVariableUpdateNGTensorOp() {
  NGRAPH_VLOG(4) << "~NGraphVariableUpdateNGTensorOp::" << name() << endl;
}

//---------------------------------------------------------------------------
// OpKernel::Compute
//---------------------------------------------------------------------------
void NGraphVariableUpdateNGTensorOp::Compute(OpKernelContext* context) {
  std::ostringstream oss;
  // Start event tracing
  ngraph::Event event_compute(oss.str(), name(), "");
  bool log_copies = false;
  OP_REQUIRES_OK(context,
                 IsNgraphTFLogTensorCopiesEnabled(ng_graph_id_, log_copies));
  std::stringstream copy_log_str;
  NGRAPH_VLOG(4) << "KERNEL[" << type_string() << "]: " << name() << "\n";
  int number_of_copies = 0;
  NGRAPH_VLOG(4) << "NGraphVariableUpdateNGTensorOp:: Compute called for: "
                 << def().name() << " ,Graph ID " << ng_graph_id_ << "\n";

  // Since we have ngraph_variable_shared_name as an attribute,
  // we can use that to get the variable from the context
  NGraphVar* var;

  OP_REQUIRES_OK(context, context->resource_manager()->Lookup<NGraphVar>(
                              context->resource_manager()->default_container(),
                              ng_variable_shared_name_, &var));

  // Set the output Ref Tensor at output_index to be an alias of the
  // input Ref Tensor at input_index.
  // REQUIRES: IsRefType(input_dtype(input_index)).
  // REQUIRES: IsRefType(output_dtype(output_index)).
  context->forward_ref_input_to_ref_output(0, 0);

  if (var->copy_tf_to_ng()) {
    number_of_copies++;
    copy_log_str << " COPY_TF_TO_NG ";
  }

  copy_log_str << " Number of copies " << number_of_copies << "\n";
  if (log_copies) {
    cout << copy_log_str.str();
  }

  // The Lookup function used in #84 calls DoLookup which ultimately calls Ref
  // as seen here:
  // https://github.com/tensorflow/tensorflow/blob/5448f25041ed5d32b8aee08250e8ec66e7353593/tensorflow/core/framework/resource_mgr.cc#L187
  // Hence, Unref var here:
  var->Unref();

  // Stop event tracing
  event_compute.Stop();

  ngraph::Event::write_trace(event_compute);

}  // end compute

}  // namespace ngraph_bridge
REGISTER_KERNEL_BUILDER(Name("NGraphVariableUpdateNGTensor").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphVariableUpdateNGTensorOp);
}  // namespace tensorflow
