/*******************************************************************************
 * Copyright 2019 Intel Corporation
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
#ifndef NGRAPH_TF_NGRAPHVAR_H_
#define NGRAPH_TF_NGRAPHVAR_H_

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/default/logging.h"

#include "ngraph/event_tracing.hpp"
#include "ngraph/runtime/backend.hpp"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// THIS CLASS IS NOT BEING USED ANYWHERE
class NGraphVar : public ResourceBase {
 public:
  explicit NGraphVar(DataType dtype, TensorShape shape, string BackendName)
      : tf_tensor_(dtype, shape),
        ng_backend_name_(BackendName),
        sync_ng_tensor_(false) {
    // TF datatype to nGraph element type
    ng::element::Type ng_element_type;
    TFDataTypeToNGraphElementType(dtype, &ng_element_type);

    // TF TensorShape to nGraphShape
    ng::Shape ng_shape(shape.dims());
    for (int j = 0; j < shape.dims(); ++j) {
      ng_shape[j] = shape.dim_size(j);
    }

    // Create Backend
    NGRAPH_VLOG(4) << "NGraphVar::Create Backend ";
    Status status;
    ng::runtime::Backend* op_backend;
    status = BackendManager::CreateBackend(ng_backend_name_);
    if (!status.ok()) {
      throw std::runtime_error("Cannot create backend " + ng_backend_name_ +
                               ". Got Exception " + status.error_message());
    }

    try {
      op_backend = BackendManager::GetBackend(ng_backend_name_);
    } catch (...) {
      throw std::runtime_error("No backend available :" + ng_backend_name_ +
                               ". Cannot execute graph");
    }

    // Create nGTensor

    // Check buffer sharing
    // -1 implies env var is not set
    int buffer_sharing_state_env = -1;
    // 0 implies buffer sharing is disabled
    // 1 implies buffer sharing is enabled
    status = GetNgraphVarBufferSharingState(buffer_sharing_state_env);
    if (!status.ok()) {
      throw std::runtime_error({"Got Exception " + status.error_message()});
    }

    ng_tf_share_buffer_ = (buffer_sharing_state_env == -1)
                              ? (ng_backend_name_ == "CPU")
                              : buffer_sharing_state_env;

    if (ng_tf_share_buffer_) {
      void* tf_src_ptr = (void*)DMAHelper::base(&tf_tensor_);
      ng_tensor_ =
          op_backend->create_tensor(ng_element_type, ng_shape, tf_src_ptr);
    } else {
      ng_tensor_ = op_backend->create_tensor(ng_element_type, ng_shape);
    }
  }
  // Not copyable or movable.
  NGraphVar(const NGraphVar&) = delete;
  NGraphVar& operator=(const NGraphVar&) = delete;

  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tf_tensor_; }
  shared_ptr<ngraph::runtime::Tensor> ng_tensor() { return ng_tensor_; };

  string DebugString() const override {
    return strings::StrCat(DataTypeString(tf_tensor_.dtype()), "/",
                           tf_tensor_.shape().DebugString());
  }

  bool need_sync_ng_tensor() { return sync_ng_tensor_; }
  void set_sync_ng_tensor(bool sync_ng_tensor) {
    sync_ng_tensor_ = sync_ng_tensor;
  }

  // Copies the NG Tensor to TF Tensor for this variable
  // Involves a copy from device to host
  // Returns the number of tensor copies made (0 or 1)
  int copy_ng_to_tf() {
    if (ng_tf_share_buffer_) {
      return 0;
    }
    ReadNGTensor(ng_tensor_, &tf_tensor_);
    return 1;
  }

  // Copies the TF Tensor to NG Tensor for this variable
  // Involves a copy from host to device
  // Returns the number of tensor copies made (0 or 1)
  int copy_tf_to_ng() {
    if (ng_tf_share_buffer_) {
      return 0;
    }
    WriteNGTensor(ng_tensor_, &tf_tensor_);
    return 1;
  }

  // If the NGTensor is behind TF Tensor (ie if NGTensor is out-of-date)
  // It updates ng_tensor by copy_tf_to_ng
  // Returns the number of tensor copies made (0 or 1)
  int sync_ng_tensor() {
    if (sync_ng_tensor_) {
      return copy_tf_to_ng();
    }
    return 0;
  }

  // updates the NGTensor with the new value
  // This new_value could be from ngraph-tensor, for e.g. when computed from
  // NGraphEncapsulateOp
  // and saved in Catalog
  // Returns the number of tensor copies made (0 or 1)
  int update_ng_tensor(shared_ptr<ngraph::runtime::Tensor> new_value) {
    ng_tensor_->copy_from(*new_value);
    return 0;
  }

  // updates the NGTensor with the new value
  // This new_value could be from tf-tensor, for e.g. when computed from a TF op
  // Returns the number of tensor copies made (0 or 1)
  int update_ng_tensor(Tensor* new_value) {
    WriteNGTensor(ng_tensor_, new_value);
    if (ng_tf_share_buffer_) {
      return 0;
    }

    return 1;
  }

 private:
  mutex mu_;
  Tensor tf_tensor_;
  shared_ptr<ngraph::runtime::Tensor> ng_tensor_;
  string ng_backend_name_;
  bool ng_tf_share_buffer_;
  // sync from tf to ng
  bool sync_ng_tensor_;
  ~NGraphVar() override {
    // Release the backend
    NGRAPH_VLOG(2) << "~NGraphVar::ReleaseBackend";
    BackendManager::ReleaseBackend(ng_backend_name_);
  }
};

}  // namespace ng-bridge
}  // namespace tf

#endif
