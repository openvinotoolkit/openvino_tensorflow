/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#pragma once

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"

namespace tensorflow {
namespace openvino_tensorflow {

class IETensor : public ngraph::runtime::Tensor {
 public:
  IETensor(const ngraph::element::Type& element_type,
           const ngraph::Shape& shape);
  IETensor(const ngraph::element::Type& element_type,
           const ngraph::PartialShape& shape);
  IETensor(const ngraph::element::Type& element_type,
           const ngraph::Shape& shape, void* memory_pointer);
  IETensor(InferenceEngine::Blob::Ptr blob);
  ~IETensor() override;

  void write(const void* src, size_t bytes) override;
  void read(void* dst, size_t bytes) const override;

  const void* get_data_ptr() const;
  InferenceEngine::Blob::Ptr get_blob() { return m_blob; }

 private:
  IETensor(const IETensor&) = delete;
  IETensor(IETensor&&) = delete;
  IETensor& operator=(const IETensor&) = delete;
  InferenceEngine::Blob::Ptr m_blob;
};

}  // namespace openvino_tensorflow
}  // namespace tensorflow
