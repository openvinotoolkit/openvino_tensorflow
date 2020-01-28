/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include "ngraph_bridge/ngraph_partial_shapes.h"

namespace tensorflow {

namespace ngraph_bridge {

PartialShape::PartialShape(std::vector<int> shape)
    : m_shape(shape), m_valid(true) {
  for (auto i : m_shape) {
    if (i < -1) {
      invalidate();
    }
  }
}
PartialShape::PartialShape() : m_valid(false) {}

PartialShape::PartialShape(
    const tensorflow::TensorShapeProto& tensor_shape_proto) {
  try {
    m_shape.resize(tensor_shape_proto.dim_size());
    for (uint shape_idx = 0; shape_idx < tensor_shape_proto.dim_size();
         shape_idx++) {
      auto num_elems_in_this_dim = tensor_shape_proto.dim(shape_idx).size();
      m_shape[shape_idx] = num_elems_in_this_dim;
      // -1 means not specified
    }
    m_valid = true;
  } catch (...) {
    invalidate();
  }
}

bool PartialShape::is_concrete() const {
  check_valid();
  return std::all_of(m_shape.begin(), m_shape.end(),
                     [](int i) { return i >= 0; });
}

int PartialShape::size() const {
  check_valid();
  return m_shape.size();
}

int PartialShape::operator[](int idx) const {
  check_valid();
  return m_shape[idx];
}

std::vector<int> PartialShape::get_shape_vector() const {
  check_valid();
  return m_shape;
}

string PartialShape::to_string() const {
  std::string st = m_valid ? "valid:" : "invalid:";
  for (auto i : m_shape) {
    st += (std::to_string(i) + ",");
  }
  return st;
}

void PartialShape::concretize(const PartialShape& shape_hint) {
  // Both PartialShapes are expected to be valid
  check_valid();
  uint base_rank = m_shape.size();
  if (base_rank != shape_hint.size()) {  // different ranks
    invalidate();
    return;
  } else {
    for (int i = 0; i < base_rank; i++) {
      if (m_shape[i] != shape_hint[i]) {
        if (m_shape[i] == -1) {
          m_shape[i] = shape_hint[i];
        } else {
          if (shape_hint[i] != -1) {
            invalidate();
            return;
          }
        }
      }
    }
    return;
  }
}

void PartialShape::check_valid() const {
  if (!m_valid) {
    throw std::runtime_error(
        string("Attempted to use an invalid PartialShape"));
  }
}

void PartialShape::invalidate() {
  m_shape.clear();
  m_valid = false;
}
}
}