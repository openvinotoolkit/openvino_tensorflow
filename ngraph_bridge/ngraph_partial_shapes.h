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

#ifndef NGRAPH_TF_BRIDGE_PARTIAL_SHAPES_H_
#define NGRAPH_TF_BRIDGE_PARTIAL_SHAPES_H_
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace ngraph_bridge {

// TODO unit test the class and move to a different file
class PartialShape {
  // This is a simple class that can represent full or partial shapes
  // a full shape has all dimensions >= 0. A partial shape has atleast one
  // dimension < 0
  // Its built like an optional/maybe, so please use is_valid before accessing
  // other functions

  // A sample lattice that this class operates on is shown for rank = 2 below
  //             Invalid
  //           /         \
  //       (1,2)          (2,2)
  //        /  \           /  \
  //    (-1,2)(1,-1)    (2,-1)(-1,2)
  //         \     \       /     /
  //           ---- (-1,-1) ----
  //
  // The PartialShape allows only 2 functionsthat modify object state
  // Copy assignment (=) and concretize()
  // Subsequent calls to concretize() can only make one move up the lattice.
  // Copy assignment can reset the state of teh object arbitrarily

  // The class is not thread safe

  // Any scalar is represented by {True, {}} by this class

 public:
  PartialShape(std::vector<int> shape);
  PartialShape();

  PartialShape(const tensorflow::TensorShapeProto& tensor_shape_proto);

  bool is_concrete() const;

  int size() const;

  int operator[](int idx) const;

  std::vector<int> get_shape_vector() const;

  bool is_valid() const { return m_valid; }

  string to_string() const;

  void concretize(const PartialShape& shape_hint);

 private:
  std::vector<int> m_shape;
  bool m_valid;

  void check_valid() const;

  void invalidate();
};
}
}
#endif  // NGRAPH_TF_BRIDGE_PARTIAL_SHAPES_H_