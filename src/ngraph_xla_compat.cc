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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
#include "ngraph/builder/xla_tuple.hpp"
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>

#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/type.hpp"
#include "ngraph_xla_compat.h"

using namespace std;

namespace xla {

namespace ngraph_plugin {

namespace compat {

op::Tuple::Tuple(const ngraph::NodeVector& nodes)
    : ngraph::Node("Tuple", ngraph::NodeVector{}), m_elements(nodes) {}

std::shared_ptr<ngraph::Node> op::Tuple::copy_with_new_args(
    const ngraph::NodeVector& new_args) const {
  return make_shared<op::Tuple>(new_args);
}

const ngraph::NodeVector& op::Tuple::get_elements() const { return m_elements; }

size_t op::Tuple::get_tuple_size() const { return m_elements.size(); }

shared_ptr<ngraph::Node> op::Tuple::get_tuple_element(size_t i) {
  return m_elements.at(i);
}

shared_ptr<ngraph::Node> op::get_tuple_element(shared_ptr<ngraph::Node> node,
                                               size_t i) {
  shared_ptr<op::Tuple> tuple = dynamic_pointer_cast<op::Tuple>(node);
  if (tuple == nullptr) {
    throw ngraph::ngraph_error("get_tuple_element called on a non-tuple");
  }
  return tuple->get_tuple_element(i);
}

namespace {
// Add the node to nodes if it's not a Tuple, otherwise add nodes for the
// elements of the tuple.
template <typename T>
void flatten(vector<shared_ptr<T>>& nodes, shared_ptr<ngraph::Node> node) {
  auto xla_tuple = dynamic_pointer_cast<op::Tuple>(node);
  if (xla_tuple == nullptr) {
    auto t_node = dynamic_pointer_cast<T>(node);
    if (t_node == nullptr) {
      throw ngraph::ngraph_error("Invalid node type type encountered");
    }
    nodes.push_back(t_node);
  } else {
    for (auto element : xla_tuple->get_elements()) {
      flatten<T>(nodes, element);
    }
  }
}

// Collect a vector of the non-Tuple nodes that underly nodes
template <typename T>
vector<shared_ptr<T>> flatten(const ngraph::NodeVector& nodes) {
  vector<shared_ptr<T>> result;
  for (auto node : nodes) {
    flatten<T>(result, node);
  }
  return result;
}
}

XLAFunction::XLAFunction(const ngraph::NodeVector& results,
                         const ngraph::NodeVector& parameters,
                         const string& name)
    : ngraph::Function(flatten<ngraph::Node>(results),
                       flatten<ngraph::op::Parameter>(parameters), name) {}

XLATuple::XLATuple(const XLAValues& elements)

    : ngraph::runtime::TensorView(
          make_shared<ngraph::descriptor::PrimaryTensorView>(
              make_shared<ngraph::TensorViewType>(ngraph::element::f32,
                                                  ngraph::Shape{}),
              "XLATuple")),
      m_elements(elements) {}

const vector<shared_ptr<ngraph::runtime::TensorView>>& XLATuple::get_elements()
    const {
  return m_elements;
}

size_t XLATuple::get_tuple_size() const { return m_elements.size(); }

shared_ptr<ngraph::runtime::TensorView> XLATuple::get_tuple_element(
    size_t i) const {
  return m_elements.at(i);
}

void XLATuple::write(const void* p, size_t tensor_offset, size_t n) {
  throw ngraph::ngraph_error("Cannot write to a tuple");
}

void XLATuple::read(void* p, size_t tensor_offset, size_t n) const {
  throw ngraph::ngraph_error("Cannot read from a tuple");
}

std::shared_ptr<ngraph::runtime::TensorView> get_tuple_element(
    std::shared_ptr<XLATuple> tuple, size_t i) {
  return tuple->get_tuple_element(i);
}

namespace {
// Collect the real tensors, expanding the tensors that are really tuples
void flatten(ngraph::runtime::TensorViewPtrs& tensors,
             shared_ptr<ngraph::runtime::TensorView> tensor) {
  auto xla_tuple = dynamic_pointer_cast<XLATuple>(tensor);
  if (xla_tuple == nullptr) {
    tensors.push_back(tensor);
  } else {
    for (auto element : xla_tuple->get_elements()) {
      flatten(tensors, element);
    }
  }
}
}

}  // namespace compat

}  // namespace ngraph_plugin

}  // namespace xla
